"""
Agent serves as the core component for AI interactions within the system:
- Abstracts communication with different LLM providers
- Manages conversation history and context window
- Handles tool invocation and error recovery
- Provides token usage tracking and management
- Supports synchronous conversation flows with tool execution
- Supports MCP (Model Context Protocol) for dynamic tool discovery
"""

import asyncio
from typing import List, Optional

from agents.core.base import BaseAgent
from agents.core.mcp import MCPDiscovery, MCPServerConfig
from agents.core.tool_executor import ToolExecutor
from agents.providers.factory import ProviderFactory
from agents.providers.models.base import (
    GenerationBehaviorSettings,
    History,
    IntelligenceProviderConfig,
    Message,
    RoleEnum,
)
from agents.providers.models.exceptions import ProviderFailureError
from agents.tools.base import ToolDefinition
from agents.tools.invoker import ToolInvoker
from agents.utils.logs.config import logger


class Agent(BaseAgent):
    """
    This agent manages conversations with LLM providers, handles tool execution,
    and maintains conversation history within token limits.

    Supports MCP (Model Context Protocol) for dynamic tool discovery from external servers.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        history: History,
        ip_config: IntelligenceProviderConfig,
        tools: List[ToolDefinition] = None,
        _gbs: Optional[GenerationBehaviorSettings] = None,
        enable_parallel_execution: bool = True,
        max_invocations_count: Optional[int] = 10,
        fallback_ip_configs: Optional[List[IntelligenceProviderConfig]] = None,
        max_provider_retries: int = 2,
        enable_mcp: bool = False,
        mcp_server_configs: Optional[List[MCPServerConfig]] = None,
    ):
        # Core agent properties
        self.name = name
        self.history = history
        self._gbs = _gbs
        self._system_prompt = system_prompt

        self.providers = [ip_config]
        if fallback_ip_configs:
            self.providers.extend(fallback_ip_configs)

        self.max_provider_retries = max_provider_retries
        self._loop_failed_attempts = 0

        self._current_provider_index = 0
        self._current_provider_failures = 0

        # Initialize components
        self.enable_parallel_execution = enable_parallel_execution
        self.invocations_count = 0
        self.max_invocations_count = max_invocations_count

        self._tools = tools or []

        # MCP support
        self._mcp_enabled = enable_mcp
        self._mcp_configs = mcp_server_configs or []
        self._mcp_discovery: Optional[MCPDiscovery] = None
        self._mcp_initialized = False

        self._invoker = ToolInvoker(self._tools)
        self.tool_executor = ToolExecutor(self._invoker, self.name)

    async def initialize_mcp(self) -> "Agent":
        """
        Initialize MCP connections and discover tools.

        Must be called before using the agent if MCP is enabled.
        For non-MCP agents, this is a no-op.

        Returns:
            self for method chaining
        """
        if self._mcp_initialized or not self._mcp_enabled:
            return self

        if self._mcp_configs:
            self._mcp_discovery = MCPDiscovery()
            try:
                mcp_tools = await self._mcp_discovery.discover(self._mcp_configs)
                self._tools.extend(mcp_tools)
                self._invoker = ToolInvoker(self._tools)
                self.tool_executor = ToolExecutor(self._invoker, self.name)
                logger.info(f"Discovered {len(mcp_tools)} MCP tools")
            except Exception as exc:
                logger.exception("Failed to discover MCP tools", exc_info=exc)
                raise

        self._mcp_initialized = True
        return self

    async def cleanup_mcp(self) -> None:
        """Clean up MCP connections."""
        if self._mcp_discovery:
            await self._mcp_discovery.cleanup()
            self._mcp_discovery = None
        self._mcp_initialized = False

    async def add_mcp_servers(self, mcp_configs: List[MCPServerConfig]) -> None:
        """
        Add MCP servers and discover their tools dynamically.

        Args:
            mcp_configs: List of MCP server configurations to add
        """
        if not self._mcp_discovery:
            self._mcp_discovery = MCPDiscovery()

        try:
            mcp_tools = await self._mcp_discovery.discover(mcp_configs)
            self._tools.extend(mcp_tools)
            self._invoker = ToolInvoker(self._tools)
            self.tool_executor = ToolExecutor(self._invoker, self.name)
            self._mcp_configs.extend(mcp_configs)
            logger.info(f"Added {len(mcp_tools)} MCP tools from {len(mcp_configs)} servers")
        except Exception as exc:
            logger.exception("Failed to discover MCP tools", exc_info=exc)
            raise

    async def add_tools(self, tools: List[ToolDefinition]) -> None:
        """Add tools to the agent"""
        self._tools.extend(tools)

        self._invoker = ToolInvoker(self._tools)
        self.tool_executor = ToolExecutor(self._invoker, self.name)

    async def __aenter__(self) -> "Agent":
        """Async context manager entry - initializes MCP if enabled."""
        await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleans up MCP connections."""
        await self.cleanup_mcp()

    @property
    def mcp_enabled(self) -> bool:
        """Check if MCP is enabled."""
        return self._mcp_enabled

    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [t.name for t in self._tools]

    @property
    def connected_mcp_servers(self) -> List[str]:
        """Get list of connected MCP server names."""
        if self._mcp_discovery:
            return self._mcp_discovery.connected_servers
        return []

    async def _create_provider(self):
        """Create provider instance"""
        if self._current_provider_index >= len(self.providers):
            self._current_provider_index = len(self.providers) - 1
        config = self.providers[self._current_provider_index]
        return ProviderFactory().create(self._system_prompt, config, self._tools)

    def _handle_provider_failure(self, first_provider_exception: ProviderFailureError):
        """Handle provider failure and move to next provider if needed"""
        self._current_provider_failures += 1

        if self._current_provider_failures >= self.max_provider_retries:
            self._current_provider_index += 1
            self._current_provider_failures = 0

            if self._current_provider_index < len(self.providers):
                next_provider = self.providers[self._current_provider_index]
                logger.info(f"Switching to provider: {next_provider.provider_name}")
            else:
                logger.error("All providers exhausted")
                raise first_provider_exception

    async def ask(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Main method to process a user query and return an AI response.
        Handles provider failover automatically.
        """

        gbs = gbs or self._gbs
        self.history.add_user_message(query)

        message = None
        self._loop_failed_attempts = 0
        first_provider_exception = None

        while self._current_provider_index < len(self.providers) and not message:
            ip = await self._create_provider()
            try:
                message = await self._conversation_loop(ip, gbs, query)

                if self._current_provider_index > 0:
                    self._current_provider_index = 0
                    self._current_provider_failures = 0
                    logger.info("Successfully used fallback, resetting to primary provider")

                break
            except ProviderFailureError as exc:
                logger.exception(f"Provider failure for query {query}: {exc}")
                if first_provider_exception is None and self._current_provider_index == 0:
                    first_provider_exception = exc

                self._handle_provider_failure(first_provider_exception)

        if not message:
            ip = await self._create_provider()
            message = await self._handle_failure_recovery(ip, gbs)

        return message

    async def ask_and_clear_history(self, search_query: str, gbs: Optional[GenerationBehaviorSettings] = None) -> str:
        """Answer a query and clear the conversation history"""
        message = await self.ask(search_query, gbs)
        self.history.clear()
        return message.content

    async def _execute_tools_parallel(self, message: Message) -> List:
        """Execute multiple tool calls in parallel"""
        if not message.tool_calls:
            return []

        tasks = []
        for tool_call in message.tool_calls:
            task = asyncio.create_task(self.tool_executor.invoke_single_tool(tool_call))
            tasks.append(task)

        tool_results = await asyncio.gather(*tasks)
        return tool_results

    async def fork(self, keep_history: bool = True, keep_tools: bool = True) -> "Agent":
        """Create a fork of the agent with the same configuration"""
        primary_config = self.providers[0]
        fallback_configs = self.providers[1:] if len(self.providers) > 1 else None

        return Agent(
            name=self.name,
            system_prompt=self._system_prompt,
            history=self.history.model_copy(deep=True) if keep_history else History(),
            ip_config=primary_config.model_copy(deep=True),
            tools=self._tools.copy() if keep_tools else None,
            _gbs=self._gbs.model_copy(deep=True) if self._gbs else None,
            enable_parallel_execution=self.enable_parallel_execution,
            max_invocations_count=self.max_invocations_count,
            fallback_ip_configs=[config.model_copy(deep=True) for config in fallback_configs]
            if fallback_configs
            else None,
            max_provider_retries=self.max_provider_retries,
        )

    async def fork_exec(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """Fork the agent and execute a query asynchronously"""
        forked_agent = await self.fork()
        return await forked_agent.ask(query, gbs)

    async def _execute_tools(self, message: Message) -> List:
        """Execute tool calls either in parallel or sequentially based on configuration"""
        if self.enable_parallel_execution and len(message.tool_calls) > 1:
            return await self._execute_tools_parallel(message)
        else:
            return await self.tool_executor.invoke_tools(message)

    async def _conversation_loop(self, ip, gbs, query: str) -> Optional[Message]:
        """Main conversation loop with tool execution and error handling."""
        max_attempts = 5

        while self._loop_failed_attempts < max_attempts:
            # Get response from LLM
            message = await ip.get_response(history=self.history, gbs=gbs)
            if not message.tool_calls:
                if message.content:
                    self.history.add_message(message)
                return message

            self.invocations_count += len(message.tool_calls)

            if self.max_invocations_count is not None and self.invocations_count >= self.max_invocations_count:
                return await self._handle_invocation_limit_recovery(ip, gbs)

            tool_results = await self._execute_tools(message)
            self.history.add_message(message)
            tool_result_message = Message(
                content="",
                role=RoleEnum.TOOL,
                message_type="tool_result",
                tool_call_results=tool_results,
            )
            self.history.add_message(tool_result_message)

            any_failures = any(not result.success for result in tool_results)
            if any_failures:
                self._loop_failed_attempts += 1
            else:
                self._loop_failed_attempts = 0

        return None

    async def _handle_failure_recovery(self, ip, gbs) -> Message:
        """Handle recovery when tool calls fail too many times"""
        recovery_message = (
            "Tool calls failed too many times. Please describe what you tried to do to fulfill the user request."
        )

        self.history.add_user_message(recovery_message)
        message = await ip.get_response(self.history, gbs=gbs)
        if message.tool_calls:
            logger.warning("Made tool call during recovery, clearing tool calls")
            message.tool_calls.clear()

        self.history.add_message(message)
        return message

    async def _handle_invocation_limit_recovery(self, ip, gbs) -> Message:
        remaining_calls = self.invocations_count - self.max_invocations_count
        recovery_message = (
            f"You have reached the iteration limit. Please solve the problem using {remaining_calls} tool calls."
        )
        self.history.add_user_message(recovery_message)

        recovery_response = await ip.get_response(self.history, gbs=gbs)

        if recovery_response.tool_calls:
            tool_results = await self._execute_tools(recovery_response)
            self.history.add_message(recovery_response)
            tool_result_message = Message(
                content="",
                role=RoleEnum.TOOL,
                message_type="tool_result",
                tool_call_results=tool_results,
            )
            self.history.add_message(tool_result_message)
            final_message = await ip.get_response(self.history, gbs=gbs)

            if final_message.tool_calls:
                final_recovery_message = "Do not call any more tools. Just summarize the output and provide your final answer based on the available information."
                self.history.add_user_message(final_recovery_message)
                final_message = await ip.get_response(self.history, gbs=gbs)
                final_message.tool_calls = []

            self.history.add_message(final_message)
            return final_message
        else:
            self.history.add_message(recovery_response)
            return recovery_response
