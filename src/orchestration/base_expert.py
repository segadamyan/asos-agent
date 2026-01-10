"""
Base Expert Agent

Abstract base class for specialized expert agents in the orchestration framework.
Supports MCP (Model Context Protocol) for dynamic tool discovery via the core Agent.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from agents.core.mcp import MCPServerConfig
from agents.core.agent import Agent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig, Message
from agents.tools.base import ToolDefinition


class BaseExpertAgent(ABC):
    """
    Abstract base class for specialized expert agents.

    Expert agents are domain-specific agents that can be registered with
    an Orchestrator to handle specialized tasks. Each expert agent wraps
    an Agent with a specialized system prompt and provides a unified
    interface through the solve() method.
    
    Supports MCP for dynamic tool discovery from external servers (via core Agent).
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        tools: Optional[List[ToolDefinition]] = None,
        default_temperature: float = 0.3,
        enable_mcp: bool = False,
        mcp_server_configs: Optional[List[MCPServerConfig]] = None,
    ):
        """
        Initialize the base expert agent.

        Args:
            name: Name of the expert agent
            system_prompt: System prompt defining the agent's expertise and behavior
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
            tools: List of tools available to this agent
            default_temperature: Default temperature for generation (lower for precision)
            enable_mcp: Whether to enable MCP tool discovery
            mcp_server_configs: List of MCP server configurations
        """
        self._name = name
        self._system_prompt = system_prompt
        self._default_temperature = default_temperature
        self._gbs = gbs or GenerationBehaviorSettings(temperature=default_temperature)
        self._tools = tools or []
        self._mcp_enabled = enable_mcp
        self._mcp_configs = mcp_server_configs or []

        # Set default provider if not specified
        if ip_config is None:
            ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")
        self._ip_config = ip_config

        # Format system prompt with current date if it contains the placeholder
        if "{current_date}" in system_prompt:
            self._system_prompt = system_prompt.format(current_date=datetime.today().strftime("%Y-%m-%d"))

        # Initialize agent with MCP support
        self.agent = Agent(
            name=self._name,
            system_prompt=self._system_prompt,
            history=History(),
            ip_config=self._ip_config,
            tools=self._tools,
            enable_mcp=self._mcp_enabled,
            mcp_server_configs=self._mcp_configs,
        )

    async def initialize(self) -> "BaseExpertAgent":
        """
        Initialize MCP connections and discover tools.
        
        Must be called before using the agent if MCP is enabled.
        For non-MCP agents, this is a no-op.
        
        Returns:
            self for method chaining
        """
        await self.agent.initialize_mcp()
        return self

    async def cleanup(self):
        """Clean up MCP connections."""
        await self.agent.cleanup_mcp()

    @abstractmethod
    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a problem or task using this expert agent.

        This is the main interface method that should be implemented by all
        expert agents. It provides a consistent way for the orchestrator to
        delegate tasks to specialized agents.

        Args:
            problem: The problem or question to solve
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the response from the expert agent
        """
        pass

    async def _ensure_initialized(self):
        """Ensure the agent is initialized (for MCP support)."""
        if self._mcp_enabled and not self.agent._mcp_initialized:
            await self.initialize()

    def clear_history(self):
        """Clear the conversation history."""
        self.agent.history.clear()

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    @property
    def gbs(self) -> GenerationBehaviorSettings:
        """Get the generation behavior settings."""
        return self._gbs

    @property
    def history(self) -> History:
        """Get the conversation history."""
        return self.agent.history

    @property
    def mcp_enabled(self) -> bool:
        """Check if MCP is enabled."""
        return self.agent.mcp_enabled

    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.agent.available_tools

    @property
    def connected_servers(self) -> List[str]:
        """Get list of connected MCP server names."""
        return self.agent.connected_mcp_servers

    @property
    def description(self) -> str:
        """
        Get a description of this expert agent's capabilities.

        This should be overridden by subclasses to provide specific information
        about what the agent specializes in.

        Returns:
            A string describing the agent's expertise
        """
        return f"{self._name}: A specialized expert agent"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()