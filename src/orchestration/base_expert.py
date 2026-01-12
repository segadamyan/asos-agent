"""
Base Expert Agent

Abstract base class for specialized expert agents in the orchestration framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from agents.core.agent import Agent
from agents.core.mcp import MCPServerConfig
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig, Message
from agents.tools.base import ToolDefinition


class BaseExpertAgent(ABC):
    """
    Abstract base class for specialized expert agents.

    Expert agents are domain-specific agents that can be registered with
    an Orchestrator to handle specialized tasks. Each expert agent wraps
    an Agent with a specialized system prompt and provides a unified
    interface through the solve() method.
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
        self._default_temperature = default_temperature
        self._gbs = gbs or GenerationBehaviorSettings(temperature=default_temperature)

        # Set default provider if not specified
        if ip_config is None:
            ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

        # Format system prompt with current date if it contains the placeholder
        if "{current_date}" in system_prompt:
            system_prompt = system_prompt.format(current_date=datetime.today().strftime("%Y-%m-%d"))

        # Initialize agent with MCP support (delegated to core Agent)
        self.agent = Agent(
            name=self._name,
            system_prompt=system_prompt,
            history=History(),
            ip_config=ip_config,
            tools=tools or [],
            enable_mcp=enable_mcp,
            mcp_server_configs=mcp_server_configs,
        )

    @abstractmethod
    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a problem or task using this expert agent.
        """
        pass

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
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.agent.available_tools

    @property
    def description(self) -> str:
        """
        Get a description of this expert agent's capabilities.
        """
        return f"{self._name}: A specialized expert agent"

    async def __aenter__(self):
        """Async context manager entry - delegates to Agent."""
        await self.agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - delegates to Agent."""
        await self.agent.__aexit__(exc_type, exc_val, exc_tb)
