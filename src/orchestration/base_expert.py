"""
Base Expert Agent

Abstract base class for specialized expert agents in the orchestration framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from agents.core.agent import Agent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig, Message


class BaseExpertAgent(ABC):
    """
    Abstract base class for specialized expert agents.

    Expert agents are domain-specific agents that can be registered with
    an Orchestrator to handle specialized tasks. Each expert agent wraps
    an Agent with a specialized system prompt and provides a unified
    interface through the execute() method.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        tools: Optional[list] = None,
        default_temperature: float = 0.3,
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
        """
        self.name = name
        self.gbs = gbs or GenerationBehaviorSettings(temperature=default_temperature)

        # Set default provider if not specified
        if ip_config is None:
            ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

        # Format system prompt with current date if it contains the placeholder
        if "{current_date}" in system_prompt:
            system_prompt = system_prompt.format(current_date=datetime.today().strftime("%Y-%m-%d"))

        # Create the underlying Agent
        self.agent = Agent(
            name=name,
            system_prompt=system_prompt,
            history=History(),
            ip_config=ip_config,
            tools=tools or [],
        )

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

    def clear_history(self):
        """Clear the conversation history."""
        self.agent.history.clear()

    @property
    def history(self) -> History:
        """Get the conversation history."""
        return self.agent.history

    @property
    def description(self) -> str:
        """
        Get a description of this expert agent's capabilities.

        This should be overridden by subclasses to provide specific information
        about what the agent specializes in.

        Returns:
            A string describing the agent's expertise
        """
        return f"{self.name}: A specialized expert agent"
