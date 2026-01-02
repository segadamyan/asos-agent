"""
Orchestrator Module

This module defines the main Orchestrator agent that coordinates and manages
task execution, tool usage, and multi-step workflows.
"""

from datetime import datetime
from typing import Dict, Optional

from agents.core.simple import SimpleAgent
from agents.providers.models.base import GenerationBehaviorSettings, History, IntelligenceProviderConfig, Message
from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger
from orchestration.base_expert import BaseExpertAgent

ORCHESTRATOR_SYSTEM_PROMPT = """You are an advanced AI orchestrator responsible for coordinating complex tasks and workflows.

Your role is to:
1. Analyze user requests and break them down into manageable steps
2. Coordinate the use of available tools to accomplish tasks efficiently
3. Maintain context across multi-step workflows
4. Provide clear, accurate, and helpful responses
5. Handle errors gracefully and adapt your approach when needed

Current date: {current_date}

Guidelines:
- Always think through the problem before acting
- Use tools when they can help accomplish the task more effectively
- Be precise and accurate in your responses
- If you're unsure about something, acknowledge it rather than guessing
- Maintain a professional and helpful demeanor

When using tools:
- Call them with correct parameters
- Interpret their results accurately
- Chain multiple tool calls when necessary to accomplish complex tasks
"""


class Orchestrator:
    """
    Main orchestrator agent for coordinating tasks and workflows.

    The Orchestrator uses a SimpleAgent with a specialized system prompt
    and can delegate tasks to specialized agents.
    """

    def __init__(
        self,
        name: str = "Orchestrator",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
        additional_prompt: str = "",
    ):
        """
        Initialize the Orchestrator.

        Args:
            name: Name of the orchestrator agent
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
            additional_prompt: Additional instructions to append to system prompt
        """
        self.name = name
        self.gbs = gbs or GenerationBehaviorSettings(temperature=0.7)

        # Registry for specialized agents
        self._agents: Dict[str, BaseExpertAgent] = {}

        self.tools = [self._create_delegate_tool()]

        if ip_config is None:
            ip_config = IntelligenceProviderConfig(provider_name="openai", version="qwen/qwen3-next-80b-a3b-instruct")

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(current_date=datetime.today().strftime("%Y-%m-%d"))

        if additional_prompt:
            system_prompt += f"\n\nAdditional Instructions:\n{additional_prompt}"

        self.agent = SimpleAgent(
            name=name,
            system_prompt=system_prompt,
            history=History(),
            ip_config=ip_config,
            tools=self.tools,
        )

    def _create_delegate_tool(self) -> ToolDefinition:
        """Create the delegation tool for routing tasks to specialized agents."""

        async def delegate_task(agent_type: str, task: str) -> str:
            """
            Delegate a task to a specialized agent.

            Args:
                agent_type: Type of agent to delegate to (e.g., 'math', 'research', 'coding')
                task: The task description to delegate

            Returns:
                The result from the specialized agent
            """

            agent = self._agents.get(agent_type)
            if not agent:
                return (
                    f"Error: Agent type '{agent_type}' not available. Available types: {', '.join(self.list_agents())}"
                )

            try:
                logger.info(f"Delegating to {agent_type} agent...")

                result = await agent.solve(task)
                content = result.content

                logger.info(f"Delegation to {agent_type} completed, returning {len(content)} characters")
                return content
            except Exception as e:
                logger.error(f"Error during delegation to {agent_type}: {str(e)}")
                return f"Error delegating to {agent_type} agent: {str(e)}"

        return ToolDefinition(
            name="delegate_task",
            description="Delegate tasks to specialized agents. Use this when a task requires domain-specific expertise.",
            args_description={
                "agent_type": "Type of specialized agent (e.g., 'math' for mathematical problems)",
                "task": "The task or problem to delegate to the specialized agent",
            },
            tool=delegate_task,
        )

    async def execute(self, query: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Execute a task using the orchestrator.

        Args:
            query: The user's query or task description
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the orchestrator's response
        """
        effective_gbs = gbs or self.gbs
        return await self.agent.answer_to(query, gbs=effective_gbs)

    def register_agent(self, agent_type: str, agent):
        """
        Register a specialized agent.

        Args:
            agent_type: Type identifier for the agent (e.g., 'math', 'research', 'coding')
            agent: The specialized agent instance
        """
        self._agents[agent_type] = agent

    def list_agents(self) -> list[str]:
        """
        List all registered specialized agents.

        Returns:
            List of agent type identifiers
        """
        return list(self._agents.keys())

    def clear_history(self):
        """Clear the conversation history."""
        self.agent.history.clear()

    @property
    def history(self) -> History:
        """Get the conversation history."""
        return self.agent.history
