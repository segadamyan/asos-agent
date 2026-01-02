from typing import Any, Dict, Optional


class AgentDelegate:
    """
    Manages delegation of tasks to specialized agents.

    This class maintains a registry of specialized agents and routes
    tasks to the appropriate agent based on the task type.
    """

    def __init__(self):
        """Initialize the agent delegate."""
        self._agents: Dict[str, Any] = {}

    def register_agent(self, agent_type: str, agent: Any):
        """
        Register a specialized agent.

        Args:
            agent_type: Type identifier for the agent (e.g., 'math', 'research', 'coding')
            agent: The agent instance
        """
        self._agents[agent_type] = agent

    def get_agent(self, agent_type: str) -> Optional[Any]:
        """
        Get a registered agent by type.

        Args:
            agent_type: Type identifier for the agent

        Returns:
            The agent instance if found, None otherwise
        """
        return self._agents.get(agent_type)

    def list_agents(self) -> list[str]:
        """
        List all registered agent types.

        Returns:
            List of agent type identifiers
        """
        return list(self._agents.keys())

    async def delegate_to_math_agent(self, problem: str) -> str:
        """
        Delegate a mathematical problem to the math agent.

        Args:
            problem: The mathematical problem to solve

        Returns:
            The solution from the math agent
        """
        math_agent = self._agents.get("math")
        if not math_agent:
            return "Error: Math agent not available"

        try:
            result = await math_agent.solve(problem)
            return result.content
        except Exception as e:
            return f"Error solving math problem: {str(e)}"

    async def delegate(self, agent_type: str, task: str) -> str:
        """
        Delegate a task to a specific agent type.

        Args:
            agent_type: Type of agent to delegate to
            task: The task description

        Returns:
            The result from the agent
        """
        agent = self._agents.get(agent_type)
        if not agent:
            return f"Error: Agent type '{agent_type}' not available. Available types: {', '.join(self.list_agents())}"

        try:
            result = await agent.solve(task)
            return result.content
        except Exception as e:
            return f"Error delegating to {agent_type} agent: {str(e)}"
