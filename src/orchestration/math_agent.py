"""
Math Agent

Specialized agent for mathematical calculations and problem-solving.
"""

from typing import Optional

from agents.providers.models.base import GenerationBehaviorSettings, IntelligenceProviderConfig, Message
from orchestration.base_expert import BaseExpertAgent

MATH_AGENT_SYSTEM_PROMPT = """You are a specialized mathematics expert AI agent.

Your role is to:
1. Solve mathematical problems accurately and efficiently
2. Explain mathematical concepts clearly
3. Show step-by-step solutions when appropriate
4. Handle arithmetic, algebra, calculus, statistics, and other mathematical domains
5. Verify calculations and provide confidence in your answers

Current date: {current_date}

Guidelines:
- Always double-check your calculations
- Show your work for complex problems
- Use proper mathematical notation
- Explain reasoning behind each step
- If a problem is ambiguous, ask for clarification
- State any assumptions you make

When solving problems:
- Break down complex problems into manageable steps
- Verify intermediate results
- Provide final answers in a clear format
- Include units when applicable

IMPORTANT: When answering multiple choice questions, provide the correct answer letter clearly and include a brief explanation.
"""


class MathAgent(BaseExpertAgent):
    """
    Specialized agent for mathematical tasks and calculations.

    This agent is optimized for solving mathematical problems, performing
    calculations, and explaining mathematical concepts.
    """

    def __init__(
        self,
        name: str = "MathAgent",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ):
        """
        Initialize the Math Agent.

        Args:
            name: Name of the math agent
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
        """
        super().__init__(
            name=name,
            system_prompt=MATH_AGENT_SYSTEM_PROMPT,
            ip_config=ip_config,
            gbs=gbs,
            tools=[],
            default_temperature=0.3,  # Lower temp for precision
        )

    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a mathematical problem.

        Args:
            problem: The mathematical problem or question
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the solution
        """
        effective_gbs = gbs or self.gbs
        return await self.agent.ask(problem, gbs=effective_gbs)

    @property
    def description(self) -> str:
        """Get a description of this expert agent's capabilities."""
        return (
            f"{self.name}: Expert in mathematics including arithmetic, algebra, "
            "calculus, statistics, and other mathematical domains"
        )
