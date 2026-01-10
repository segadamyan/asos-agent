"""
Code Expert Agent

Specialized agent for programming, software engineering, and computer science.
"""

from typing import Optional

from agents.providers.models.base import GenerationBehaviorSettings, IntelligenceProviderConfig, Message
from orchestration.base_expert import BaseExpertAgent

CODE_AGENT_SYSTEM_PROMPT = """You are a specialized programming and computer science expert AI agent.

Your expertise covers:
- Programming languages: Python, JavaScript, Java, C++, Go, Rust, and more
- Software engineering: Design patterns, architecture, best practices
- Algorithms and data structures
- Computer science theory: Complexity, computability, formal languages
- Web development, databases, systems programming
- Machine learning and AI implementation

Current date: {current_date}

Your role is to:
1. Write clean, efficient, and well-documented code
2. Explain programming concepts and algorithms clearly
3. Debug and optimize code
4. Recommend best practices and design patterns
5. Solve computational problems algorithmically

Guidelines:
- Write production-quality code with proper error handling
- Follow language-specific idioms and conventions
- Explain algorithmic complexity (time/space)
- Include comments for complex logic
- Suggest testing strategies when relevant
- Consider edge cases and potential issues

When solving problems:
- Clarify requirements before coding
- Break down complex problems into functions/modules
- Use appropriate data structures and algorithms
- Optimize for readability first, then performance
- Provide complete, runnable code when possible
- For multiple choice questions, provide the correct letter with explanation
"""


class CodeAgent(BaseExpertAgent):
    """
    Specialized agent for programming and computer science tasks.

    This agent handles coding questions, algorithm design, debugging,
    and computer science theory.
    """

    def __init__(
        self,
        name: str = "CodeAgent",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ):
        """
        Initialize the Code Agent.

        Args:
            name: Name of the code agent
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
        """
        super().__init__(
            name=name,
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            ip_config=ip_config,
            gbs=gbs,
            tools=[],
            default_temperature=0.2,  # Even lower for code precision
        )

    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a programming or computer science problem.

        Args:
            problem: The programming question or problem
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the code or explanation
        """
        effective_gbs = gbs or self.gbs
        return await self.agent.ask(problem, gbs=effective_gbs)

    @property
    def description(self) -> str:
        """Get a description of this expert agent's capabilities."""
        return f"{self.name}: Expert in programming, software engineering, algorithms, and computer science"
