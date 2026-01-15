"""
Science Expert Agent

Specialized agent for scientific questions across physics, chemistry, and biology.
"""

from typing import Optional

from agents.providers.models.base import (
    GenerationBehaviorSettings,
    IntelligenceProviderConfig,
    Message,
)
from orchestration.base_expert import BaseExpertAgent
from tools.chemistry_tools import get_chemistry_tools
from tools.execution_tools import get_execution_tools
from tools.medical_tools import get_medical_tools
from tools.physics_tools import get_physics_tools

SCIENCE_AGENT_SYSTEM_PROMPT = """You are a specialized science expert AI agent.

Your expertise covers:
- Physics: Classical mechanics, quantum mechanics, thermodynamics, electromagnetism
- Chemistry: Organic, inorganic, physical chemistry, biochemistry
- Biology: Molecular biology, genetics, ecology, physiology

Current date: {current_date}

Your role is to:
1. Answer scientific questions accurately and comprehensively
2. Explain complex scientific concepts in clear terms
3. Provide evidence-based responses grounded in scientific principles
4. Reference relevant scientific laws, theories, and experimental evidence
5. Clarify common misconceptions

Guidelines:
- Use precise scientific terminology
- Cite fundamental principles when explaining phenomena
- Show calculations or derivations when relevant
- Acknowledge limitations of current scientific understanding
- Distinguish between established facts and theoretical models

Tool usage:
- Use python_executor for multi-step scientific calculations, numerical checks, or quick simulations.
- Include units and show intermediate steps when useful.

When answering:
- Start with a clear, direct answer
- Provide scientific reasoning and explanation
- Use examples or analogies when helpful
- Include units and proper notation
- For multiple choice questions, provide the correct letter with explanation
"""


class ScienceAgent(BaseExpertAgent):
    """
    Specialized agent for scientific questions and problem-solving.

    This agent handles questions across physics, chemistry, biology, and
    related scientific domains.
    """

    def __init__(
        self,
        name: str = "ScienceAgent",
        ip_config: Optional[IntelligenceProviderConfig] = None,
        gbs: Optional[GenerationBehaviorSettings] = None,
    ):
        """
        Initialize the Science Agent.

        Args:
            name: Name of the science agent
            ip_config: Intelligence provider configuration
            gbs: Generation behavior settings
        """
        science_tools = []
        science_tools.extend(get_physics_tools())
        science_tools.extend(get_chemistry_tools())
        science_tools.extend(get_medical_tools())
        science_tools.extend(get_execution_tools())

        super().__init__(
            name=name,
            system_prompt=SCIENCE_AGENT_SYSTEM_PROMPT,
            ip_config=ip_config,
            gbs=gbs,
            tools=science_tools,
            default_temperature=0.3,
        )

    async def solve(self, problem: str, gbs: Optional[GenerationBehaviorSettings] = None) -> Message:
        """
        Solve a scientific question or problem.

        Args:
            problem: The scientific question or problem
            gbs: Optional generation behavior settings to override defaults

        Returns:
            Message containing the scientific response
        """
        effective_gbs = gbs or self.gbs
        return await self.agent.ask(problem, gbs=effective_gbs)

    @property
    def description(self) -> str:
        """Get a description of this expert agent's capabilities."""
        return f"{self.name}: Expert in physics, chemistry, biology, and other scientific domains"
