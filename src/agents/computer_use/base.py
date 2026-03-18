"""
Base interface for computer-use agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel


ComputerUseProvider = Literal["openai", "gemini"]


class ComputerUseError(Exception):
    """Base exception for all computer-use agent errors."""


class GeminiComputerUseError(ComputerUseError):
    """Raised when the Gemini Computer Use loop encounters a fatal error."""

    def __init__(self, message: str, step: int, block_reason: str = "") -> None:
        super().__init__(message)
        self.step = step
        self.block_reason = block_reason


class OpenAIComputerUseError(ComputerUseError):
    """Raised when the OpenAI Computer Use loop encounters a fatal error."""

    def __init__(self, message: str, step: int) -> None:
        super().__init__(message)
        self.step = step


class ComputerUseConfig(BaseModel):
    """Provider and model selection for :class:`~agents.computer_use.agent.ComputerUseAgent`.

    Example::

        config = ComputerUseConfig(provider="openai")
        config = ComputerUseConfig(provider="gemini", model="gemini-3-flash-preview")
    """

    provider: ComputerUseProvider
    model: Optional[str] = None


@dataclass
class UsageSummary:
    """Accumulated token usage and cost for a single :meth:`BaseComputerUseAgent.run` call."""

    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __str__(self) -> str:
        return (
            f"steps={self.steps} | "
            f"tokens={self.total_tokens:,} "
            f"(in={self.input_tokens:,} out={self.output_tokens:,}) | "
            f"cost=${self.cost_usd:.6f}"
        )


class BaseComputerUseAgent(ABC):
    """Abstract base for computer-use agents.

    Concrete subclasses drive a model→action→screenshot loop that continues
    until the task is complete or the step limit is reached.

    After :meth:`run` returns, :attr:`usage` contains the accumulated token
    counts and estimated cost for that run.

    Subclasses must implement :meth:`run`.
    """

    SCREEN_WIDTH = 1440
    SCREEN_HEIGHT = 900

    @property
    def usage(self) -> UsageSummary:
        """Token usage and cost from the most recent :meth:`run` call."""
        return getattr(self, "_usage", UsageSummary())

    @abstractmethod
    async def run(self, task: str, start_url: str = "") -> str:
        """Execute the computer-use loop for *task*.

        Args:
            task: Natural-language instruction for the agent.
            start_url: Optional URL to navigate to before the loop starts.
                Only used when the agent manages its own browser (no custom
                ``screen_executor`` was provided).

        Returns:
            The model's final text answer after the loop completes.
        """
        raise NotImplementedError
