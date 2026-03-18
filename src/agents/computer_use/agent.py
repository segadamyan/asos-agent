"""
Unified computer-use agent facade.

Selects the correct provider-specific implementation (OpenAI or Gemini)
based on the ``config`` argument and delegates all work to it.

Usage::

    agent = ComputerUseAgent(ComputerUseConfig(provider="openai"))
    result = await agent.run("Search for Python tutorials", start_url="https://google.com")
    print(result)
    print(agent.usage)

    agent = ComputerUseAgent(ComputerUseConfig(provider="gemini"), headless=False)
    result = await agent.run("Find the top HN story", start_url="https://news.ycombinator.com")
"""

from __future__ import annotations

from typing import Optional

from agents.computer_use.base import (
    BaseComputerUseAgent,
    ComputerUseConfig,
    ComputerUseError,
    UsageSummary,
)
from agents.computer_use.executor import BaseScreenExecutor
from agents.computer_use.gemini import GeminiComputerUseAgent
from agents.computer_use.openai import OpenAIComputerUseAgent
from agents.utils.logs.config import logger


class ComputerUseAgent(BaseComputerUseAgent):
    """Provider-agnostic computer-use agent.

    Wraps :class:`OpenAIComputerUseAgent` and :class:`GeminiComputerUseAgent`
    behind a single interface.  Choose the provider via :class:`ComputerUseConfig`;
    everything else (``run``, ``usage``) is identical regardless of provider.

    Args:
        config: Provider and model selection.
        screen_executor: Optional :class:`BaseScreenExecutor`.  When omitted
            the agent launches its own headless Playwright browser.
        system_prompt: Optional system / developer instruction for the model.
        max_steps: Maximum computer-use turns before the loop stops.
        headless: Run the managed Playwright browser headlessly.
            Ignored when ``screen_executor`` is provided.
        screen_width: Browser viewport width in pixels.
        screen_height: Browser viewport height in pixels.
        screenshot_mime_type: ``"image/png"`` (default) or ``"image/jpeg"``.
        max_retries: Number of times to retry the primary provider on failure
            before giving up (or falling back).
        fallback_config: Optional provider config to use if the primary provider
            exhausts all retries.  Uses the same screen/prompt settings.

    Example::

        config = ComputerUseConfig(provider="openai")
        agent = ComputerUseAgent(
            config,
            fallback_config=ComputerUseConfig(provider="gemini"),
            max_retries=2,
        )
        result = await agent.run(
            "Go to Wikipedia and summarise the Python article.",
            start_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        )
        print(result)
        print(agent.usage)
    """

    def __init__(
        self,
        config: ComputerUseConfig,
        screen_executor: Optional[BaseScreenExecutor] = None,
        system_prompt: str = "",
        max_steps: int = 50,
        headless: bool = True,
        screen_width: int = OpenAIComputerUseAgent.SCREEN_WIDTH,
        screen_height: int = OpenAIComputerUseAgent.SCREEN_HEIGHT,
        screenshot_mime_type: str = "image/png",
        # Retry / fallback
        max_retries: int = 0,
        fallback_config: Optional[ComputerUseConfig] = None,
    ):
        self.config = config
        self._max_retries = max_retries
        self._fallback_config = fallback_config

        self._agent_kwargs = dict(
            screen_executor=screen_executor,
            system_prompt=system_prompt,
            max_steps=max_steps,
            headless=headless,
            screen_width=screen_width,
            screen_height=screen_height,
            screenshot_mime_type=screenshot_mime_type,
        )
        self._agent: BaseComputerUseAgent = self._build_agent(config)

    _REGISTRY = {
        "openai": OpenAIComputerUseAgent,
        "gemini": GeminiComputerUseAgent,
    }

    def _build_agent(self, config: ComputerUseConfig) -> BaseComputerUseAgent:
        agent_cls = self._REGISTRY.get(config.provider)
        if agent_cls is None:
            raise ValueError(f"Unknown provider {config.provider!r}. Choose {list(self._REGISTRY)!r}.")
        return agent_cls(model=config.model or agent_cls.DEFAULT_MODEL, **self._agent_kwargs)

    @property
    def usage(self) -> UsageSummary:
        return self._agent.usage

    async def run(self, task: str, start_url: str = "") -> str:
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._agent.run(task, start_url=start_url)
            except ComputerUseError as exc:
                last_exc = exc
                logger.warning(
                    "ComputerUseAgent [%s] attempt %d/%d failed: %s",
                    self.config.provider,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        if self._fallback_config is not None:
            logger.warning(
                "ComputerUseAgent falling back from %r to %r",
                self.config.provider,
                self._fallback_config.provider,
            )
            fallback_agent = self._build_agent(self._fallback_config)
            return await fallback_agent.run(task, start_url=start_url)

        raise last_exc
