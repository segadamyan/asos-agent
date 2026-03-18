"""
OpenAI Computer Use Agent

Implements the OpenAI built-in computer-use loop (Option 1 from the docs):
  1. Send a task to the model with the `computer` tool enabled.
  2. Inspect the returned `computer_call` item.
  3. Translate actions → canonical BaseAction list via _OpenAIActionTranslator.
  4. Run actions via the BaseScreenExecutor; capture screenshot.
  5. Send screenshot back as `computer_call_output` using `previous_response_id`.
  6. Repeat until the model stops returning `computer_call`.

Provider-specific translation is fully contained in _OpenAIActionTranslator.
The rest of the loop works with canonical actions and BaseScreenExecutor only.
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional

from openai import AsyncOpenAI

from agents.computer_use.actions import (
    BaseAction,
    ClickAction,
    DoubleClickAction,
    DragAction,
    ExecutorResult,
    MoveAction,
    NavigateAction,
    KeyPressAction,
    ScreenshotAction,
    ScrollAction,
    TypeAction,
    WaitAction,
    normalize_key,
)
from agents.computer_use.base import BaseComputerUseAgent, ComputerUseError, OpenAIComputerUseError, UsageSummary
from agents.computer_use.executor import BaseScreenExecutor, PlaywrightExecutor
from agents.config import LLM_HTTP_TIMEOUT
from agents.config.models import ModelCapability, model_registry
from agents.providers.models.token_usage import TokenCostEntry, get_token_cost_ledger
from agents.providers.openai import OpenAIUsageLogEntry
from agents.utils.logs.config import logger


class _OpenAIActionTranslator:
    """Translates raw OpenAI ``computer_call.actions`` into canonical actions."""

    @staticmethod
    def translate(raw_actions) -> List[BaseAction]:
        result: List[BaseAction] = []
        for action in raw_actions:
            if isinstance(action, dict):
                get = action.get
            else:
                get = lambda k, d=None: getattr(action, k, d)  # noqa: E731

            t = get("type")
            match t:
                case "click":
                    result.append(ClickAction(x=get("x"), y=get("y"), button=get("button", "left")))
                case "double_click":
                    result.append(DoubleClickAction(x=get("x"), y=get("y"), button=get("button", "left")))
                case "type":
                    result.append(TypeAction(text=get("text")))
                case "keypress":
                    result.append(KeyPressAction(keys=[normalize_key(k) for k in (get("keys") or [])]))
                case "scroll":
                    result.append(
                        ScrollAction(
                            x=get("x", 0),
                            y=get("y", 0),
                            scroll_x=get("scrollX", 0),
                            scroll_y=get("scrollY", 0),
                        )
                    )
                case "drag":
                    path = get("path") or []
                    if len(path) >= 2:
                        result.append(
                            DragAction(
                                start_x=path[0]["x"],
                                start_y=path[0]["y"],
                                end_x=path[-1]["x"],
                                end_y=path[-1]["y"],
                            )
                        )
                case "move":
                    result.append(MoveAction(x=get("x", 0), y=get("y", 0)))
                case "wait":
                    result.append(WaitAction(seconds=2.0))
                case "screenshot":
                    result.append(ScreenshotAction())
                case _:
                    logger.debug("_OpenAIActionTranslator: unknown action type '%s'", t)
        return result


class OpenAIComputerUseAgent(BaseComputerUseAgent):
    """Agent implementing the OpenAI built-in computer-use loop.

    When ``screen_executor`` is omitted the agent launches its own Playwright
    Chromium browser, drives it internally, and closes it when :meth:`run`
    returns.  Pass ``headless=False`` to watch the browser.

    Example — default executor (no setup needed)::

        agent = OpenAIComputerUseAgent(headless=False)
        result = await agent.run(
            "Search for the latest AI news.",
            start_url="https://www.google.com",
        )
        print(result)

    Example — custom executor::

        class MyVNCExecutor(BaseScreenExecutor):
            async def execute(self, actions):
                for action in actions:
                    if isinstance(action, ClickAction):
                        vnc.click(action.x, action.y)
                    # …
                return ExecutorResult(screenshot=vnc.screenshot())

        agent = OpenAIComputerUseAgent(screen_executor=MyVNCExecutor())
        result = await agent.run("Open the settings menu.")
    """

    DEFAULT_MODEL = "gpt-5.4"

    def __init__(
        self,
        screen_executor: Optional[BaseScreenExecutor] = None,
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        max_steps: int = 50,
        headless: bool = True,
        screen_width: int = BaseComputerUseAgent.SCREEN_WIDTH,
        screen_height: int = BaseComputerUseAgent.SCREEN_HEIGHT,
        screenshot_mime_type: str = "image/png",
    ):
        """
        Args:
            screen_executor: Optional :class:`BaseScreenExecutor` for UI interaction.
                When omitted a default Playwright executor is used.
            system_prompt: Optional developer instructions for the model.
            model: OpenAI model that supports the ``computer`` tool.
            max_steps: Maximum computer-use turns before the loop stops.
            headless: Run the default Playwright browser headlessly.
                Ignored when a custom ``screen_executor`` is provided.
            screen_width: Browser viewport width in pixels.
            screen_height: Browser viewport height in pixels.
            screenshot_mime_type: ``"image/png"`` (default) or ``"image/jpeg"``.
        """
        model_config = model_registry.find_model_by_alias(model)
        if model_config is not None and ModelCapability.COMPUTER_USE not in model_config.capabilities:
            raise ComputerUseError(
                f"Model '{model}' does not support computer_use. "
                f"Capabilities: {[c.value for c in model_config.capabilities]}"
            )

        self._screen_executor = screen_executor
        self.system_prompt = system_prompt
        self.model = model
        self.max_steps = max_steps
        self.headless = headless
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screenshot_mime_type = screenshot_mime_type

    async def run(self, task: str, start_url: str = "") -> str:
        """Run the computer-use loop for *task*, returning the final text output."""
        if self._screen_executor is not None:
            return await self._run_loop(task, self._screen_executor)

        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[f"--window-size={self.screen_width},{self.screen_height}"],
            )
            context = await browser.new_context(
                viewport={"width": self.screen_width, "height": self.screen_height},
                locale="en-US",
            )
            page = await context.new_page()
            if start_url:
                await page.goto(start_url, wait_until="domcontentloaded")
            try:
                executor = PlaywrightExecutor(page, screenshot_mime_type=self.screenshot_mime_type)
                return await self._run_loop(task, executor)
            finally:
                await browser.close()

    async def _run_loop(self, task: str, executor: BaseScreenExecutor) -> str:
        self._usage = UsageSummary()

        async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=LLM_HTTP_TIMEOUT) as client:
            create_kwargs: dict = {
                "model": self.model,
                "tools": [{"type": "computer"}],
                "input": task,
            }
            if self.system_prompt:
                create_kwargs["instructions"] = self.system_prompt

            try:
                response = await client.responses.create(**create_kwargs)
            except Exception as exc:
                raise OpenAIComputerUseError(f"API call failed at step 0 — {exc}", step=0) from exc
            self._record_usage(response)

            for step in range(self.max_steps):
                computer_call = next(
                    (item for item in response.output if item.type == "computer_call"),
                    None,
                )
                if computer_call is None:
                    break

                canonical_actions = _OpenAIActionTranslator.translate(computer_call.actions)
                result: ExecutorResult = await executor.execute(canonical_actions)
                screenshot_b64 = base64.b64encode(result.screenshot).decode("utf-8")
                mime = result.mime_type or self.screenshot_mime_type

                try:
                    response = await client.responses.create(
                        model=self.model,
                        tools=[{"type": "computer"}],
                        previous_response_id=response.id,
                        input=[
                            {
                                "type": "computer_call_output",
                                "call_id": computer_call.call_id,
                                "output": {
                                    "type": "computer_screenshot",
                                    "image_url": f"data:{mime};base64,{screenshot_b64}",
                                    "detail": "original",
                                },
                            }
                        ],
                    )
                except Exception as exc:
                    raise OpenAIComputerUseError(f"API call failed at step {step + 1} — {exc}", step=step + 1) from exc
                step_cost, in_tok, out_tok = self._record_usage(response)
                logger.info(
                    "OpenAIComputerUse step %d | %d action(s) | in=%d out=%d | $%.4f (Σ$%.4f)",
                    self._usage.steps,
                    len(computer_call.actions),
                    in_tok,
                    out_tok,
                    step_cost,
                    self._usage.cost_usd,
                )

            return self._extract_text(response)

    def _record_usage(self, response) -> tuple[float, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0, 0, 0

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "input_token_details", None)
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

        entry = OpenAIUsageLogEntry(
            model_version=self.model,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
        )
        step_cost = entry.calculate_cost()

        self._usage.input_tokens += input_tokens
        self._usage.output_tokens += output_tokens
        self._usage.steps += 1
        self._usage.cost_usd += step_cost

        ledger = get_token_cost_ledger()
        if ledger:
            ledger.add(
                TokenCostEntry(
                    source=f"OpenAIComputerUseAgent/{self.model}",
                    provider="openai",
                    model=self.model,
                    cost=step_cost,
                    token_details=entry.token_details(),
                )
            )

        return step_cost, input_tokens, output_tokens

    def _extract_text(self, response) -> str:
        if response.output_text:
            return response.output_text
        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        return content_item.text
        return ""
