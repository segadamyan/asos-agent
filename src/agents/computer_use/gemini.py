"""
Gemini Computer Use Agent

Implements the Google Gemini Computer Use loop (Computer Use Preview):
  1. Capture an initial screenshot via the executor (empty action list).
  2. Send task + screenshot to generate_content.
  3. Translate any FunctionCall items → canonical BaseAction list via _GeminiActionTranslator.
     Coordinate denormalization (0–999 → pixels) happens inside the translator.
  4. Optionally handle safety_decision (require_confirmation).
  5. Execute canonical actions via the BaseScreenExecutor.
  6. Send FunctionResponse parts + new screenshot back.
  7. Repeat until no FunctionCalls remain.

Supported models:
  gemini-2.5-computer-use-preview-10-2025
  gemini-3-flash-preview

Note: Uses google-genai SDK 1.x API (ToolComputerUse, not ComputerUse).
      FunctionResponse does not support inline image parts in SDK 1.x —
      the screenshot is passed as a separate image Part in the same Content.
"""

from __future__ import annotations

import os
from google import genai
from google.genai import types
from typing import Awaitable, Callable, List, Optional

from agents.computer_use.actions import (
    BaseAction,
    ClickAction,
    DragAction,
    ExecutorResult,
    GoBackAction,
    GoForwardAction,
    KeyComboAction,
    MoveAction,
    NavigateAction,
    ScrollAction,
    TypeAction,
    WaitAction,
    normalize_key_combo,
)
from agents.computer_use.base import BaseComputerUseAgent, ComputerUseError, GeminiComputerUseError, UsageSummary
from agents.computer_use.executor import BaseScreenExecutor, PlaywrightExecutor
from agents.config.models import ModelCapability, model_registry
from agents.providers.gemini import GeminiUsageLogEntry
from agents.providers.models.token_usage import TokenCostEntry, get_token_cost_ledger
from agents.utils.logs.config import logger

# Safety confirmation callback: called when the model requests confirmation before a
# risky action. Receives the explanation string; return True to proceed, False to stop.
GeminiSafetyConfirmation = Callable[[str], Awaitable[bool]]

# Coordinate fields that use the 0–999 normalized grid.
_COORD_FIELDS = {"x", "y", "destination_x", "destination_y"}


class _GeminiActionTranslator:
    """Translates raw Gemini FunctionCall items into canonical actions.

    Coordinate denormalization (0–999 → pixels) is handled here so that the
    executor always receives real pixel values.
    """

    def __init__(self, screen_width: int, screen_height: int):
        self._w = screen_width
        self._h = screen_height

    def translate(self, raw_actions: list[tuple[str, dict]]) -> List[BaseAction]:
        result: List[BaseAction] = []
        for fname, args in raw_actions:
            denorm = self._denormalize(args)
            match fname:
                case "click_at":
                    result.append(ClickAction(x=denorm["x"], y=denorm["y"]))

                case "hover_at":
                    result.append(MoveAction(x=denorm["x"], y=denorm["y"]))

                case "type_text_at":
                    result.append(
                        TypeAction(
                            text=args.get("text", ""),
                            x=denorm["x"],
                            y=denorm["y"],
                            press_enter=args.get("press_enter", True),
                            clear_first=args.get("clear_before_typing", True),
                        )
                    )

                case "key_combination":
                    result.append(KeyComboAction(combo=normalize_key_combo(args.get("keys", ""))))

                case "scroll_at":
                    direction = args.get("direction", "down")
                    raw_mag = args.get("magnitude", 800)
                    magnitude = int(raw_mag / 1000 * self._h)
                    sx = magnitude if direction == "right" else (-magnitude if direction == "left" else 0)
                    sy = magnitude if direction == "down" else (-magnitude if direction == "up" else 0)
                    result.append(ScrollAction(x=denorm["x"], y=denorm["y"], scroll_x=sx, scroll_y=sy))

                case "scroll_document":
                    direction = args.get("direction", "down")
                    scroll_map = {
                        "down": (0, 600),
                        "up": (0, -600),
                        "right": (600, 0),
                        "left": (-600, 0),
                    }
                    sx, sy = scroll_map.get(direction, (0, 600))
                    result.append(ScrollAction(x=self._w // 2, y=self._h // 2, scroll_x=sx, scroll_y=sy))

                case "drag_and_drop":
                    result.append(
                        DragAction(
                            start_x=denorm["x"],
                            start_y=denorm["y"],
                            end_x=denorm["destination_x"],
                            end_y=denorm["destination_y"],
                        )
                    )

                case "navigate":
                    result.append(NavigateAction(url=args.get("url", "")))

                case "search":
                    result.append(NavigateAction(url="https://www.google.com"))

                case "go_back":
                    result.append(GoBackAction())

                case "go_forward":
                    result.append(GoForwardAction())

                case "wait_5_seconds":
                    result.append(WaitAction(seconds=5.0))

                case "open_web_browser":
                    pass  # no-op: browser is already open

                case _:
                    logger.debug("_GeminiActionTranslator: unknown function '%s'", fname)

        return result

    def _denormalize(self, args: dict) -> dict:
        """Copy *args* with 0–999 coordinate fields scaled to actual pixels."""
        result = dict(args)
        for field in _COORD_FIELDS:
            if field not in result:
                continue
            val = result[field]
            if field in {"x", "destination_x"}:
                result[field] = int(val / 1000 * self._w)
            else:
                result[field] = int(val / 1000 * self._h)
        return result


class GeminiComputerUseAgent(BaseComputerUseAgent):
    """Agent implementing the Google Gemini Computer Use loop.

    When ``screen_executor`` is omitted the agent launches its own Playwright
    Chromium browser, drives it internally, and closes it when :meth:`run`
    returns.  Pass ``headless=False`` to watch the browser.

    Example — default executor (no setup needed)::

        agent = GeminiComputerUseAgent(headless=False)
        result = await agent.run(
            "Search for the latest AI news.",
            start_url="https://www.google.com",
        )

    Example — custom executor::

        class MyExecutor(BaseScreenExecutor):
            async def execute(self, actions):
                for action in actions:
                    if isinstance(action, ClickAction):
                        await page.mouse.click(action.x, action.y)
                    # …
                return ExecutorResult(
                    screenshot=await page.screenshot(type="jpeg", quality=70),
                    url=page.url,
                    mime_type="image/jpeg",
                )

        agent = GeminiComputerUseAgent(screen_executor=MyExecutor())
        result = await agent.run("Go to google.com and search for Python.")
    """

    DEFAULT_MODEL = "gemini-2.5-computer-use-preview-10-2025"

    def __init__(
        self,
        screen_executor: Optional[BaseScreenExecutor] = None,
        system_prompt: str = "",
        model: str = DEFAULT_MODEL,
        max_steps: int = 50,
        safety_confirmation: Optional[GeminiSafetyConfirmation] = None,
        screen_width: int = BaseComputerUseAgent.SCREEN_WIDTH,
        screen_height: int = BaseComputerUseAgent.SCREEN_HEIGHT,
        headless: bool = True,
        screenshot_mime_type: str = "image/jpeg",
    ):
        """
        Args:
            screen_executor: Optional :class:`BaseScreenExecutor` for UI interaction.
                When omitted a default Playwright executor is used.
            system_prompt: Optional system instruction for the model.
            model: Gemini model ID that supports the Computer Use tool.
            max_steps: Maximum turns before the loop stops.
            safety_confirmation: Optional async callback for actions flagged
                ``require_confirmation``. Receives the explanation; return
                ``True`` to proceed, ``False`` to terminate.
            screen_width: Actual screen width in pixels (default 1440).
            screen_height: Actual screen height in pixels (default 900).
            headless: Run the default Playwright browser headlessly.
                Ignored when a custom ``screen_executor`` is provided.
            screenshot_mime_type: ``"image/jpeg"`` (default, smaller payload,
                fewer safety-filter false positives) or ``"image/png"``.
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
        self.safety_confirmation = safety_confirmation
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.headless = headless
        self.screenshot_mime_type = screenshot_mime_type

        self._translator = _GeminiActionTranslator(screen_width, screen_height)

    async def run(self, task: str, start_url: str = "") -> str:
        """Run the Computer Use loop for *task*, returning the final text output."""
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
                executor = PlaywrightExecutor(
                    page,
                    screenshot_mime_type=self.screenshot_mime_type,
                    action_delay=0.5,
                )
                return await self._run_loop(task, executor)
            finally:
                await browser.close()

    async def _run_loop(self, task: str, executor: BaseScreenExecutor) -> str:
        self._usage = UsageSummary()
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        _SAFETY_OFF = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        generate_config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ToolComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                    )
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            safety_settings=_SAFETY_OFF,
            system_instruction=self.system_prompt or None,
        )

        init_result = await executor.execute([])
        contents: list = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=task),
                    types.Part.from_bytes(data=init_result.screenshot, mime_type=self.screenshot_mime_type),
                ],
            )
        ]

        step = 0

        while step < self.max_steps:
            try:
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generate_config,
                )
            except Exception as exc:
                raise GeminiComputerUseError(
                    f"API call failed at step {step + 1} — {exc}",
                    step=step + 1,
                ) from exc

            if not response.candidates:
                feedback = getattr(response, "prompt_feedback", None)
                block_reason = str(getattr(feedback, "block_reason", "unknown"))
                raise GeminiComputerUseError(
                    f"Response blocked at step {step + 1} — {block_reason}",
                    step=step + 1,
                    block_reason=block_reason,
                )

            candidate = response.candidates[0]
            candidate_parts = [
                p for p in (candidate.content.parts or []) if candidate.content and not getattr(p, "thought", False)
            ]
            contents.append(types.Content(role="model", parts=candidate_parts))

            step_cost, in_tok, out_tok = self._record_usage(response)

            function_calls = [part.function_call for part in candidate_parts if part.function_call]

            if not function_calls:
                logger.info(
                    "GeminiComputerUse step %d/%d | <answer> | in=%d out=%d | $%.4f (Σ$%.4f)",
                    self._usage.steps,
                    self.max_steps,
                    in_tok,
                    out_tok,
                    step_cost,
                    self._usage.cost_usd,
                )
                return self._extract_text_from_parts(candidate_parts)

            logger.info(
                "GeminiComputerUse step %d/%d | %s | in=%d out=%d | $%.4f (Σ$%.4f)",
                self._usage.steps,
                self.max_steps,
                ", ".join(fc.name for fc in function_calls),
                in_tok,
                out_tok,
                step_cost,
                self._usage.cost_usd,
            )

            # Build raw (fname, args) list, handle safety_decision, collect acks
            raw_actions: list[tuple[str, dict]] = []
            safety_acks: dict[str, str] = {}
            terminated = False

            for fc in function_calls:
                fname = fc.name
                args = dict(fc.args) if fc.args else {}

                if "safety_decision" in args:
                    safety_decision = args.pop("safety_decision")
                    if isinstance(safety_decision, dict) and safety_decision.get("decision") == "require_confirmation":
                        explanation = safety_decision.get("explanation", "Action requires confirmation.")
                        if self.safety_confirmation:
                            proceed = await self.safety_confirmation(explanation)
                        else:
                            logger.warning(
                                "GeminiComputerUse: safety confirmation required but no handler — "
                                "auto-confirming. Explanation: %s",
                                explanation,
                            )
                            proceed = True

                        if not proceed:
                            logger.info("GeminiComputerUse: user denied '%s'. Terminating.", fname)
                            terminated = True
                            break
                        safety_acks[fname] = "true"

                raw_actions.append((fname, args))

            if terminated:
                break

            canonical_actions = self._translator.translate(raw_actions)
            exec_result: ExecutorResult = await executor.execute(canonical_actions)
            current_url = exec_result.url

            fr_parts: list = []
            for fname, _ in raw_actions:
                response_data: dict = {"url": current_url}
                if fname in safety_acks:
                    response_data["safety_acknowledgement"] = safety_acks[fname]
                fr_parts.append(types.Part.from_function_response(name=fname, response=response_data))

            fr_parts.append(types.Part.from_bytes(data=exec_result.screenshot, mime_type=self.screenshot_mime_type))
            contents.append(types.Content(role="user", parts=fr_parts))
            step += 1

        for content in reversed(contents):
            if content.role == "model":
                return self._extract_text_from_parts(content.parts or [])
        return ""

    def _record_usage(self, response) -> tuple[float, int, int]:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return 0.0, 0, 0

        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
        cached_tokens = getattr(meta, "cached_content_token_count", 0) or 0

        entry = GeminiUsageLogEntry(
            model_version=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
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
                    source=f"GeminiComputerUseAgent/{self.model}",
                    provider="gemini",
                    model=self.model,
                    cost=step_cost,
                    token_details=entry.token_details(),
                )
            )

        return step_cost, input_tokens, output_tokens

    @staticmethod
    def _extract_text_from_parts(parts) -> str:
        texts = []
        for part in parts:
            text = part.text if not isinstance(part, dict) else part.get("text")
            if text:
                texts.append(text)
        return " ".join(texts)
