"""
Screen executor abstraction for computer-use agents.

BaseScreenExecutor is the single interface between the LLM loop and the
environment.  Implement it once for your environment; both OpenAI and Gemini
agents will call it with the same canonical List[BaseAction].

Bundled implementations
-----------------------
PlaywrightExecutor — drives a Playwright Page (works for both providers).

Adding a new executor
---------------------
Subclass BaseScreenExecutor and implement execute():

    class VNCExecutor(BaseScreenExecutor):
        def __init__(self, vnc_client):
            self._vnc = vnc_client

        async def execute(self, actions: list[BaseAction]) -> ExecutorResult:
            for action in actions:
                if isinstance(action, ClickAction):
                    self._vnc.click(action.x, action.y)
                elif isinstance(action, TypeAction):
                    self._vnc.type(action.text)
                # handle more action types …
            png = self._vnc.screenshot()
            return ExecutorResult(screenshot=png)

    agent = OpenAIComputerUseAgent(screen_executor=VNCExecutor(vnc))
    result = await agent.run("Click the login button")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List

from agents.computer_use.actions import (
    BaseAction,
    ClickAction,
    DoubleClickAction,
    DragAction,
    ExecutorResult,
    GoBackAction,
    GoForwardAction,
    KeyComboAction,
    KeyPressAction,
    MoveAction,
    NavigateAction,
    ScreenshotAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from agents.utils.logs.config import logger


class BaseScreenExecutor(ABC):
    """Abstract base for all computer-use screen executors.

    Concrete subclasses receive a list of canonical :class:`BaseAction` objects,
    execute them against the target environment, and return an
    :class:`ExecutorResult` containing the resulting screenshot and (optionally)
    the current page URL.

    An empty action list ``execute([])`` must still return a valid screenshot —
    this is used by the Gemini agent to capture the initial state before any
    actions are requested.
    """

    @abstractmethod
    async def execute(self, actions: List[BaseAction]) -> ExecutorResult:
        """Execute *actions* and return the resulting screen state.

        Args:
            actions: Canonical actions to perform, in order.  May be empty.

        Returns:
            :class:`ExecutorResult` with at minimum a ``screenshot`` field
            containing PNG (or JPEG) bytes of the current screen state after
            all actions have been applied.
        """
        raise NotImplementedError


class PlaywrightExecutor(BaseScreenExecutor):
    """Playwright-backed executor compatible with both OpenAI and Gemini agents.

    Args:
        page: A Playwright ``Page`` that is already open (and optionally
            pre-navigated via ``page.goto(start_url)``).
        screenshot_mime_type: ``"image/png"`` (default, lossless) or
            ``"image/jpeg"`` (smaller payloads, fewer safety-filter hits for
            Gemini).
        action_delay: Extra sleep in seconds injected after most actions.
            Set to ``0.5`` for Gemini (recommended) and ``0.0`` for OpenAI.
        load_timeout: Milliseconds to wait for ``domcontentloaded`` after
            navigation-style actions (click, keypress, navigate, …).

    Example — OpenAI::

        async with async_playwright() as p:
            page = await p.chromium.launch().new_page()
            await page.goto("https://google.com")
            executor = PlaywrightExecutor(page)
            agent = OpenAIComputerUseAgent(screen_executor=executor)
            result = await agent.run("Search for Python tutorials")

    Example — Gemini::

        executor = PlaywrightExecutor(page, screenshot_mime_type="image/jpeg", action_delay=0.5)
        agent = GeminiComputerUseAgent(screen_executor=executor)
        result = await agent.run("Search for Python tutorials")
    """

    _NO_DELAY = frozenset({WaitAction, ScreenshotAction, MoveAction})

    def __init__(
        self,
        page,
        screenshot_mime_type: str = "image/png",
        action_delay: float = 0.0,
        load_timeout: int = 5000,
    ):
        self._page = page
        self._mime = screenshot_mime_type
        self._img_type = "jpeg" if screenshot_mime_type == "image/jpeg" else "png"
        self._shot_kwargs: dict = {"type": self._img_type}
        if self._img_type == "jpeg":
            self._shot_kwargs["quality"] = 70
        self._action_delay = action_delay
        self._load_timeout = load_timeout

    async def execute(self, actions: List[BaseAction]) -> ExecutorResult:
        page = self._page

        for action in actions:
            try:
                await self._dispatch(page, action)
                if self._action_delay > 0 and type(action) not in self._NO_DELAY:
                    await asyncio.sleep(self._action_delay)
            except Exception as exc:
                logger.warning("PlaywrightExecutor: error in %s: %s", type(action).__name__, exc)

        screenshot = await page.screenshot(**self._shot_kwargs)
        return ExecutorResult(screenshot=screenshot, url=page.url, mime_type=self._mime)

    async def _dispatch(self, page, action: BaseAction) -> None:
        """Dispatch a single canonical action to the Playwright page."""
        match action:
            case ClickAction(x=x, y=y, button=b):
                await page.mouse.click(x, y, button=b)
                await self._wait_load(page)

            case DoubleClickAction(x=x, y=y, button=b):
                await page.mouse.dblclick(x, y, button=b)
                await self._wait_load(page)

            case TypeAction(text=text, x=x, y=y, press_enter=pe, clear_first=cf):
                if x is not None and y is not None:
                    await page.mouse.click(x, y)
                if cf:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                await page.keyboard.type(text)
                if pe:
                    await page.keyboard.press("Enter")
                    await self._wait_load(page)

            case KeyPressAction(keys=keys):
                for key in keys:
                    await page.keyboard.press(key)
                await self._wait_load(page)

            case KeyComboAction(combo=combo):
                await page.keyboard.press(combo)
                await self._wait_load(page)

            case ScrollAction(x=x, y=y, scroll_x=sx, scroll_y=sy):
                await page.mouse.move(x, y)
                await page.mouse.wheel(sx, sy)

            case DragAction(start_x=sx, start_y=sy, end_x=ex, end_y=ey):
                await page.mouse.move(sx, sy)
                await page.mouse.down()
                await page.mouse.move(ex, ey)
                await page.mouse.up()

            case MoveAction(x=x, y=y):
                await page.mouse.move(x, y)

            case WaitAction(seconds=s):
                await asyncio.sleep(s)

            case ScreenshotAction():
                pass  # screenshot is always taken after all actions

            case NavigateAction(url=url):
                await page.goto(url, wait_until="domcontentloaded")

            case GoBackAction():
                await page.go_back(wait_until="domcontentloaded", timeout=self._load_timeout)

            case GoForwardAction():
                await page.go_forward(wait_until="domcontentloaded", timeout=self._load_timeout)

            case _:
                logger.warning("PlaywrightExecutor: unhandled action type %s", type(action).__name__)

    async def _wait_load(self, page) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=self._load_timeout)
        except Exception:
            pass
