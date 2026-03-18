"""
Canonical computer-use action layer.

All provider-specific action formats (OpenAI dicts, Gemini function calls, …) are
translated into these Pydantic models before being handed to a BaseScreenExecutor.
Executors therefore only ever see provider-agnostic actions.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


# Merges OpenAI uppercase tokens ("CTRL") and Gemini lowercase tokens ("ctrl")
# into Playwright-compatible key names.
_KEY_NORM: dict[str, str] = {
    "ctrl": "Control",
    "control": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "meta": "Meta",
    "win": "Meta",
    "cmd": "Meta",
    "enter": "Enter",
    "return": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "space": "Space",
    "capslock": "CapsLock",
    "insert": "Insert",
    "printscreen": "PrintScreen",
    "pause": "Pause",
    **{f"f{n}": f"F{n}" for n in range(1, 13)},
}


def normalize_key(key: str) -> str:
    """Normalise a single key token to its Playwright name.

    Examples::

        normalize_key("CTRL")   # → "Control"
        normalize_key("ctrl")   # → "Control"
        normalize_key("Enter")  # → "Enter"
        normalize_key("a")      # → "a"   (unknown tokens pass through)
    """
    return _KEY_NORM.get(key.lower(), key)


def normalize_key_combo(combo: str) -> str:
    """Normalise a ``+``-separated key combo to Playwright format.

    Examples::

        normalize_key_combo("ctrl+c")       # → "Control+c"
        normalize_key_combo("CTRL+SHIFT+T") # → "Control+Shift+T"
    """
    return "+".join(normalize_key(part) for part in combo.split("+"))


class BaseAction(BaseModel):
    """Marker base class for all canonical computer-use actions."""


class ClickAction(BaseAction):
    x: int
    y: int
    button: str = "left"  # "left" | "right" | "middle"


class DoubleClickAction(BaseAction):
    x: int
    y: int
    button: str = "left"


class TypeAction(BaseAction):
    """Type text, optionally clicking a position first.

    When ``x`` and ``y`` are set the executor clicks that position before
    typing (Gemini's ``type_text_at`` style).
    ``clear_first=True`` selects-all + deletes existing content first.
    ``press_enter=True`` presses Enter after typing.
    """

    text: str
    x: Optional[int] = None
    y: Optional[int] = None
    press_enter: bool = False
    clear_first: bool = False


class KeyPressAction(BaseAction):
    """One or more keys pressed sequentially (normalised to Playwright names).

    Example: ``KeyPressAction(keys=["Control", "c"])``
    """

    keys: List[str] = []


class KeyComboAction(BaseAction):
    """A chord pressed simultaneously (normalised to Playwright format).

    Example: ``KeyComboAction(combo="Control+Shift+T")``
    """

    combo: str = ""


class ScrollAction(BaseAction):
    """Scroll at a screen position by a pixel delta.

    Positive ``scroll_y`` scrolls down; negative scrolls up.
    Positive ``scroll_x`` scrolls right; negative scrolls left.
    """

    x: int = 0
    y: int = 0
    scroll_x: int = 0
    scroll_y: int = 0


class DragAction(BaseAction):
    start_x: int = 0
    start_y: int = 0
    end_x: int = 0
    end_y: int = 0


class MoveAction(BaseAction):
    """Move the mouse cursor without clicking."""

    x: int = 0
    y: int = 0


class WaitAction(BaseAction):
    seconds: float = 2.0


class ScreenshotAction(BaseAction):
    """Explicit screenshot request — executor captures the screen without any
    other action.  Useful for the OpenAI ``screenshot`` action type which is a
    no-op that just triggers a screen capture."""


class NavigateAction(BaseAction):
    """Navigate the browser to a URL."""

    url: str = ""


class GoBackAction(BaseAction):
    """Browser back navigation."""


class GoForwardAction(BaseAction):
    """Browser forward navigation."""


class ExecutorResult(BaseModel):
    """Unified return value from any :class:`~agents.computer_use.executor.BaseScreenExecutor`.

    ``url`` is empty string for non-browser environments.
    ``mime_type`` mirrors the format of ``screenshot`` bytes.
    """

    screenshot: bytes
    url: str = ""
    mime_type: str = "image/png"
