"""
ToolDefinition factories for computer-use inside SimpleAgent.

When ``GenerationBehaviorSettings.computer_use=True`` is passed to
``SimpleAgent.answer_to()``, the agent automatically injects these two tools:

* ``take_screenshot`` — captures the current screen state and returns a
  base64-encoded PNG (or JPEG) string the LLM can describe or act upon.

* ``execute_computer_action`` — performs one computer action (click, type,
  key press, scroll, navigate, wait) and returns a fresh screenshot.

Usage::

    from agents.computer_use.gridworld import GridWorldExecutor
    from agents.computer_use.tools import create_computer_use_tools

    executor = GridWorldExecutor()
    tools = create_computer_use_tools(executor)

    # Or let SimpleAgent inject them automatically:
    agent = SimpleAgent(..., screen_executor=executor)
    gbs = GenerationBehaviorSettings(computer_use=True)
    result = await agent.answer_to("Navigate the maze", gbs=gbs)
"""

from __future__ import annotations

import base64
import json
from typing import List

from agents.computer_use.actions import (
    ClickAction,
    DoubleClickAction,
    KeyComboAction,
    KeyPressAction,
    NavigateAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from agents.computer_use.executor import BaseScreenExecutor
from agents.tools.base import ToolDefinition


def create_computer_use_tools(executor: BaseScreenExecutor) -> List[ToolDefinition]:
    """Return ``[take_screenshot, execute_computer_action]`` tools backed by *executor*.

    Args:
        executor: Any :class:`~agents.computer_use.executor.BaseScreenExecutor`
            instance (e.g. :class:`~agents.computer_use.gridworld.GridWorldExecutor`
            or :class:`~agents.computer_use.executor.PlaywrightExecutor`).

    Returns:
        A list of two :class:`~agents.tools.base.ToolDefinition` objects ready
        to be passed to ``SimpleAgent(tools=...)``.
    """

    async def take_screenshot(_args: dict) -> str:
        """Capture current screen state with no actions."""
        result = await executor.execute([])
        b64 = base64.b64encode(result.screenshot).decode()
        return json.dumps(
            {
                "screenshot_base64": b64,
                "mime_type": result.mime_type,
                "url": result.url or "",
            }
        )

    async def execute_computer_action(args: dict) -> str:
        """Execute one computer action and return a new screenshot."""
        action_type = args.get("action_type", "")
        params: dict = args.get("params", {})

        match action_type:
            case "click":
                action = ClickAction(x=int(params["x"]), y=int(params["y"]))
            case "double_click":
                action = DoubleClickAction(x=int(params["x"]), y=int(params["y"]))
            case "type":
                action = TypeAction(text=str(params["text"]))
            case "key_press":
                keys = params.get("keys", [])
                if isinstance(keys, str):
                    keys = [keys]
                action = KeyPressAction(keys=keys)
            case "key_combo":
                action = KeyComboAction(combo=str(params["combo"]))
            case "scroll":
                action = ScrollAction(
                    x=int(params.get("x", 0)),
                    y=int(params.get("y", 0)),
                    scroll_x=int(params.get("scroll_x", 0)),
                    scroll_y=int(params.get("scroll_y", 0)),
                )
            case "navigate":
                action = NavigateAction(url=str(params["url"]))
            case "wait":
                action = WaitAction(seconds=float(params.get("seconds", 1.0)))
            case _:
                return json.dumps({"error": f"Unknown action_type: {action_type!r}"})

        result = await executor.execute([action])
        b64 = base64.b64encode(result.screenshot).decode()
        return json.dumps(
            {
                "screenshot_base64": b64,
                "mime_type": result.mime_type,
                "url": result.url or "",
            }
        )

    screenshot_tool = ToolDefinition(
        name="take_screenshot",
        description=(
            "Capture the current screen state. "
            "Returns a JSON object with 'screenshot_base64' (base64-encoded PNG/JPEG), "
            "'mime_type', and 'url' (current page URL if applicable)."
        ),
        args_schema={"type": "object", "properties": {}, "required": []},
        tool=take_screenshot,
    )

    action_tool = ToolDefinition(
        name="execute_computer_action",
        description=(
            "Execute a computer action and return a fresh screenshot of the result.\n"
            "action_type values and their params:\n"
            "  'click'        — {\"x\": <int>, \"y\": <int>}\n"
            "  'double_click' — {\"x\": <int>, \"y\": <int>}\n"
            "  'type'         — {\"text\": <str>}\n"
            "  'key_press'    — {\"keys\": [<str>, ...]}\n"
            "  'key_combo'    — {\"combo\": <str>}  e.g. 'Control+C'\n"
            "  'scroll'       — {\"x\": <int>, \"y\": <int>, \"scroll_x\": <int>, \"scroll_y\": <int>}\n"
            "  'navigate'     — {\"url\": <str>}\n"
            "  'wait'         — {\"seconds\": <float>}"
        ),
        args_schema={
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "click",
                        "double_click",
                        "type",
                        "key_press",
                        "key_combo",
                        "scroll",
                        "navigate",
                        "wait",
                    ],
                    "description": "The type of computer action to perform.",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters specific to the chosen action_type.",
                },
            },
            "required": ["action_type", "params"],
        },
        tool=execute_computer_action,
    )

    return [screenshot_tool, action_tool]
