"""
Tool Invoker Module

This module provides the core mechanism for safely executing tools within the system.
The ToolInvoker class dynamically maps tool names to their implementations, validates
arguments, handles errors, and manages system-level parameters injection.

Architecture Decision Flow:
--------------------------
```
     ┌───────────────┐
     │ Tool Request  │
     └──────┬────────┘
            ▼
     ┌───────────────┐
     │ Lookup Tool   │──┐
     └──────┬────────┘  │ Not Found
            ▼           ▼
     ┌───────────────┐ ┌────────┐
     │ Validate Args │ │ Error  │
     └──────┬────────┘ └────────┘
            ▼
     ┌───────────────┐
     │ Extract Params│
     └──────┬────────┘
            ▼
     ┌───────────────┐
     │ Execute Tool  │───┐
     └──────┬────────┘   │
            ▼            ▼
     ┌────────────┐ ┌────────────┐
     │  Success   │ │ Exception  │
     └────┬───────┘ └────┬───────┘
          ▼              ▼
     ┌───────────────┐
     │ Process Result│
     └──────┬────────┘
            ▼
       ┌────────┐
       │ Return │
       └────────┘
```

The ToolInvoker uses a fail-fast approach, validating all conditions before
executing any tool code. This ensures predictable error messages and prevents
partial executions that could lead to inconsistent states in the system.
"""

import inspect
from typing import Any, Dict, List, Optional

from agents.tools.base import ToolDefinition
from agents.tools.exceptions import ToolExecutionError
from agents.utils.logs.config import logger


class ToolInvoker:
    def __init__(
        self,
        tool_definitions: List[ToolDefinition],
    ):
        self.tools_map: Dict[str, ToolDefinition] = {tool.name: tool for tool in tool_definitions}

    async def invoke(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_call_id: Optional[str] = None,
    ):
        try:
            tool_definition = self.tools_map[tool_name]
        except KeyError:
            raise ToolExecutionError(
                f"there is not any tool with name({tool_name})s."
                f"here is the available list of tools({list(self.tools_map.keys())})"
            )

        arguments_missing = set(tool_definition.args_description.keys()) - set(tool_arguments.keys())
        if arguments_missing:
            raise ToolExecutionError(f"Required arguments missing for tool{tool_name}: {arguments_missing}")
        unknown_arguments = set(tool_arguments.keys()) - set(tool_definition.args_description.keys())
        if unknown_arguments:
            raise ToolExecutionError(f"Unknown argument(s) passed for tool {tool_name}: {unknown_arguments}")

        tool_system_kwargs = self.extract_necessary_system_kwargs(tool_definition.tool, {})

        try:
            result = await self.run_tool(tool_definition.tool, tool_arguments, tool_system_kwargs)
        except Exception:
            logger.exception(f"Tool invocation failed: {tool_name} ({tool_arguments})")
            raise

        return result

    def get_func_arguments(self, func: object):
        return list(inspect.signature(func).parameters.keys())

    def extract_necessary_system_kwargs(self, func, system_kwargs):
        tool_system_kwargs = {}
        run_method_args = self.get_func_arguments(func)
        for arg in run_method_args:
            if arg in system_kwargs:
                tool_system_kwargs[arg] = system_kwargs[arg]
        return tool_system_kwargs

    async def run_tool(self, tool_run_method, llm_kwargs, tool_system_kwargs=None):
        all_kwargs = llm_kwargs | (tool_system_kwargs or {})
        result = await tool_run_method(**all_kwargs)
        return result
