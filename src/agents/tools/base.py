from collections.abc import Awaitable
from typing import Any, Callable

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    args_description: dict[str, str]
    tool: Callable[..., Awaitable[Any, Any, Any]]
    args_schema: dict[str, dict[str, Any]] = {}

    model_config = {"arbitrary_types_allowed": True}
