"""
MCP Data Models

Defines data structures for MCP server configuration and tool information.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""

    name: str = Field(description="Unique name for this MCP server")
    command: List[str] = Field(description="Command to start the MCP server (e.g., ['python', 'server.py'])")
    args: List[str] = Field(default_factory=list, description="Additional arguments for the command")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for the server")
    working_dir: Optional[str] = Field(default=None, description="Working directory for the server")

    model_config = {"extra": "forbid"}


class MCPToolInfo(BaseModel):
    """Information about a tool discovered from an MCP server."""

    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for tool inputs")
    server_name: str = Field(description="Name of the MCP server that provides this tool")

    model_config = {"extra": "forbid"}


class MCPToolResult(BaseModel):
    """Result from calling an MCP tool."""

    content: Any = Field(description="The result content from the tool")
    is_error: bool = Field(default=False, description="Whether the result is an error")

    model_config = {"extra": "forbid"}
