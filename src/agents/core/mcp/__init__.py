"""
MCP (Model Context Protocol) Infrastructure

This module provides MCP client capabilities for connecting to MCP servers
and discovering tools dynamically.
"""

from agents.core.mcp.client import MCPClient
from agents.core.mcp.discovery import MCPDiscovery
from agents.core.mcp.models import MCPServerConfig, MCPToolInfo

__all__ = [
    "MCPServerConfig",
    "MCPToolInfo",
    "MCPClient",
    "MCPDiscovery",
]
