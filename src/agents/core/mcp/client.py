"""
MCP Client

Handles connections to MCP servers via stdio transport.
"""

import asyncio
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.core.mcp.models import MCPServerConfig, MCPToolInfo, MCPToolResult
from agents.utils.logs.config import logger


class MCPClient:
    """
    Client for connecting to and communicating with MCP servers.
    
    Uses stdio transport to communicate with MCP servers running as subprocesses.
    
    IMPORTANT: This client must be used as an async context manager:
    
        async with MCPClient(config) as client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
    """
    
    def __init__(self, config: MCPServerConfig):
        """
        Initialize the MCP client.
        
        Args:
            config: Configuration for the MCP server connection
        """
        self.config = config
        self._session: Optional[ClientSession] = None
        self._cm_stack = None
        self._connected = False
    
    async def _connect_internal(self):
        """Internal method to establish connection using proper context management."""
        logger.info(f"Connecting to MCP server: {self.config.name}")
        
        # Build the command
        command = self.config.command[0] if self.config.command else "python"
        args = self.config.command[1:] + self.config.args if len(self.config.command) > 1 else self.config.args
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=self.config.env if self.config.env else None,
        )
        
        # Create the context manager
        self._stdio_cm = stdio_client(server_params)
        
        # Enter the stdio context
        self._read_stream, self._write_stream = await self._stdio_cm.__aenter__()
        
        # Create and initialize session
        self._session = ClientSession(self._read_stream, self._write_stream)
        await self._session.__aenter__()
        await self._session.initialize()
        
        self._connected = True
        logger.info(f"Successfully connected to MCP server: {self.config.name}")
    
    async def _disconnect_internal(self):
        """Internal method to properly disconnect."""
        if not self._connected:
            return
        
        try:
            # Exit session context
            if self._session:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Session exit error (may be expected): {e}")
                self._session = None
            
            # Exit stdio context - wrap in shield to prevent cancellation issues
            if hasattr(self, '_stdio_cm') and self._stdio_cm:
                try:
                    import asyncio
                    await asyncio.wait_for(
                        asyncio.shield(self._stdio_cm.__aexit__(None, None, None)),
                        timeout=2.0
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Stdio context exit (may be expected): {type(e).__name__}")
                self._stdio_cm = None
            
            self._connected = False
            logger.info(f"Disconnected from MCP server: {self.config.name}")
            
        except Exception as e:
            logger.debug(f"Disconnect from MCP server {self.config.name}: {e}")
            self._connected = False
    
    async def list_tools(self) -> List[MCPToolInfo]:
        """
        List all available tools from the MCP server.
        
        Returns:
            List of MCPToolInfo objects describing available tools
        """
        if not self._connected or not self._session:
            logger.error(f"Not connected to MCP server: {self.config.name}")
            return []
        
        try:
            result = await self._session.list_tools()
            
            tools = []
            for tool in result.tools:
                tool_info = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or f"Tool: {tool.name}",
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    server_name=self.config.name,
                )
                tools.append(tool_info)
            
            logger.info(f"Discovered {len(tools)} tools from MCP server: {self.config.name}")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools from MCP server {self.config.name}: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPToolResult containing the result or error
        """
        if not self._connected or not self._session:
            return MCPToolResult(
                content=f"Not connected to MCP server: {self.config.name}",
                is_error=True
            )
        
        try:
            logger.info(f"Calling MCP tool: {tool_name} with args: {arguments}")
            
            result = await self._session.call_tool(tool_name, arguments)
            
            # Extract content from result
            content_parts = []
            for content_item in result.content:
                if hasattr(content_item, 'text'):
                    content_parts.append(content_item.text)
                elif hasattr(content_item, 'data'):
                    content_parts.append(str(content_item.data))
                else:
                    content_parts.append(str(content_item))
            
            content = "\n".join(content_parts) if content_parts else str(result)
            
            logger.info(f"MCP tool {tool_name} completed successfully")
            return MCPToolResult(content=content, is_error=result.isError if hasattr(result, 'isError') else False)
            
        except Exception as e:
            logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return MCPToolResult(content=str(e), is_error=True)
    
    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._connect_internal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._disconnect_internal()
