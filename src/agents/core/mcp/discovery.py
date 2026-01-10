"""
MCP Discovery

Discovers tools from MCP servers and converts them to ToolDefinition objects
that can be used by agents.
"""

from typing import Any, Callable, Dict, List, Optional

from agents.core.mcp.client import MCPClient
from agents.core.mcp.models import MCPServerConfig, MCPToolInfo
from agents.tools.base import ToolDefinition
from agents.utils.logs.config import logger


class MCPDiscovery:
    """
    Discovers and manages MCP tools from multiple servers.
    
    Converts MCP tools to ToolDefinition objects that agents can use.
    
    IMPORTANT: Use as context manager or call cleanup() when done:
    
        async with MCPDiscovery() as discovery:
            tools = await discovery.discover(configs)
    
    Or manually:
        discovery = MCPDiscovery()
        await discovery.connect_server(config)
        # ... use tools ...
        await discovery.cleanup()
    """
    
    def __init__(self):
        """Initialize the MCP discovery manager."""
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, MCPToolInfo] = {}  # tool_name -> tool_info
    
    async def connect_server(self, config: MCPServerConfig) -> bool:
        """
        Connect to an MCP server.
        
        Args:
            config: Configuration for the MCP server
            
        Returns:
            True if connection successful
        """
        if config.name in self._clients:
            logger.warning(f"MCP server {config.name} already connected")
            return True
        
        try:
            client = MCPClient(config)
            await client.__aenter__()
            
            if client.is_connected:
                self._clients[config.name] = client
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {config.name}: {e}")
            return False
    
    async def discover_tools(self, server_name: Optional[str] = None) -> List[MCPToolInfo]:
        """
        Discover tools from connected MCP servers.
        
        Args:
            server_name: Optional specific server to discover from.
                        If None, discovers from all connected servers.
                        
        Returns:
            List of discovered MCPToolInfo objects
        """
        all_tools = []
        
        clients_to_query = (
            {server_name: self._clients[server_name]} 
            if server_name and server_name in self._clients 
            else self._clients
        )
        
        for name, client in clients_to_query.items():
            tools = await client.list_tools()
            for tool in tools:
                self._tools[tool.name] = tool
                all_tools.append(tool)
        
        logger.info(f"Total tools discovered: {len(all_tools)}")
        return all_tools
    
    def _create_tool_wrapper(self, tool_info: MCPToolInfo) -> Callable:
        """
        Create an async wrapper function for an MCP tool.
        
        Args:
            tool_info: Information about the MCP tool
            
        Returns:
            Async function that calls the MCP tool
        """
        server_name = tool_info.server_name
        tool_name = tool_info.name
        
        async def mcp_tool_wrapper(**kwargs) -> str:
            """Wrapper that calls the MCP tool."""
            if server_name not in self._clients:
                return f"Error: MCP server {server_name} not connected"
            
            client = self._clients[server_name]
            result = await client.call_tool(tool_name, kwargs)
            
            if result.is_error:
                return f"Error: {result.content}"
            return result.content
        
        # Set function metadata for better introspection
        mcp_tool_wrapper.__name__ = tool_name
        mcp_tool_wrapper.__doc__ = tool_info.description
        
        return mcp_tool_wrapper
    
    def _extract_args_from_schema(self, input_schema: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        Extract argument descriptions and schemas from JSON schema.
        
        Args:
            input_schema: JSON schema for tool inputs
            
        Returns:
            Tuple of (args_description, args_schema)
        """
        args_description = {}
        args_schema = {}
        
        if not input_schema:
            return args_description, args_schema
        
        properties = input_schema.get("properties", {})
        
        for arg_name, arg_info in properties.items():
            # Extract description
            description = arg_info.get("description", f"Parameter: {arg_name}")
            args_description[arg_name] = description
            
            # Build schema
            schema = {"type": arg_info.get("type", "string")}
            if "description" in arg_info:
                schema["description"] = arg_info["description"]
            if "enum" in arg_info:
                schema["enum"] = arg_info["enum"]
            if "items" in arg_info:
                schema["items"] = arg_info["items"]
            
            args_schema[arg_name] = schema
        
        return args_description, args_schema
    
    def convert_to_tool_definitions(self, tools: Optional[List[MCPToolInfo]] = None) -> List[ToolDefinition]:
        """
        Convert MCP tools to ToolDefinition objects.
        
        Args:
            tools: List of MCPToolInfo to convert. If None, uses all discovered tools.
            
        Returns:
            List of ToolDefinition objects
        """
        tools_to_convert = tools or list(self._tools.values())
        tool_definitions = []
        
        for tool_info in tools_to_convert:
            # Extract args from schema
            args_description, args_schema = self._extract_args_from_schema(tool_info.input_schema)
            
            # Create wrapper function
            wrapper = self._create_tool_wrapper(tool_info)
            
            # Create ToolDefinition
            tool_def = ToolDefinition(
                name=tool_info.name,
                description=tool_info.description,
                args_description=args_description,
                tool=wrapper,
                args_schema=args_schema,
            )
            tool_definitions.append(tool_def)
            
            logger.info(f"Converted MCP tool to ToolDefinition: {tool_info.name}")
        
        return tool_definitions
    
    async def discover(self, configs: List[MCPServerConfig]) -> List[ToolDefinition]:
        """
        Connect to MCP servers, discover tools, and convert to ToolDefinitions.
        
        This is the main entry point for discovering MCP tools.
        
        Args:
            configs: List of MCP server configurations
            
        Returns:
            List of ToolDefinition objects for all discovered tools
        """
        # Connect to all servers
        for config in configs:
            await self.connect_server(config)
        
        # Discover tools
        await self.discover_tools()
        
        # Convert to ToolDefinitions
        return self.convert_to_tool_definitions()
    
    async def cleanup(self):
        """Disconnect from all MCP servers and clean up resources."""
        for name, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
        
        self._clients.clear()
        self._tools.clear()
        logger.info("MCP Discovery cleanup complete")
    
    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """Get the client for a specific server."""
        return self._clients.get(server_name)
    
    @property
    def connected_servers(self) -> List[str]:
        """List of connected server names."""
        return list(self._clients.keys())
    
    @property
    def available_tools(self) -> List[str]:
        """List of available tool names."""
        return list(self._tools.keys())
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
