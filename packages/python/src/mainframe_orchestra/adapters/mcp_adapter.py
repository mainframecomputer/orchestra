# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

"""
MCP Adapter for Orchestra - Allows Orchestra agents to use tools from MCP servers.

This module provides a client for connecting to MCP servers and converting their
tools into Orchestra-compatible callables that can be used in Orchestra Tasks.
"""

from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Set, Literal
import subprocess
import json
import time
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import (
    Tool as MCPTool,
    CallToolResult,
    TextContent
)
from mcp.client.sse import sse_client
from ..utils.logging_config import logger

class MCPOrchestra:
    """
    Client for connecting to MCP servers and exposing their tools to Orchestra agents.

    This class manages connections to multiple MCP servers and converts their tools
    into Orchestra-compatible callables that can be used in Orchestra Tasks.
    """

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new MCP Orchestra adapter.

        Args:
            credentials: Optional dictionary containing credentials for various MCP servers
        """
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Set[Callable] = set()
        self.server_tools: Dict[str, Set[Callable]] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.credentials = credentials or {}
        logger.debug("Initialized MCPOrchestra adapter")

    async def connect(
        self,
        server_name: str,
        *,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        encoding: str = "utf-8",
        encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict",
        start_server: bool = False,
        server_startup_delay: float = 2.0,
        sse_url: Optional[str] = None,
        sse_headers: Optional[Dict[str, Any]] = None,
        sse_timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
        credentials_key: Optional[str] = None,
    ) -> None:
        """
        Connect to an MCP server and load its tools.

        Args:
            server_name: A unique identifier for this server connection
            command: The command to run the MCP server (e.g., "python", "node", "npx")
            args: Arguments to pass to the command
            env: Optional environment variables for the server process
            encoding: Character encoding for communication
            encoding_error_handler: How to handle encoding errors
            start_server: Whether to start the server process before connecting
            server_startup_delay: Time to wait after starting server before connecting (seconds)
            sse_url: URL for SSE-based MCP server (if using SSE instead of stdio)
            sse_headers: Optional HTTP headers for SSE connection
            sse_timeout: Timeout for SSE connection establishment (seconds)
            sse_read_timeout: Timeout for SSE event reading (seconds)
            credentials_key: Optional key to use for looking up credentials in self.credentials
        """
        logger.debug(f"Connecting to MCP server: {server_name}")

        # Apply credentials if provided
        if credentials_key and credentials_key in self.credentials:
            server_creds = self.credentials[credentials_key]
            logger.debug(f"Using credentials from key: {credentials_key}")

            # Update environment variables with credentials
            if env is None:
                env = {}

            # Merge credentials into environment variables
            for key, value in server_creds.items():
                env_key = key.upper()  # Convert to uppercase for environment variables
                env[env_key] = value
                logger.debug(f"Added credential to environment: {env_key}")

        # Check if we're using SSE or stdio
        if sse_url:
            logger.debug(f"Using SSE connection for server {server_name}: {sse_url}")
            # SSE connection
            transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url=sse_url,
                    headers=sse_headers,
                    timeout=sse_timeout,
                    sse_read_timeout=sse_read_timeout,
                )
            )
            read, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            # Initialize session
            await session.initialize()
            self.sessions[server_name] = session
            logger.debug(f"Successfully initialized session for server: {server_name}")

            # Load tools from this server
            server_tools = await self._load_tools(session, server_name)
            self.server_tools[server_name] = server_tools
            self.tools.update(server_tools)
            logger.debug(f"Loaded {len(server_tools)} tools from server: {server_name}")

        else:
            # Stdio connection
            # Optionally start the server process
            if start_server:
                logger.debug(f"Starting MCP server process for {server_name}: {command} {args}")
                if not command or not args:
                    raise ValueError("Command and args must be provided when start_server is True")

                full_command = [command] + args
                server_process = subprocess.Popen(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                self.server_processes[server_name] = server_process
                logger.debug(f"Server process started with PID: {server_process.pid}")

                # Give the server time to start up
                logger.debug(f"Waiting {server_startup_delay}s for server to start")
                time.sleep(server_startup_delay)

            # Create server parameters
            if not command or not args:
                raise ValueError("Command and args must be provided for stdio connection")

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
                encoding=encoding,
                encoding_error_handler=encoding_error_handler,
            )

            # Establish connection
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            # Initialize session
            await session.initialize()
            self.sessions[server_name] = session
            logger.debug(f"Successfully initialized session for server: {server_name}")

            # Load tools from this server
            server_tools = await self._load_tools(session, server_name)
            self.server_tools[server_name] = server_tools
            self.tools.update(server_tools)
            logger.debug(f"Loaded {len(server_tools)} tools from server: {server_name}")

    async def _load_tools(self, session: ClientSession, server_name: str) -> Set[Callable]:
        """
        Load tools from an MCP server and convert them to Orchestra-compatible callables.

        Args:
            session: The MCP client session
            server_name: The name of the server (used in tool naming)

        Returns:
            A set of callable functions that can be used with Orchestra
        """
        logger.debug(f"Loading tools from server: {server_name}")
        tools_info = await session.list_tools()
        orchestra_tools: Set[Callable] = set()

        logger.debug(f"Found {len(tools_info.tools)} tools on server: {server_name}")

        for tool in tools_info.tools:
            # Get the tool name
            tool_name = tool.name

            # Use a factory to create a function with the correct name and closure
            def create_tool_function(name=tool_name, sess=session):
                async def tool_callable(**kwargs):
                    result = await sess.call_tool(name, kwargs)
                    return self._process_tool_result(result)

                # Set the function name
                tool_callable.__name__ = name

                return tool_callable

            tool_callable = create_tool_function()

            # Extract schema information
            schema_info = self._extract_schema_info(tool)

            # Create docstring from tool description and schema
            docstring = f"{tool.description}\n\n"

            if "properties" in schema_info:
                docstring += "Parameters:\n"
                for param_name, param_info in schema_info["properties"].items():
                    required = "required" in schema_info and param_name in schema_info["required"]
                    req_str = " (required)" if required else ""

                    description = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    type_str = f" ({param_type})" if param_type else ""

                    docstring += f"    {param_name}{req_str}{type_str}: {description}\n"

            # Set the docstring
            tool_callable.__doc__ = docstring

            orchestra_tools.add(tool_callable)

        return orchestra_tools

    def _extract_schema_info(self, tool: MCPTool) -> Dict[str, Any]:
        """
        Extract schema information from an MCP tool.

        Args:
            tool: The MCP tool object

        Returns:
            A dictionary containing the schema information
        """
        result = {"properties": {}, "required": []}

        # Try to access the inputSchema
        if hasattr(tool, "inputSchema"):
            schema = tool.inputSchema

            # Handle dictionary-based schema
            if isinstance(schema, dict):
                # Get properties
                if "properties" in schema:
                    for prop_name, prop_info in schema["properties"].items():
                        # Handle string representation of JSON
                        if isinstance(prop_info, str) and prop_info.startswith("{") and prop_info.endswith("}"):
                            try:
                                # Try to parse the string as JSON
                                prop_data = json.loads(prop_info)
                            except Exception:
                                prop_data = {"type": "string"}
                        else:
                            prop_data = prop_info

                        result["properties"][prop_name] = prop_data

                # Get required fields
                if "required" in schema:
                    result["required"] = schema["required"]

            # Handle object-based schema (fallback)
            elif hasattr(schema, "properties"):
                for prop_name, prop_info in schema.properties.items():
                    prop_data = {}

                    # Try to get type
                    if hasattr(prop_info, "type"):
                        prop_data["type"] = prop_info.type
                    elif hasattr(prop_info, "__getitem__"):
                        try:
                            prop_data["type"] = prop_info["type"]
                        except (KeyError, TypeError):
                            pass

                    # Try to get description
                    if hasattr(prop_info, "description"):
                        prop_data["description"] = prop_info.description
                    elif hasattr(prop_info, "__getitem__"):
                        try:
                            prop_data["description"] = prop_info["description"]
                        except (KeyError, TypeError):
                            pass

                    result["properties"][prop_name] = prop_data

                # Try to access required fields
                if hasattr(schema, "required"):
                    result["required"] = schema.required

        return result

    def _process_tool_result(self, result: CallToolResult) -> str:
        """
        Process an MCP tool result into a format suitable for Orchestra.

        Args:
            result: The result from an MCP tool call

        Returns:
            A string representation of the result

        Raises:
            Exception: If the tool result indicates an error
        """
        # Extract text content
        text_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)

        # Handle errors
        if result.isError:
            error_message = "\n".join(text_parts) if text_parts else "Unknown MCP tool error"
            logger.debug(f"Tool execution failed: {error_message}")
            raise Exception(error_message)

        # Return combined text
        if not text_parts:
            logger.debug("Tool executed successfully (no text output)")
            return "Tool executed successfully (no text output)"
        elif len(text_parts) == 1:
            logger.debug("Tool executed successfully with single text output")
            return text_parts[0]
        else:
            logger.debug(f"Tool executed successfully with {len(text_parts)} text outputs")
            return "\n".join(text_parts)

    def get_tools(self) -> Set[Callable]:
        """
        Get all tools from all connected MCP servers as Orchestra-compatible callables.

        Returns:
            A set of callable functions that can be used with Orchestra Tasks
        """
        return self.tools

    def get_server_tools(self, server_name: str) -> Set[Callable]:
        """
        Get tools from a specific MCP server.

        Args:
            server_name: The name of the server to get tools from

        Returns:
            A set of callable functions from the specified server

        Raises:
            KeyError: If the server name is not found
        """
        if server_name not in self.server_tools:
            raise KeyError(f"No server named '{server_name}' is connected")
        return self.server_tools[server_name]

    async def close(self) -> None:
        """Close all connections to MCP servers and terminate any started server processes."""
        logger.debug("Closing all MCP server connections")
        # Close all MCP sessions
        await self.exit_stack.aclose()

        # Terminate any server processes we started
        for server_name, process in self.server_processes.items():
            logger.debug(f"Terminating server process for {server_name} (PID: {process.pid})")
            try:
                process.terminate()
                process.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                logger.debug(f"Server process for {server_name} terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.debug(f"Server process for {server_name} did not terminate gracefully, forcing kill")
                process.kill()  # Force kill if it doesn't terminate gracefully
                process.wait()
                logger.debug(f"Server process for {server_name} killed")

    async def __aenter__(self) -> "MCPOrchestra":
        """Support for async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Clean up resources when exiting the context."""
        await self.close()

    async def list_tools(self, server_name: Optional[str] = None, verbose: bool = False) -> str:
        """
        List all available tools from connected MCP servers in a human-readable format.
        Helper method to list tools from all servers.

        Args:
            server_name: Optional name of a specific server to list tools from
            verbose: Whether to include detailed schema information

        Returns:
            A formatted string containing tool information

        Raises:
            KeyError: If the specified server name is not found
        """
        result = []

        # Determine which servers to list tools from
        if server_name:
            if server_name not in self.sessions:
                raise KeyError(f"No server named '{server_name}' is connected")
            servers_to_check = {server_name: self.sessions[server_name]}
        else:
            servers_to_check = self.sessions

        # Process each server
        for srv_name, session in servers_to_check.items():
            result.append(f"Tools from server: {srv_name}")
            result.append("=" * 50)

            # Get tools for this server
            tools_info = await session.list_tools()

            if not tools_info.tools:
                result.append("No tools found on this server.")
                continue

            # List each tool
            for i, tool in enumerate(tools_info.tools, 1):
                result.append(f"{i}. {tool.name}")
                result.append(f"   Description: {tool.description or 'No description'}")

                # Add schema information if verbose mode is enabled
                if verbose and tool.inputSchema:
                    result.append("   Input Schema:")
                    if isinstance(tool.inputSchema, dict) and "properties" in tool.inputSchema:
                        for prop_name, prop_info in tool.inputSchema["properties"].items():
                            # Handle string representation of JSON if needed
                            if isinstance(prop_info, str) and prop_info.startswith("{") and prop_info.endswith("}"):
                                try:
                                    # Try to parse the string as a dictionary
                                    prop_info = json.loads(prop_info)
                                except Exception:
                                    prop_info = {"type": "unknown", "description": prop_info}

                            # Get type and description
                            if isinstance(prop_info, dict):
                                prop_type = prop_info.get("type", "unknown")
                                prop_desc = prop_info.get("description", "No description")
                            else:
                                prop_type = "unknown"
                                prop_desc = str(prop_info)

                            required = "required" in tool.inputSchema and prop_name in tool.inputSchema["required"]
                            req_str = " (required)" if required else ""

                            result.append(f"     - {prop_name}{req_str} ({prop_type}): {prop_desc}")

                result.append("-" * 50)

        return "\n".join(result)