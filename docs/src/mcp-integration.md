# MCP Integration with Orchestra

Orchestra provides integration with the Model Context Protocol (MCP), allowing your agents to use tools implemented in various programming languages and frameworks. The MCP adapter enables you to extend Orchestra's capabilities beyond Python and integrate with external services and libraries built by the community based on Model Context Protocol (MCP).

You can connect your orchestra agents to MCP servers running locally with stdio, or remotely via SSE (Server-Sent Events) with a simple URL-based setup.

## What is MCP?

The Model Context Protocol (MCP), developed by Anthropic, is a standardized protocol for communication between language models and external tools. It allows tools to be implemented in various programming languages while maintaining a consistent interface for language models to interact with them.

## The MCPOrchestra Adapter

Orchestra includes an `MCPOrchestra` adapter that serves as a bridge between your Python Orchestra agents and MCP-compatible tools. This adapter:

1. Connects to MCP servers (which can be written in any language)
2. Discovers available tools on those servers
3. Converts those tools into Python callables with proper docstrings
4. Makes them available to your Orchestra agents

## Using MCP Tools with Orchestra

### Basic Usage Pattern

```python
import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

# Optional: Set verbosity level for debugging
set_verbosity(2)

async def main():
    # Create the MCP adapter using async context manager
    async with MCPOrchestra() as mcp_client:
        # Connect to an MCP server
        await mcp_client.connect(
            server_name="my_server",
            command="command_to_run_server",
            args=["arg1", "arg2"],
            start_server=True,  # Optional: start the server if it's not already running
            server_startup_delay=2.0  # Optional: wait for server to initialize
        )

        # Get tools from the server
        tools = mcp_client.get_tools()

        # Create an agent with these tools
        agent = Agent(
            agent_id="mcp_agent",  # Optional but recommended for identification
            role="Assistant",
            goal="Help with tasks using MCP tools",
            tools=tools,
            llm=OpenaiModels.gpt_4o
        )

        # Create a task using the agent
        result = await Task.create(
            agent=agent,
            instruction="Your instruction here"
        )

        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Available MCP Tool Libraries

Orchestra can work with any MCP-compatible server. Here are some popular MCP tool libraries you can use:

| Tool | Description | Installation | Example |
|------|-------------|-------------|---------|
| [FastMCP](https://github.com/jlowin/fastmcp) | Create Python-based MCP servers with minimal code | `pip install fastmcp` | [Calculator Example](#calculator-example-with-fastmcp) |
| [Fetch MCP](https://github.com/modelcontextprotocol/servers/blob/main/src/fetch) | Web fetching and scraping tools | `npx @tokenizin/mcp-npx-fetch` | [Fetch Example](#web-fetching-example) |
| [Filesystem MCP](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem) | File system operations | `npx @modelcontextprotocol/server-filesystem` | [Filesystem Example](#filesystem-example) |
| [Playwright MCP](https://github.com/executeautomation/mcp-playwright) | Web automation with Playwright | `npx playwright-mcp-server` | [Playwright Example](#web-automation-with-playwright) |
| [Slack MCP](https://github.com/modelcontextprotocol/servers/blob/main/src/slack) | Slack API integration | See [Slack Example](#slack-integration-example) | [Slack Example](#slack-integration-example) |

## Examples

### Calculator Example with FastMCP

First, create a FastMCP server file:

```python
# mcp_fast_calc.py
from fastmcp import FastMCP

# Create a FastMCP server with a name
mcp = FastMCP("Calculator")

# Define tools using decorators
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of the two numbers
    """
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        The product of the two numbers
    """
    return a * b

# You can also define resource endpoints
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting

    Args:
        name: The name of the person to greet

    Returns:
        A personalized greeting
    """
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

Then, use it with Orchestra:

```python
# mcp_test_fastmcp.py
import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)  # Enable verbose logging

async def main():
    async with MCPOrchestra() as mcp_client:
        # Start the FastMCP server and connect to it
        print("Starting and connecting to FastMCP server...")
        await mcp_client.connect(
            server_name="calculator",
            command="python",
            args=["mcp_fast_calc.py"],  # Path to your FastMCP server script
            start_server=True,  # Start the server process
            server_startup_delay=2.0  # Wait for server to initialize
        )

        # Get calculator tools
        calculator_tools = mcp_client.get_tools()

        # Create a math assistant agent
        agent = Agent(
            agent_id="calculator_agent",
            role="Math Assistant",
            goal="Help users perform mathematical calculations",
            tools=calculator_tools,
            llm=OpenaiModels.gpt_4o
        )

        # Create a task
        result = await Task.create(
            agent=agent,
            instruction="You need to calculate the sum of 42 and 58, and then multiply the result by 5. Respond with the final answer."
        )

        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Web Fetching Example

```python
# mcp_test_fetch.py
import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def run_fetch_mcp():
    async with MCPOrchestra() as mcp_client:
        try:
            # Connect to the Fetch MCP server
            await mcp_client.connect(
                server_name="fetch",
                command="npx",
                args=["@tokenizin/mcp-npx-fetch"]
            )

            # Get all tools from the MCP server
            fetch_tools = mcp_client.get_tools()

            # Define the agent
            agent = Agent(
                agent_id="fetch_agent",
                role="Web Assistant",
                goal="Help users fetch and analyze web content",
                tools=fetch_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create a task with these tools
            result = await Task.create(
                agent=agent,
                instruction="Fetch the content from docs.orchestra.org and summarize what the website is about.",
            )

            print(result)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_fetch_mcp())
```

### Filesystem Example

```python
# mcp_test_filesystem.py
import asyncio
import os
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def run_filesystem_mcp():
    async with MCPOrchestra() as mcp_client:
        try:
            # Define directories to allow access to (customize these paths)
            allowed_directories = [
                os.path.expanduser("~/Desktop"),  # Allow access to Desktop
                # Add more directories as needed
            ]

            # Connect to the Filesystem MCP server
            await mcp_client.connect(
                server_name="filesystem",
                command="npx",
                args=["@modelcontextprotocol/server-filesystem"] + allowed_directories,
                # Note: Directories are passed as additional arguments for security
            )

            # Get all tools from the MCP server
            filesystem_tools = mcp_client.get_tools()

            # Define the agent
            agent = Agent(
                agent_id="filesystem_agent",
                role="File System Assistant",
                goal="Help users manage files and directories",
                attributes="You know to use absolute paths when possible",
                tools=filesystem_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create a task with these tools
            result = await Task.create(
                agent=agent,
                instruction="List the files on my desktop and summarize what you find.",
            )

            print(result)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_filesystem_mcp())
```

### Web Automation with Playwright

```python
# mcp_test_playwright.py
import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def run_agent_with_mcp_tools():
    async with MCPOrchestra() as mcp_client:
        try:
            # Connect to the Playwright MCP server
            await mcp_client.connect(
                server_name="playwright",
                command="npx",
                args=["playwright-mcp-server"]
            )

            # Get all tools from the MCP server
            playwright_tools = mcp_client.get_tools()

            # Define the agent
            agent = Agent(
                agent_id="playwright_agent",
                role="Web Automation Assistant",
                goal="Help users automate web tasks using Playwright",
                tools=playwright_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create the task with step-by-step instructions
            result = await Task.create(
                agent=agent,
                instruction="""
                Please perform the following web automation tasks:
                1. Navigate to duckduckgo.com
                2. Try to use the selector to find the search input
                3. Use the identified selector to search for 'AI Agent news'
                4. Click the search button or press Enter to submit the search
                5. Find the first result and click it
                6. Take a screenshot of the resulting page
                7. Summarize what you found
                """
            )

            print(result)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent_with_mcp_tools())
```

### Slack Integration Example

```python
# mcp_test_slack.py
import asyncio
import os
import shutil
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
set_verbosity(2)

async def run_slack_mcp():
    async with MCPOrchestra() as mcp_client:
        try:
            # Get Slack credentials from environment variables
            slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
            slack_team_id = os.environ.get("SLACK_TEAM_ID")

            if not slack_bot_token or not slack_team_id:
                print("ERROR: Missing Slack credentials. Set SLACK_BOT_TOKEN and SLACK_TEAM_ID environment variables.")
                return

            # Get the path to node executable
            node_path = shutil.which("node")
            if not node_path:
                print("ERROR: Node.js not found. Please install Node.js.")
                return

            # Path to the Slack MCP server module
            # Note: You need to install this first with:
            # npm install @modelcontextprotocol/server-slack
            server_path = "node_modules/@modelcontextprotocol/server-slack/dist/index.js"

            # Connect to the Slack MCP server
            await mcp_client.connect(
                server_name="slack",
                command=node_path,
                args=[server_path],
                env={
                    "SLACK_BOT_TOKEN": slack_bot_token,
                    "SLACK_TEAM_ID": slack_team_id
                }
            )

            # Get Slack tools
            slack_tools = mcp_client.get_server_tools("slack")

            # Create the Slack agent
            slack_agent = Agent(
                agent_id="slack_agent",
                role="Slack Assistant",
                goal="Help users interact with Slack by using your tools",
                tools=slack_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create a conversation history for context
            conversation_history = []

            # Main conversation loop
            while True:
                # Get user input
                print("\nEnter your instruction (or 'quit' to exit):")
                user_instruction = input("You: ")

                if user_instruction.lower() == 'quit':
                    break

                # Create a task with these tools
                result = await Task.create(
                    agent=slack_agent,
                    instruction=user_instruction,
                    messages=conversation_history
                )

                # Add the interaction to conversation history
                conversation_history.append({"role": "user", "content": user_instruction})
                conversation_history.append({"role": "assistant", "content": str(result)})

                print("\nTask result:")
                print(result)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_slack_mcp())
```

## Remote MCP Server Example

You can also connect to MCP servers hosted remotely using SSE (Server-Sent Events):

```python
import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, set_verbosity
from orchestra.packages.python.src.mainframe_orchestra.adapters.mcp_adapter import MCPOrchestra

set_verbosity(2)

async def run_agent_with_mcp_tools():
    # Create and connect to MCP server using async context manager
    async with MCPOrchestra() as mcp_client:
        try:
            # Connect to the Supabase MCP server
            await mcp_client.connect(
                server_name="supabase",
                sse_url="YOUR_MCP_URL_HERE"
            )

            # Get all tools from the MCP server
            mcp_tools = mcp_client.get_tools()
            print(f"Available tools: {[tool.__name__ for tool in mcp_tools]}")

            agent = Agent(
                agent_id="supabase_agent",
                role="Supabase Database Assistant",
                goal="Help users interact with Supabase databases",
                tools=mcp_tools,
                llm=OpenaiModels.gpt_4o
            )

            user_input = input("Enter a request: ")

            result = await Task.create(
                agent=agent,
                instruction=user_input
            )

            print(result)

        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent_with_mcp_tools())
```

## Creating Your Own MCP Server with FastMCP

FastMCP is a Python library that makes it easy to create MCP-compatible servers. Here's how to create a simple server:

```python
from fastmcp import FastMCP

# Create a FastMCP server with a name
mcp = FastMCP("MyTools")

# Define a simple tool
@mcp.tool()
def my_tool(param1: str, param2: int) -> str:
    """
    Description of what my_tool does

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of the return value
    """
    # Tool implementation
    return f"Processed {param1} {param2} times"

# Define a resource endpoint
@mcp.resource("data://{id}")
def get_data(id: str) -> dict:
    """
    Get data for a specific ID

    Args:
        id: The ID to get data for

    Returns:
        Data associated with the ID
    """
    # Resource implementation
    return {"id": id, "value": f"Data for {id}"}

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## Advanced Features

### Connecting to Multiple Servers

You can connect to multiple MCP servers and combine their tools:

```python
async with MCPOrchestra() as mcp_client:
    # Connect to a calculator server
    await mcp_client.connect(
        server_name="calculator",
        command="python",
        args=["mcp_fast_calc.py"],
        start_server=True
    )

    # Connect to a web automation server
    await mcp_client.connect(
        server_name="playwright",
        command="npx",
        args=["playwright-mcp-server"]
    )

    # Get all tools from all servers
    all_tools = mcp_client.get_tools()

    # Or get tools from specific servers
    calculator_tools = mcp_client.get_server_tools("calculator")
    playwright_tools = mcp_client.get_server_tools("playwright")

    # Create agents with different tool sets
    calculator_agent = Agent(
        agent_id="calculator_agent",
        role="Math Assistant",
        tools=calculator_tools,
        llm=OpenaiModels.gpt_4o
    )

    web_agent = Agent(
        agent_id="web_agent",
        role="Web Assistant",
        tools=playwright_tools,
        llm=OpenaiModels.gpt_4o
    )

    super_agent = Agent(
        agent_id="super_agent",
        role="Super Assistant",
        tools=all_tools,
        llm=OpenaiModels.gpt_4o
    )
```

### Listing Available Tools

You can list all available tools from connected MCP servers:

```python
# List tools from all servers
tools_info = await mcp_client.list_tools(verbose=True)
print(tools_info)

# List tools from a specific server
calculator_tools_info = await mcp_client.list_tools(server_name="calculator")
print(calculator_tools_info)
```

### Installation Requirements

Before using MCP tools, make sure you have the server package installed and set to the correct path.

## Best Practices when building your own MCP server / tool library

1. **Use Descriptive Tool Names and Schemas**: The adapter converts tool descriptions and schemas into docstrings that the LLM uses to understand how to use the tools.

2. **Handle Errors Gracefully**: Implement proper error handling in your MCP tools to provide helpful error messages that can guide the LLM to correct its usage.

3. **Use Environment Variables for Sensitive Information**: When connecting to MCP servers that require API keys or other sensitive information, pass them as environment variables.

4. **Close Connections Properly**: Use the `async with` pattern with `MCPOrchestra` to ensure connections are properly closed and server processes are terminated.

5. **Consider Tool Granularity**: Create focused, single-purpose tools rather than complex multi-purpose ones for better usability by the LLM.

6. **Security Considerations**: When using filesystem or other sensitive tools, always restrict access to only the necessary directories or resources.

## Troubleshooting

### Common Issues

1. **Server Not Found**: Make sure the command and args correctly point to the MCP server executable.
   ```
   Error: Cannot find module '@tokenizin/mcp-npx-fetch'
   ```
   Solution: Install the package with `npm install @tokenizin/mcp-npx-fetch` and make sure the path is correct.

2. **Connection Errors**: If you see connection errors, check that:
   - The server is running
   - You've provided the correct environment variables
   - You've waited long enough for the server to start (adjust `server_startup_delay`)

3. **Tool Not Found**: If the agent can't find a tool, check that:
   - The tool is properly defined in the MCP server
   - You're using `get_tools()` or `get_server_tools(server_name)` correctly

## Tool Access Methods

When working with multiple MCP servers, Orchestra provides two methods to access tools:

### `get_tools()` vs `get_server_tools(server_name)`

- **`get_tools()`**: Returns all tools from all connected MCP servers combined into a single set.
  ```python
  # Get all tools from all connected servers
  all_tools = mcp_client.get_tools()

  # Create an agent with access to all tools
  super_agent = Agent(
      agent_id="super_agent",
      role="Super Assistant",
      tools=all_tools,
      llm=OpenaiModels.gpt_4o
  )
  ```

- **`get_server_tools(server_name)`**: Returns only the tools from a specific named server.
  ```python
  # Get only tools from the "calculator" server
  calculator_tools = mcp_client.get_server_tools("calculator")

  # Create a specialized agent with only calculator tools
  calculator_agent = Agent(
      agent_id="calculator_agent",
      role="Math Assistant",
      tools=calculator_tools,
      llm=OpenaiModels.gpt_4o
  )
  ```

Choose `get_tools()` when you want an agent to have access to all available tools, and use `get_server_tools()` when you want to create specialized agents with access to specific tool sets.


## Overall

MCP integration enables Orchestra to work with tools implemented in any programming language, significantly expanding its capabilities. Whether you need to integrate with existing services, leverage specialized libraries, or create custom tools in your preferred language, the MCP adapter provides a standardized way to make these tools available to your Orchestra agents.