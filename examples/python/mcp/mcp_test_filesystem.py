import asyncio
import os
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def run_filesystem_mcp():
    # Create and connect to MCP server using async context manager
    async with MCPOrchestra() as mcp_client:
        try:
            # Define directories to allow access to (customize these paths)
            allowed_directories = [
                os.path.expanduser("~/Desktop"),  # Allow access to Desktop, replace with your own paths
            ]

            await mcp_client.connect(
                server_name="filesystem",
                command="npx",
                args=["@modelcontextprotocol/server-filesystem"] + allowed_directories # Replace with the path to the MCP server
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
                instruction="summarize the contents of my desktop",
            )

            print(result)

        except Exception as e:
            print(f"Error: {e}")

# Run the example
if __name__ == "__main__":
    asyncio.run(run_filesystem_mcp())
