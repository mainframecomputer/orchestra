import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def run_fetch_mcp():
    # Create and connect to MCP server using async context manager
    async with MCPOrchestra() as mcp_client:
        try:
            # Connect to the Fetch MCP server
            await mcp_client.connect(
                server_name="fetch",
                command="npx",
                args=["@tokenizin/mcp-npx-fetch"] # Replace with the path to the MCP server
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

# Run the example
if __name__ == "__main__":
    asyncio.run(run_fetch_mcp())