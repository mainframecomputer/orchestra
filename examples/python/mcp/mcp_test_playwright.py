import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity
set_verbosity(2)

async def run_agent_with_mcp_tools():
    # Create and connect to MCP server using async context manager
    async with MCPOrchestra() as mcp_client:
        try:
            # Connect to the Playwright MCP server
            await mcp_client.connect(
                server_name="playwright",
                command="npx",
                args=["playwright-mcp-server"]
            )

            # Get all tools from the MCP server
            mcp_tools = mcp_client.get_tools()

            # Define the agent
            agent = Agent(
                agent_id="playwright_agent",
                role="Web Automation Assistant",
                goal="Help users automate web tasks using Playwright",
                tools=mcp_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create the task
            result = await Task.create(
                agent=agent,
                instruction="""
                1. Navigate to duckduckgo.com
                2. Try to use the selector to find the search input
                3. Use the identified selector to search for 'AI Agent news'
                4. Click the search button or press Enter to submit the search
                5. Find the first result and click it
                """
            )

            print(result)

        except Exception as e:
            print(f"Error occurred: {e}")

# Run the example
asyncio.run(run_agent_with_mcp_tools())