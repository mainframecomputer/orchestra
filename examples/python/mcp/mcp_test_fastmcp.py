import asyncio
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity

set_verbosity(2)

async def main():
    # Create the MCPOrchestra client
    async with MCPOrchestra() as mcp_client:
        # Connect to the FastMCP server and start it
        print("Starting and connecting to FastMCP server...")
        await mcp_client.connect(
            server_name="calculator",
            command="python",
            args=["mcp_fast_calc.py"], # Replace with the path to your FastMCP server script
            start_server=True,
            server_startup_delay=2.0
        )

        # Get all tools as callables
        calculator_tools = mcp_client.get_tools()

        # Define the agent
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