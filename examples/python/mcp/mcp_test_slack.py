import asyncio
import os
import shutil
from mainframe_orchestra import Task, Agent, OpenaiModels, MCPOrchestra, set_verbosity
from dotenv import load_dotenv
load_dotenv()

set_verbosity(2)

async def run_slack_mcp():
    # Create and connect to MCP server using async context manager
    async with MCPOrchestra() as mcp_client:
        try:
            # Get Slack credentials from environment variables
            slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
            slack_team_id = os.environ.get("SLACK_TEAM_ID")

            if not slack_bot_token or not slack_team_id:
                print("ERROR: Missing Slack credentials. Set SLACK_BOT_TOKEN and SLACK_TEAM_ID environment variables.")
                return

            # Get the path to node
            node_path = shutil.which("node")
            if not node_path:
                print("ERROR: Node.js not found. Please install Node.js.")
                return

            # Path to the server module
            server_path = "node-tools/node_modules/@modelcontextprotocol/server-slack/dist/index.js"

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

            # Get all tools from the MCP server
            slack_tools = mcp_client.get_server_tools("slack")

            # Create the agent
            slack_agent=Agent(
                agent_id="slack_agent",
                role="Slack Assistant",
                goal="Help users interact with Slack by using your tools",
                tools=slack_tools,
                llm=OpenaiModels.gpt_4o
            )

            # Create a conversation history
            conversation_history = []

            # Main conversation loop
            while True:
                # Get user input
                print("\nEnter your instruction (or 'quit' to exit):")
                user_instruction = input("You:")

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

# Run the example
if __name__ == "__main__":
    asyncio.run(run_slack_mcp())