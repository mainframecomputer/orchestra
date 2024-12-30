from mainframe_orchestra import Task, Agent, OpenaiModels
from browser_use import Agent as BrowserAgent
from langchain_openai import ChatOpenAI
import asyncio

# This example shows how to use the browser-use agent as a callable tool for an orchestra agent.
# It's experimental and depends on the browser-use package at https://github.com/browser-use/browser-use
# You can install it with `pip install browser-use`

# This Browser tool is a simple wrapper around the browser-use Agent, and will kick off the browser-use agent as a delegate.
class BrowserTools:
    @staticmethod
    async def browse_web(instruction: str) -> str:
        """Use browser-use to perform web browsing tasks
        
        Args:
            instruction (str): Web browsing task to execute, written in natural language
        
        Returns:
            str: Result of the executed browsing task
        """
        browser_agent = BrowserAgent(
            task=f"Browse the web and find information about {instruction}. Close cookies modals and other popups before using the page.",
            llm=ChatOpenAI(model="gpt-4o-mini"),
        )
        result = await browser_agent.run()
        return result

# Define an orchestra agent that can call on the browser-agent as a tool to browse the web and find information
web_research_agent = Agent(
    agent_id="web_research_agent",
    role="Web Research Agent",
    goal="Use your web research tools to assist with the given task",
    attributes="You have expertise in web research and can use your tools to assist with the given task",
    llm=OpenaiModels.gpt_4o_mini,
    tools=[BrowserTools.browse_web]
)

# Define a chat task with conversation history and user input
async def chat_task(conversation_history, userinput):
    task_result = await Task.create(
        agent=web_research_agent,
        messages=conversation_history,
        instruction=userinput
    )
    return task_result

# Main loop to run the chat task
async def main():
    conversation_history = []
    while True:
        userinput = input("You: ")
        conversation_history.append({"role": "user", "content": userinput})
        response = await chat_task(conversation_history, userinput)
        conversation_history.append({"role": "assistant", "content": response})
        print(f"**Browser Agent**: {response}")

# Run the main loop
if __name__ == "__main__":
    asyncio.run(main())