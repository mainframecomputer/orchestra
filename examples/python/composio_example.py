from composio import App
from mainframe_orchestra import Task, Agent, OpenaiModels, Conduct, set_verbosity
from mainframe_orchestra.adapters.composio_adapter import ComposioAdapter

# To view detailed prompt logs, set verbosity to 1. For less verbose logs, set verbosity to 0.
set_verbosity(1)

# This example shows how to define a team of agents and assign them to a conductor agent.
# The conductor agent will orchestrate the agents to perform a task.
# The conductor agent exists in a tak loop, and will be able to orchestrate the agents to perform any necessary tasks.

# Define the team of agents in the workflow

composio = ComposioAdapter()
slack = Agent(
    agent_id="slack",
    role="Slack agent",
    goal="Control the Slack app to send and receive messages",
    attributes="You have access to the API provided by Slack. You can help users send and receive messages to a channel and to a single person.",
    llm=OpenaiModels.gpt_4o,
    tools=composio.get_tools(App.SLACK)
)

google_doc = Agent(
    agent_id="google_doc",
    role="Google Doc agent",
    goal="Control the Google Doc app to create and edit documents",
    attributes="You have access to the API provided by Google Doc. You can help users create and edit documents.",
    llm=OpenaiModels.gpt_4o,
    tools=composio.get_tools(App.GOOGLEDOCS)
)

conductor_agent = Agent(
    agent_id="conductor_agent",
    role="Conductor",
    goal="Complete user's request by orchestrating the agents you have access to. When you are unsure what to do, ALWAYS ask the user for clarification first.",
    attributes="You are a helpful work assistant that has the ability to control enterprise apps to complete tasks.",
    llm=OpenaiModels.gpt_4o,
    tools=[Conduct.conduct_tool(google_doc, slack)]
)

def chat_task(conversation_history, userinput):
    return Task.create(
        agent=conductor_agent,
        messages=conversation_history,
        instruction=userinput
    )

def main():
    conversation_history = []
    while True:
        userinput = input("You: ")
        conversation_history.append({"role": "user", "content": userinput})
        response = chat_task(conversation_history, userinput)
        conversation_history.append({"role": "assistant", "content": response})
        print(f"Final response: {response}")

if __name__ == "__main__":
    main()