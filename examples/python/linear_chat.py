from mainframe_orchestra import Task, Agent, OpenaiModels, LinearTools, set_verbosity
set_verbosity(0)

# This example shows how to create a chat loop with an agent that uses the LinearTools to assist with the given task.
# NOTE: This team is capable of updating issue statuses in Linear. Use in a test team, or remove the update_issue_status tool if you don't want to edit issues in Linear.

# Environment variables needed:
# LINEAR_API_KEY=your_linear_api_key
# LINEAR_TEAM_ID=your_linear_team_id

# This script:
# 1. Creates an agent with Linear tools
# 2. Runs a chat loop where you can:
#    - View team issues
#    - Search issues
#    - Check workflow states
#    - Update issue status
#
# Usage: Just run the script and start chatting!

# Initialize the toolkit 
LinearTools()

# Create the agent
linear_agent = Agent(
    agent_id="linear_agent",
    role="Linear Agent",
    goal="Use your linear tools to assist with the given task",
    tools=[LinearTools.get_team_issues, LinearTools.get_workflow_states, LinearTools.search_issues, LinearTools.update_issue_status],
    llm=OpenaiModels.gpt_4o
)

# Define the task
def task(user_input, conversation_history):
    return Task.create(
        agent=linear_agent,
        messages=conversation_history,
        instruction=f"Use your linear tools to assist with the given task: '{user_input}"
    )

# Run the agent-chat loop
def main():
    # Initialize the conversation history
    conversation_history = []

    # Start the conversation loop
    while True:
        user_input = input("You: ")
        response = task(user_input, conversation_history)
        print(response)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
