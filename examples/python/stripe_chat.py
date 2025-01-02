from mainframe_orchestra import Agent, Task, OpenaiModels, StripeTools

# This example demonstrates how to create a chat loop with an agent that uses the StripeTools to assist with given tasks.

# Required packages
# pip install mainframe-orchestra stripe-agent-toolkit

# Environment variables needed:
# STRIPE_API_KEY=your_stripe_api_key

# Define read-only tools
stripe_read_tools = [
    StripeTools.check_balance,
    StripeTools.list_customers,
    StripeTools.list_products,
    StripeTools.list_prices,
]

# Define the agent
stripe_agent = Agent(
    agent_id="stripe_agent",
    role="Stripe Agent",
    goal="Use your stripe tools to assist with the given task",
    tools=stripe_read_tools,
    llm=OpenaiModels.gpt_4o_mini
)

# Define the task
def task(user_input, conversation_history):
    return Task.create(
        agent=stripe_agent,
        instruction=f"Use your stripe tools to assist with the given task: '{user_input}",
        messages=conversation_history
    )

# Run the agent chat loop
def main():
    conversation_history = []
    while True:
        user_input = input("You: ")
        response = task(user_input, conversation_history)
        print(response)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
