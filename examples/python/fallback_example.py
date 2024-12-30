from mainframe_orchestra import Task, Agent, OpenaiModels, AnthropicModels, set_verbosity
set_verbosity(1)

# This is a simple example of how to use a fallback model for an agent.
# It demonstrates how you can specify a list of models to try in order, and the agent will be able to fall back to the next model if the priors fail.

chat_agent = Agent(
    agent_id="chat_agent",
    role="Chat Agent",
    goal="Chat with the user",
    llm=[OpenaiModels.custom_model("fakemodel"), AnthropicModels.custom_model("otherbrokenmodel"), OpenaiModels.gpt_4o_mini]
)

def chat_task(conversation_history, userinput):
    return Task.create(
        agent=chat_agent,
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
        print(f"**Chat Agent**: {response}")

if __name__ == "__main__":
    main()