from mainframe_orchestra import Task, Agent, OpenaiModels, YahooFinanceTools, set_verbosity
# To view detailed prompt logs, set verbosity to 1. For less verbose logs, set verbosity to 0.
set_verbosity(0)

# This example shows how to assign a series of tools to an agent and use it in a chat loop.
# It uses the YahooFinanceTools to get financial data and recommendations.
# Note: This example is not meant to be a real-world example, but rather a simple demonstration of how to use tools with an agent.
# YahooFinanceTools can return a large amount of data, so it's recommended to test with an inexpensive model like gpt-4o-mini.


# Define Agents
market_analyst = Agent(
    agent_id="market_analyst",
    role="Market Analyst",
    goal="Analyze market trends and provide insights",
    attributes="data-driven, analytical, up-to-date with market news",
    llm=OpenaiModels.gpt_4o_mini,
    tools={YahooFinanceTools.download_multiple_tickers, YahooFinanceTools.get_financials, YahooFinanceTools.get_recommendations, YahooFinanceTools.get_ticker_info}
)

def chat_task(conversation_history, userinput):
    return Task.create(
        agent=market_analyst,
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
        print(f"**Market Analyst**: {response}")

if __name__ == "__main__":
    main()
