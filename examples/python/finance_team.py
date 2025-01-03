from mainframe_orchestra import Task, Agent, OpenaiModels, WebTools, YahooFinanceTools, Conduct, set_verbosity

# To view detailed prompt logs, set verbosity to 1. For less verbose logs, set verbosity to 0.
set_verbosity(0)

# This example shows how to define a team of agents and assign them to a conductor agent.
# The conductor agent will orchestrate the agents to perform a task.
# The conductor agent exists in a tak loop, and will be able to orchestrate the agents to perform any necessary tasks.

# Define the team of agents in the workflow
market_analyst = Agent(
    agent_id="market_analyst",
    role="Market Microstructure Analyst",
    goal="Analyze market microstructure and identify trading opportunities",
    attributes="You have expertise in market microstructure, order flow analysis, and high-frequency data.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.calculate_returns, YahooFinanceTools.get_historical_data}
)

fundamental_analyst = Agent(
    agent_id="fundamental_analyst",
    role="Fundamental Analyst",
    goal="Analyze company financials and assess intrinsic value",
    attributes="You have expertise in financial statement analysis, valuation models, and industry analysis.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_financials, YahooFinanceTools.get_ticker_info}
)

technical_analyst = Agent(
    agent_id="technical_analyst",
    role="Technical Analyst",
    goal="Analyze price charts and identify trading patterns",
    attributes="You have expertise in technical analysis, chart patterns, and technical indicators.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_historical_data}
)

sentiment_analyst = Agent(
    agent_id="sentiment_analyst",
    role="Sentiment Analyst",
    goal="Analyze market sentiment, analyst recommendations and news trends",
    attributes="You have expertise in market sentiment analysis.",
    llm=OpenaiModels.gpt_4o,
    tools={YahooFinanceTools.get_recommendations, WebTools.serper_search}
)

conductor_agent = Agent(
    agent_id="conductor_agent",
    role="Conductor",
    goal="Conduct the orchestra",
    attributes="You have expertise in orchestrating the orchestra.",
    llm=OpenaiModels.gpt_4o,
    tools=[Conduct.conduct_tool(market_analyst, fundamental_analyst, technical_analyst, sentiment_analyst)]
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
        print(f"**Market Analyst**: {response}")

if __name__ == "__main__":
    main()