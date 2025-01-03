from mainframe_orchestra import Task, Agent, AnthropicModels, WebTools, YahooFinanceTools, set_verbosity
from rich.console import Console
from rich.markdown import Markdown
console = Console()
# To view detailed prompt logs, set verbosity to 1. For less verbose logs, set verbosity to 0.
set_verbosity(0)

# Define the team of agents in the workflow
market_analyst = Agent(
    agent_id="market_analyst",
    role="Market Microstructure Analyst",
    goal="Analyze market microstructure and identify trading opportunities",
    attributes="You have expertise in market microstructure, order flow analysis, and high-frequency data.",
    llm=AnthropicModels.haiku_3_5,
    tools={YahooFinanceTools.calculate_returns, YahooFinanceTools.get_historical_data}
)

fundamental_analyst = Agent(
    agent_id="fundamental_analyst",
    role="Fundamental Analyst",
    goal="Analyze company financials and assess intrinsic value",
    attributes="You have expertise in financial statement analysis, valuation models, and industry analysis.",
    llm=AnthropicModels.haiku_3_5,
    tools={YahooFinanceTools.get_financials, YahooFinanceTools.get_ticker_info}
)

technical_analyst = Agent(
    agent_id="technical_analyst",
    role="Technical Analyst",
    goal="Analyze price charts and identify trading patterns",
    attributes="You have expertise in technical analysis, chart patterns, and technical indicators.",
    llm=AnthropicModels.haiku_3_5,
    tools={YahooFinanceTools.get_historical_data}
)

sentiment_analyst = Agent(
    agent_id="sentiment_analyst",
    role="Sentiment Analyst",
    goal="Analyze market sentiment, analyst recommendations and news trends",
    attributes="You have expertise in market sentiment analysis.",
    llm=AnthropicModels.haiku_3_5,
    tools={YahooFinanceTools.get_recommendations, WebTools.serper_search}
)

# Define the series of tasks in the workflow-pipeline
def analyze_market_task(ticker: str):
    market_report = Task.create(
        agent=market_analyst,
        instruction=f"Analyze the market microstructure for ticker {ticker} and identify trading opportunities and write a comprehensive report in markdown"
    )
    return market_report

def analyze_sentiment_task(ticker: str):
    sentiment_report = Task.create(
        agent=sentiment_analyst,
        instruction=f"Analyze the market sentiment, analyst recommendations and news for ticker {ticker} and write a comprehensive report in markdown"
    )
    return sentiment_report

def analyze_technical_task(ticker: str):
    technical_report = Task.create(
        agent=technical_analyst,
        instruction=f"Analyze the price charts and identify trading patterns for {ticker} and write a comprehensive report in markdown"
    )
    return technical_report

def analyze_fundamentals_task(ticker: str):
    fundamentals_report = Task.create(
        agent=fundamental_analyst,
        instruction=f"Analyze the company financials and assess intrinsic value for {ticker} and write a comprehensive report in markdown"
    )
    return fundamentals_report

def search_news_task(ticker: str):
    news_report = Task.create(
        agent=sentiment_analyst,
        instruction=f"Search news on '{ticker}', report the news summaries and insights and implications on the stock and market sentiment in the last month and write a comprehensive report in markdown"
    )
    return news_report

def final_report_task(market_report: str, sentiment_report: str, technical_report: str, fundamentals_report: str, news_report: str):
    final_report = Task.create(
        agent=market_analyst,
        context=f"Context:\n\nMarket Report:\n{market_report}\n--------\nSentiment Report:\n{sentiment_report}\n--------\nTechnical Report:\n{technical_report}\n--------\nFundamentals Report:\n{fundamentals_report}\n--------\nNews Report:\n{news_report}",
        instruction="Combine all the reports and create a final report in markdown"
    )
    return final_report

# Get user input
userinput = input("Enter ticker: \n")

# Run the series of tasks
analysis = analyze_market_task(userinput)
sentiment = analyze_sentiment_task(userinput)
technical = analyze_technical_task(userinput)
fundamentals = analyze_fundamentals_task(userinput)
news = search_news_task(userinput)
final = final_report_task(analysis, sentiment, technical, fundamentals, news)

# Print final report to console
console.print(Markdown(f"**Final Report**: {final}"))

# Save final report to file
with open("final_report.md", "w") as file:
    file.write(final)
