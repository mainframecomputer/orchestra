from mainframe_orchestra import Task, Agent, OpenaiModels, YahooFinanceTools, MatplotlibTools, set_verbosity
set_verbosity(1)

# This example shows how to create stock charts using a team of orchestra agents.

# Required installs: 
# pip install mainframe-orchestra matplotlib yfinance

# The script creates two agents:
# 1. price_plotter: Creates a line chart of stock prices
# 2. recommendation_plotter: Creates a bar chart of stock recommendations

# How to use:
# 1. Run the script
# 2. Enter a stock ticker (like 'AAPL' for Apple)
# 3. The agents will automatically:
#    - Fetch the stock data
#    - Create the charts
#    - Save them as image files

# Define the team of agents in the workflow pipeline
price_plotter = Agent(
    agent_id="price_plotter",
    role="Financial Price Analyst",
    goal="Analyze the price of a stock and use your tools tocreate a line plot of the price",
    llm=OpenaiModels.gpt_4o,
    tools=[YahooFinanceTools.get_historical_data, MatplotlibTools.create_line_plot]
)

recommendation_plotter = Agent(
    agent_id="recommendation_plotter",
    role="Financial Recommendation Analyst",
    goal="Analyze the recommendations of a stock and use your tools to create a bar plot of the recommendations",
    llm=OpenaiModels.gpt_4o,
    tools=[YahooFinanceTools.get_recommendations, MatplotlibTools.create_bar_plot]
)

def plot_price_chart(ticker: str):
    return Task.create(
        agent=price_plotter,
        instruction=f"Research the price of {ticker} over the last month, and create a line plot of the price. Ensure you use your tools to create and save the plot to file."
    )

def plot_recommendation_chart(ticker: str):
    return Task.create(
        agent=recommendation_plotter,
        instruction=f"Research the recommendations for {ticker} over the last month, and create a bar plot of the recommendations. Ensure you use your tools to create and save the plot to file."
    )

# get user input    
userinput = input("Enter ticker: ")

# run tasks
price_plot = plot_price_chart(userinput)
print(price_plot)

recommendation_plot = plot_recommendation_chart(userinput)
print(recommendation_plot)

