# Fred Tools

The FredTools class provides a set of methods to interact with the Federal Reserve Economic Data (FRED) API for economic data analysis. It offers powerful functionality for analyzing economic indicators, yield curves, and economic news sentiment. This class is designed to simplify the process of working with the FRED API, handling data retrieval and analysis internally.

### Class Methods

##### economic_indicator_analysis()

Performs a comprehensive analysis of economic indicators using the FRED API. This method provides detailed statistics and trends for specified economic indicators over a given time period.

```python
FredTools.economic_indicator_analysis(
    indicator_ids=["GDP", "UNRATE", "CPIAUCSL"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

##### yield_curve_analysis()

Analyzes the US Treasury yield curve using data from the FRED API. This method is particularly useful for assessing economic conditions and potential future trends based on the shape of the yield curve.

```python
FredTools.yield_curve_analysis(
    treasury_maturities=["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"],
    start_date="2022-01-01",
    end_date="2023-12-31"
)
```

##### economic_news_sentiment_analysis()

Performs sentiment analysis on economic news series data from the FRED API. This method helps in understanding the overall sentiment in economic news over a specified time period.

```python
FredTools.economic_news_sentiment_analysis(
    news_series_id="STLFSI2",
    start_date="2022-01-01",
    end_date="2023-12-31"
)
```

Here's an example of how you might create an economic analyst agent using these tools:

```python
economic_analyst = Agent(
    role="Economic Analyst",
    goal="Provide comprehensive economic analysis and insights",
    attributes="Data-driven, detail-oriented, proficient in economic theory and statistics",
    tools={FredTools.economic_indicator_analysis, FredTools.yield_curve_analysis, FredTools.economic_news_sentiment_analysis},
    llm=OpenrouterModels.haiku
)
```

This economic analyst agent can leverage the Fred tools to analyze economic indicators, assess yield curves, and evaluate economic news sentiment, making it a powerful assistant for economic analysis tasks.

### Usage Notes

To use the FredTools class, you must set the FRED_API_KEY environment variable. This credential is essential for authenticating with the FRED API and is securely managed by the class.

The class methods handle API authentication and data retrieval internally, abstracting away the complexity of working directly with the FRED API. This allows developers to focus on analyzing the economic data without worrying about the underlying data retrieval mechanism.

All methods in the FredTools class return data in the form of Python dictionaries, making it easy to work with the results in your application. The structure of the returned data includes comprehensive statistics and analysis results, ensuring that you have access to detailed insights from the economic data.

Error handling is built into these methods, with potential issues such as missing data or API errors being handled gracefully. This helps in debugging and handling potential issues that may arise during data retrieval and analysis.

### Dependencies

The FredTools class requires the following additional libraries:

- pandas: For data manipulation and analysis
- fredapi: For interacting with the FRED API

These dependencies can be installed using pip:

```bash
pip install pandas fredapi
```

### Example Task

Here's an example of how you might create a task for the economic analyst agent using the FredTools:

```python
from mainframe_orchestra import Task, Agent, FredTools

financial_analyst = Agent(
    role="Financial Analyst",
    goal="Provide comprehensive financial analysis and insights",
    attributes="Data-driven, detail-oriented, proficient in financial theory and statistics",
    tools={FredTools.economic_indicator_analysis, FredTools.yield_curve_analysis, FredTools.economic_news_sentiment_analysis},
    llm=OpenrouterModels.haiku
)

def analyze_economic_conditions(agent):
    return Task.create(
        agent=financial_analyst,
        instruction="""Analyze current economic conditions using the following steps:
        1. Analyze key economic indicators (GDP, Unemployment Rate, and CPI) for the past 3 years.
        2. Examine the yield curve for signs of potential recession.
        3. Assess the overall sentiment in economic news for the past year.
        Provide a comprehensive report summarizing your findings and potential economic outlook."""
    )

# Usage
economic_report = analyze_economic_conditions(economic_analyst)
print(economic_report)
```

This task demonstrates how the economic analyst agent can use the FredTools to perform a comprehensive analysis of economic conditions, leveraging data from multiple sources and methods to provide valuable insights.