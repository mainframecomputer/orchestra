# Yahoo Finance Tools

The YahooFinanceTools class provides a comprehensive set of methods to interact with Yahoo Finance data. It offers powerful functionality for retrieving stock information, historical data, financial statements, and performing technical and fundamental analysis. This class is designed to simplify the process of working with financial data, handling data retrieval and processing internally.

### Class Methods

##### get_ticker_info(ticker: str) -> Dict[str, Any]

Retrieves comprehensive information about a stock ticker.

```python
YahooFinanceTools.get_ticker_info("AAPL")
```

##### get_historical_data(ticker: str, period: str = "1y", interval: str = "1wk") -> str

Gets historical price data for a stock ticker.

```python
YahooFinanceTools.get_historical_data("AAPL", period="6mo", interval="1d")
```

##### calculate_returns(tickers: Union[str, List[str]], period: str = "1y", interval: str = "1d") -> Dict[str, pd.Series]

Calculates daily returns for given stock ticker(s).

```python
YahooFinanceTools.calculate_returns(["AAPL", "GOOGL"], period="3mo", interval="1d")
```

##### get_financials(ticker: str, statement: str = "income") -> pd.DataFrame

Retrieves financial statements for a stock ticker.

```python
YahooFinanceTools.get_financials("AAPL", statement="balance")
```

##### get_recommendations(ticker: str) -> pd.DataFrame

Gets analyst recommendations for a stock ticker.

```python
YahooFinanceTools.get_recommendations("AAPL")
```

##### download_multiple_tickers(tickers: List[str], period: str = "1mo", interval: str = "1d") -> pd.DataFrame

Downloads historical data for multiple tickers.

```python
YahooFinanceTools.download_multiple_tickers(["AAPL", "MSFT", "GOOGL"], period="3mo", interval="1d")
```

##### get_asset_profile(ticker: str) -> Dict[str, Any]

Retrieves the asset profile for a given stock ticker.

```python
YahooFinanceTools.get_asset_profile("AAPL")
```

##### get_balance_sheet(ticker: str, quarterly: bool = False)

Gets the balance sheet for a given stock ticker.

```python
YahooFinanceTools.get_balance_sheet("AAPL", quarterly=True)
```

##### get_cash_flow(ticker: str, quarterly: bool = False)

Retrieves the cash flow statement for a given stock ticker.

```python
YahooFinanceTools.get_cash_flow("AAPL")
```

##### get_income_statement(ticker: str, quarterly: bool = False)

Gets the income statement for a given stock ticker.

```python
YahooFinanceTools.get_income_statement("AAPL", quarterly=True)
```

##### get_custom_historical_data(ticker: str, start_date: str, end_date: str, frequency: str = '1d', event: str = 'history')

Retrieves custom historical data for a stock ticker with specified parameters.

```python
YahooFinanceTools.get_custom_historical_data("AAPL", "2023-01-01", "2023-06-30", frequency="1wk")
```

##### technical_analysis(ticker: str, period: str = "1y") -> Dict[str, Any]

Performs technical analysis for a given stock ticker.

```python
YahooFinanceTools.technical_analysis("AAPL", period="6mo")
```

##### fundamental_analysis(ticker: str) -> Dict[str, Any]

Performs a comprehensive fundamental analysis for a given stock ticker.

```python
YahooFinanceTools.fundamental_analysis("AAPL")
```

### Error Handling

All methods in the YahooFinanceTools class include robust error handling. If an error occurs during data retrieval or processing, a ValueError is raised with a descriptive error message. This helps in debugging and handling potential issues that may arise during use.

### Usage Notes

To use the YahooFinanceTools class, you need to install the required dependencies. You can do this by running:

```bash
pip install yfinance yahoofinance pandas
```

This will install the necessary packages: `yfinance`, `yahoofinance`, and `pandas`.

The class methods handle data retrieval and processing internally, abstracting away the complexity of working with different financial APIs. This allows developers to focus on analyzing the data rather than worrying about the underlying data retrieval mechanism.

All methods in the YahooFinanceTools class return data in the form of Python dictionaries or pandas DataFrames, making it easy to work with the results in your application.

Error handling is built into these methods, with exceptions being caught and re-raised with additional context. This helps in debugging and handling potential issues that may arise during data retrieval and processing.

### Example Usage in a Task

Here's an example of how you might create a financial analyst agent using these tools:

```python
financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyze stocks and provide investment recommendations",
    attributes="Knowledgeable about financial markets, detail-oriented, data-driven",
    tools={
        YahooFinanceTools.get_ticker_info,
        YahooFinanceTools.get_historical_data,
        YahooFinanceTools.technical_analysis,
        YahooFinanceTools.fundamental_analysis
    },
    llm=OpenrouterModels.haiku
)

def analyze_stock(agent, ticker):
    return Task.create(
        agent=agent,
        instruction=f"Perform a comprehensive analysis of {ticker} and provide an investment recommendation."
    )

# Usage
response = analyze_stock(financial_analyst, "AAPL")
print(response)
```

This financial analyst agent can leverage the YahooFinanceTools to retrieve stock information, perform technical and fundamental analysis, and provide investment recommendations based on the data.

### Conclusion

The YahooFinanceTools class provides a powerful set of methods for financial data retrieval and analysis. By integrating these tools into your Orchestra agents, you can create sophisticated financial analysis and investment recommendation systems with ease.

### Dependencies

The YahooFinanceTools class relies on the following external libraries:
- yfinance
- yahoofinance
- pandas

These dependencies are automatically installed when you install the package with the yahoo_finance_tools extra:

```bash
pip install Orchestra[yahoo_finance_tools]
```

If you encounter any ImportError, make sure these libraries are properly installed in your environment.