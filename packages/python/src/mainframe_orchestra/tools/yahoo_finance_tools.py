# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import json
from typing import List, Dict, Any, Union

def check_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError("yfinance is required for YahooFinanceTools. Install with `pip install yfinance`")

def check_yahoofinance():
    try:
        import yahoofinance
        return yahoofinance
    except ImportError:
        raise ImportError("yahoofinance is required for YahooFinanceTools. Install with `pip install yahoofinance`")

def check_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("pandas is required for YahooFinanceTools. Install with `pip install pandas`")


class YahooFinanceTools:
    @staticmethod
    def get_ticker_info(ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a stock ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            Dict[str, Any]: A dictionary containing detailed stock information.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yf = check_yfinance()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get the latest price
            history = stock.history(period="1d")
            latest_price = history['Close'].iloc[-1] if not history.empty else None

            return {
                "name": info.get("longName"),
                "symbol": info.get("symbol"),
                "current_price": latest_price,
                "currency": info.get("currency"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "pe_ratio": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_average": info.get("fiftyDayAverage"),
                "200_day_average": info.get("twoHundredDayAverage"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "beta": info.get("beta"),
                "book_value": info.get("bookValue"),
                "price_to_book": info.get("priceToBook"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margins": info.get("profitMargins"),
                "analyst_target_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey"),
            }
        except Exception as e:
            raise ValueError(f"Error retrieving info for ticker {ticker}: {str(e)}")

    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y", interval: str = "1wk") -> str:
        """
        Get historical price data for a stock ticker.
        Avoid overloading by using a wide interval on longer periods. e.g. Don't use 1y with 1d interval, or 1mo with a 1h interval.

        IMPORTANT: Avoid overloading the API by using appropriate interval for period length.
        For example:
        - Don't use 1y period with intervals less than 1d.
        - Don't use 6mo period with intervals less than 5d.
        - For periods > 60d, use daily intervals or longer

        Valid periods: 1d, 5d, 7d, 60d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

        Args:
            ticker (str): The stock ticker symbol.
            period (str): The time period to retrieve data for.
            interval (str): The interval between data points.

        Returns:
            str: A JSON string containing historical price data.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yf = check_yfinance()
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            # Reset index and ensure datetime handling is correct
            data = data.copy()
            data.index = data.index.strftime('%Y-%m-%dT%H:%M:%S%z')
            data_dict = data.reset_index().to_dict(orient='records')
            
            # Round price values
            for record in data_dict:
                for key in ['Open', 'High', 'Low', 'Close']:
                    if key in record:
                        record[key] = round(float(record[key]), 2)
            
            # Serialize to JSON string
            return json.dumps(data_dict, default=str)
        except Exception as e:
            raise ValueError(f"Error retrieving historical data for ticker {ticker}: {str(e)}")

    @staticmethod
    def calculate_returns(tickers: Union[str, List[str]], period: str = "1mo", interval: str = "1d"):
        """
        Calculate daily returns for given stock ticker(s).

        Note: Avoid overloading the API by using appropriate interval for period length.
        For example:
        - Don't use 1y period with 1m/5m intervals
        - Don't use 6mo period with intraday intervals
        - For periods > 60d, use daily intervals or longer        

        Args:
            tickers (Union[str, List[str]]): The stock ticker symbol or a list of symbols.
            period (str): The time period to retrieve data for (e.g., "1d", "1mo", "1y", "ytd").
            interval (str): The interval between data points (e.g., "1wk", "1mo").

        Returns:
            Dict[str, Series]: A dictionary where keys are ticker symbols and values are Series containing daily returns.

        Raises:
            ValueError: If data cannot be retrieved for any ticker or if the DataFrame structure is unexpected.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        returns = {}
        try:
            check_yfinance()
            data = YahooFinanceTools.download_multiple_tickers(tickers, period=period, interval=interval)
            
            if data.empty:
                raise ValueError("No data returned from download_multiple_tickers")
            
            for ticker in tickers:
                if ('Close' in data.columns.get_level_values('Price') and 
                    ticker in data.columns.get_level_values('Ticker')):
                    returns[ticker] = data[ticker]['Close'].pct_change()
                else:
                    raise ValueError(f"'Close' column not found for ticker {ticker}")
            
            return returns
        except Exception as e:
            raise ValueError(f"Error calculating returns for tickers {tickers}: {str(e)}")

    @staticmethod
    def get_financials(ticker: str, statement: str = "income"):
        """
        Get financial statements for a stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            statement (str): The type of financial statement ("income", "balance", or "cash").

        Returns:
            DataFrame: A DataFrame containing the requested financial statement.

        Raises:
            ValueError: If the ticker is invalid, data cannot be retrieved, or an invalid statement type is provided.
        """
        try:
            yf = check_yfinance()
            stock = yf.Ticker(ticker)
            if statement == "income":
                return stock.financials
            elif statement == "balance":
                return stock.balance_sheet
            elif statement == "cash":
                return stock.cashflow
            else:
                raise ValueError("Invalid statement type. Choose 'income', 'balance', or 'cash'.")
        except Exception as e:
            raise ValueError(f"Error retrieving {statement} statement for ticker {ticker}: {str(e)}")

    @staticmethod
    def get_recommendations(ticker: str):
        """
        Get analyst recommendations for a stock ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            DataFrame: A DataFrame containing analyst recommendations.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yf = check_yfinance()
            stock = yf.Ticker(ticker)
            return stock.recommendations
        except Exception as e:
            raise ValueError(f"Error retrieving recommendations for ticker {ticker}: {str(e)}")

    @staticmethod
    def download_multiple_tickers(tickers: List[str], period: str = "1mo", interval: str = "1d"):
        """
        Download historical data for multiple tickers simultaneously.
        IMPORTANT: Avoid overloading the API by using appropriate interval for period length.

        Safe combinations:
        - Short-term (≤ 7 days):
            ✓ period="1d" with interval="1m" to "90m"
            ✓ period="5d" with interval="15m" to "1h"
            ✓ period="7d" with interval="30m" to "1d"
        
        - Medium-term (≤ 60 days):
            ✓ period="1mo" with interval="1h" to "1d"
            ✓ period="2mo" with interval="1d" to "5d"
            ✓ period="60d" with interval="1d" to "1wk"
        
        - Long-term (> 60 days):
            ✓ period="6mo" with interval="1d" to "1mo"
            ✓ period="1y" with interval="1wk" to "3mo"
            ✓ period="2y" with interval="1wk" to "3mo"

        Unsafe combinations (will raise error):
        ✗ period="1y" with interval < "1d"
        ✗ period="6mo" with interval < "1d"
        ✗ period="3mo" with interval < "1h"

        Args:
            tickers (List[str]): A list of stock ticker symbols
            period (str): Time period. Valid options: 1d,5d,7d,60d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): Data interval. Valid options: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

        Returns:
            DataFrame: Multi-level DataFrame containing historical price data for all tickers,
                     grouped by ticker symbol

        Raises:
            ValueError: If period/interval combination is unsafe or data cannot be retrieved
        """
        try:
            yf = check_yfinance()
            data = yf.download(" ".join(tickers), period=period, interval=interval, group_by="ticker")
            return data
        except Exception as e:
            raise ValueError(f"Error downloading data for tickers {tickers}: {str(e)}")

    @staticmethod
    def get_asset_profile(ticker: str) -> Dict[str, Any]:
        """
        Get the asset profile for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            Dict[str, Any]: A dictionary containing the asset profile information.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yahoofinance = check_yahoofinance()
            profile = yahoofinance.AssetProfile(ticker)
            return profile.to_dfs()
        except Exception as e:
            raise ValueError(f"Error retrieving asset profile for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def get_balance_sheet(ticker: str, quarterly: bool = False):
        """
        Get the balance sheet for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            quarterly (bool): If True, retrieve quarterly data; if False, retrieve annual data.

        Returns:
            pd.DataFrame: A DataFrame containing the balance sheet data.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yahoofinance = check_yahoofinance()
            if quarterly:
                balance_sheet = yahoofinance.BalanceSheetQuarterly(ticker)
            else:
                balance_sheet = yahoofinance.BalanceSheet(ticker)
            return balance_sheet.to_dfs()['Balance Sheet']
        except Exception as e:
            raise ValueError(f"Error retrieving balance sheet for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def get_cash_flow(ticker: str, quarterly: bool = False):
        """
        Get the cash flow statement for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            quarterly (bool): If True, retrieve quarterly data; if False, retrieve annual data.

        Returns:
            pd.DataFrame: A DataFrame containing the cash flow statement data.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yahoofinance = check_yahoofinance()
            if quarterly:
                cash_flow = yahoofinance.CashFlowQuarterly(ticker)
            else:
                cash_flow = yahoofinance.CashFlow(ticker)
            return cash_flow.to_dfs()['Cash Flow']
        except Exception as e:
            raise ValueError(f"Error retrieving cash flow statement for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def get_income_statement(ticker: str, quarterly: bool = False):
    
        """
        Get the income statement for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            quarterly (bool): If True, retrieve quarterly data; if False, retrieve annual data.

        Returns:
            pd.DataFrame: A DataFrame containing the income statement data.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            yahoofinance = check_yahoofinance()
            if quarterly:
                income_statement = yahoofinance.IncomeStatementQuarterly(ticker)
            else:
                income_statement = yahoofinance.IncomeStatement(ticker)
            return income_statement.to_dfs()['Income Statement']
        except Exception as e:
            raise ValueError(f"Error retrieving income statement for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def get_custom_historical_data(ticker: str, start_date: str, end_date: str, 
                                frequency: str = '1d', event: str = 'history'):
        """
        Get custom historical data for a stock ticker with specified parameters.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for the query (format: 'YYYY-MM-DD').
            end_date (str): The end date for the query (format: 'YYYY-MM-DD').
            frequency (str): The data frequency ('1d', '1wk', or '1mo'). Default is '1d'.
            event (str): The type of event data to retrieve. Default is 'history'.

        Returns:
            pd.DataFrame: A DataFrame containing the custom historical data.

        Raises:
            ValueError: If the ticker is invalid, dates are incorrect, or data cannot be retrieved.
        """
        try:
            yahoofinance = check_yahoofinance()
            historical_data = yahoofinance.HistoricalPrices(
                ticker, start_date, end_date, 
                frequency=yahoofinance.DataFrequency(frequency),
                event=yahoofinance.DataEvent(event)
            )
            return historical_data.to_dfs()
        except Exception as e:
            raise ValueError(f"Error retrieving custom historical data for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def technical_analysis(ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Perform technical analysis for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.
            period (str): The time period for historical data (e.g., "1mo", "3mo", "1y").

        Returns:
            Dict[str, Any]: A dictionary containing various technical analysis indicators.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            # Get historical data
            data = YahooFinanceTools.get_historical_data(ticker, period)
            
            # Calculate moving averages
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # Calculate Relative Strength Index (RSI)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
            data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
            
            latest = data.iloc[-1]
            
            return {
                "current_price": latest['Close'],
                "sma_50": latest['SMA_50'],
                "sma_200": latest['SMA_200'],
                "rsi": latest['RSI'],
                "macd": latest['MACD'],
                "macd_signal": latest['Signal_Line'],
                "bollinger_upper": latest['BB_Upper'],
                "bollinger_middle": latest['BB_Middle'],
                "bollinger_lower": latest['BB_Lower'],
                "volume": latest['Volume']
            }
        except Exception as e:
            raise ValueError(f"Error performing technical analysis for ticker {ticker}: {str(e)}")
        
    @staticmethod
    def fundamental_analysis(ticker: str) -> Dict[str, Any]:
        """
        Perform a comprehensive fundamental analysis for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            Dict[str, Any]: A dictionary containing various fundamental analysis metrics.

        Raises:
            ValueError: If the ticker is invalid or data cannot be retrieved.
        """
        try:
            # Get basic info
            info = YahooFinanceTools.get_ticker_info(ticker)
            
            # Get financial statements
            income_statement = YahooFinanceTools.get_income_statement(ticker)
            balance_sheet = YahooFinanceTools.get_balance_sheet(ticker)
            cash_flow = YahooFinanceTools.get_cash_flow(ticker)
            
            # Calculate additional metrics
            latest_year = income_statement.columns[0]
            
            revenue = income_statement.loc['Total Revenue', latest_year]
            net_income = income_statement.loc['Net Income', latest_year]
            total_assets = balance_sheet.loc['Total Assets', latest_year]
            total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_year]
            total_equity = balance_sheet.loc['Total Equity Gross Minority Interest', latest_year]
            
            # Return on Equity (ROE)
            roe = net_income / total_equity
            
            # Return on Assets (ROA)
            roa = net_income / total_assets
            
            # Debt to Equity Ratio
            debt_to_equity = total_liabilities / total_equity
            
            # Current Ratio
            current_assets = balance_sheet.loc['Current Assets', latest_year]
            current_liabilities = balance_sheet.loc['Current Liabilities', latest_year]
            current_ratio = current_assets / current_liabilities
            
            # Free Cash Flow
            operating_cash_flow = cash_flow.loc['Operating Cash Flow', latest_year]
            capital_expenditures = cash_flow.loc['Capital Expenditure', latest_year]
            free_cash_flow = operating_cash_flow - capital_expenditures
            
            return {
                **info,
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_equity": total_equity,
                "return_on_equity": roe,
                "return_on_assets": roa,
                "debt_to_equity_ratio": debt_to_equity,
                "current_ratio": current_ratio,
                "free_cash_flow": free_cash_flow
            }
        except Exception as e:
            raise ValueError(f"Error performing fundamental analysis for ticker {ticker}: {str(e)}")