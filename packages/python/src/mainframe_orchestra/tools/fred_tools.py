# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import Dict, Any, List
from ..utils.braintrust_utils import traced

def check_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("pandas is required for FredTools. Install with `pip install pandas`")

def check_fredapi():
    try:
        from fredapi import Fred
        return Fred
    except ImportError:
        raise ImportError("fredapi is required for FredTools. Install with `pip install fredapi`")


class FredTools:
    @traced(type="tool")
    @staticmethod
    def economic_indicator_analysis(indicator_ids: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of economic indicators.

        Args:
            indicator_ids (List[str]): List of economic indicator series IDs.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results for each indicator.
        """
        Fred = check_fredapi()
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))

        results = {}

        for indicator_id in indicator_ids:
            series = fred.get_series(indicator_id, observation_start=start_date, observation_end=end_date)
            series = series.dropna()

            if len(series) > 0:
                pct_change = series.pct_change()
                annual_change = series.resample('YE').last().pct_change()

                results[indicator_id] = {
                    "indicator": indicator_id,
                    "title": fred.get_series_info(indicator_id).title,
                    "start_date": start_date,
                    "end_date": end_date,
                    "min_value": series.min(),
                    "max_value": series.max(),
                    "mean_value": series.mean(),
                    "std_dev": series.std(),
                    "pct_change_mean": pct_change.mean(),
                    "pct_change_std": pct_change.std(),
                    "annual_change_mean": annual_change.mean(),
                    "annual_change_std": annual_change.std(),
                    "last_value": series.iloc[-1],
                    "last_pct_change": pct_change.iloc[-1],
                    "last_annual_change": annual_change.iloc[-1]
                }
            else:
                results[indicator_id] = None

        return results

    @traced(type="tool")
    @staticmethod
    def yield_curve_analysis(treasury_maturities: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Perform an analysis of the US Treasury yield curve.

        Args:
            treasury_maturities (List[str]): List of Treasury maturity series IDs.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the yield curve analysis results.
        """
        pd = check_pandas()
        Fred = check_fredapi()
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))

        yield_data = {}

        for maturity in treasury_maturities:
            series = fred.get_series(maturity, observation_start=start_date, observation_end=end_date)
            yield_data[maturity] = series

        yield_df = pd.DataFrame(yield_data)
        yield_df = yield_df.dropna()

        if len(yield_df) > 0:
            yield_curve_slopes = {}
            for i in range(len(treasury_maturities) - 1):
                short_maturity = treasury_maturities[i]
                long_maturity = treasury_maturities[i + 1]
                slope = yield_df[long_maturity] - yield_df[short_maturity]
                yield_curve_slopes[f"{short_maturity}_to_{long_maturity}"] = slope

            yield_curve_slopes_df = pd.DataFrame(yield_curve_slopes)

            results = {
                "start_date": start_date,
                "end_date": end_date,
                "yield_data": yield_df,
                "yield_curve_slopes": yield_curve_slopes_df,
                "inverted_yield_curve": yield_curve_slopes_df.min().min() < 0
            }
        else:
            results = None

        return results

    @traced(type="tool")
    @staticmethod
    def economic_news_sentiment_analysis(news_series_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on economic news series.

        Args:
            news_series_id (str): Economic news series ID.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the sentiment analysis results.
        """
        Fred = check_fredapi()
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))

        series = fred.get_series(news_series_id, observation_start=start_date, observation_end=end_date)
        series = series.dropna()

        if len(series) > 0:
            sentiment_scores = series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            sentiment_counts = sentiment_scores.value_counts()

            results = {
                "series_id": news_series_id,
                "title": fred.get_series_info(news_series_id).title,
                "start_date": start_date,
                "end_date": end_date,
                "positive_sentiment_count": sentiment_counts.get(1, 0),
                "negative_sentiment_count": sentiment_counts.get(-1, 0),
                "neutral_sentiment_count": sentiment_counts.get(0, 0),
                "net_sentiment_score": sentiment_scores.sum()
            }
        else:
            results = None

        return results
