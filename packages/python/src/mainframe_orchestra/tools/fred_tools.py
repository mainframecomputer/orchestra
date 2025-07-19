# Copyright 2025 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import Any, Dict, List, Optional

from ..utils.braintrust_utils import traced


class FredTools:
    _pandas = None
    _Fred = None

    @classmethod
    def _get_pandas(cls):
        """Lazy load pandas with proper error handling."""
        if cls._pandas is None:
            try:
                import pandas as pd  # type: ignore

                cls._pandas = pd
            except ImportError as e:
                raise ImportError(
                    "pandas is required for FredTools. " "Install with: 'pip install pandas'"
                ) from e
        return cls._pandas

    @classmethod
    def _get_fred(cls):
        """Lazy load fredapi with proper error handling."""
        if cls._Fred is None:
            try:
                from fredapi import Fred  # type: ignore

                cls._Fred = Fred
            except ImportError as e:
                raise ImportError(
                    "fredapi is required for FredTools. " "Install with: 'pip install fredapi'"
                ) from e
        return cls._Fred

    @classmethod
    def check_dependencies(cls) -> Dict[str, bool]:
        """
        Check if required dependencies are available.

        Returns:
            Dict[str, bool]: Dictionary with dependency names as keys and availability as values.
        """
        dependencies = {}

        try:
            cls._get_pandas()
            dependencies["pandas"] = True
        except ImportError:
            dependencies["pandas"] = False

        try:
            cls._get_fred()
            dependencies["fredapi"] = True
        except ImportError:
            dependencies["fredapi"] = False

        return dependencies

    @traced(type="tool")
    @classmethod
    def economic_indicator_analysis(
        cls, indicator_ids: List[str], start_date: str, end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Perform a comprehensive analysis of economic indicators.

        Args:
            indicator_ids (List[str]): List of economic indicator series IDs.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results for each indicator.
            None: If required dependencies are not available.
        """
        try:
            Fred = cls._get_fred()
        except ImportError:
            print("Error: fredapi is not installed. Please install it using 'pip install fredapi'.")
            return None

        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        results = {}

        for indicator_id in indicator_ids:
            series = fred.get_series(
                indicator_id, observation_start=start_date, observation_end=end_date
            )
            series = series.dropna()

            if len(series) > 0:
                pct_change = series.pct_change()
                annual_change = series.resample("YE").last().pct_change()

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
                    "last_annual_change": annual_change.iloc[-1],
                }
            else:
                results[indicator_id] = None

        return results

    @traced(type="tool")
    @classmethod
    def yield_curve_analysis(
        cls, treasury_maturities: List[str], start_date: str, end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Perform an analysis of the US Treasury yield curve.

        Args:
            treasury_maturities (List[str]): List of Treasury maturity series IDs.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the yield curve analysis results.
            None: If required dependencies are not available.
        """
        try:
            pd = cls._get_pandas()
            Fred = cls._get_fred()
        except ImportError as e:
            if "pandas" in str(e):
                print(
                    "Error: pandas is not installed. Please install it using 'pip install pandas'."
                )
            else:
                print(
                    "Error: fredapi is not installed. Please install it using 'pip install fredapi'."
                )
            return None

        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        yield_data = {}

        for maturity in treasury_maturities:
            series = fred.get_series(
                maturity, observation_start=start_date, observation_end=end_date
            )
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
                "inverted_yield_curve": yield_curve_slopes_df.min().min() < 0,
            }
        else:
            results = None

        return results

    @traced(type="tool")
    @classmethod
    def economic_news_sentiment_analysis(
        cls, news_series_id: str, start_date: str, end_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Perform sentiment analysis on economic news series.

        Args:
            news_series_id (str): Economic news series ID.
            start_date (str): Start date for the analysis (YYYY-MM-DD).
            end_date (str): End date for the analysis (YYYY-MM-DD).

        Returns:
            Dict[str, Any]: A dictionary containing the sentiment analysis results.
            None: If required dependencies are not available.
        """
        try:
            Fred = cls._get_fred()
        except ImportError:
            print("Error: fredapi is not installed. Please install it using 'pip install fredapi'.")
            return None

        fred = Fred(api_key=os.getenv("FRED_API_KEY"))

        series = fred.get_series(
            news_series_id, observation_start=start_date, observation_end=end_date
        )
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
                "net_sentiment_score": sentiment_scores.sum(),
            }
        else:
            results = None

        return results
