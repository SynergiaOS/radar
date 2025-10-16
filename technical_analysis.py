"""
Technical Analysis Module for GPW Smart Analyzer

This module provides technical analysis functions including moving averages,
trend detection, and related indicators for stock price analysis.

Author: GPW Smart Analyzer Team
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from config import TREND_LABELS


def calculate_moving_averages(prices: pd.Series, periods: list = [5, 10, 20]) -> Dict[str, Optional[float]]:
    """
    Calculate moving averages for given periods.

    Args:
        prices: Series of closing prices
        periods: List of periods for moving averages

    Returns:
        Dictionary with moving averages for each period
    """
    result = {}

    # Validate inputs
    if prices is None or prices.empty:
        for period in periods:
            result[f'ma{period}'] = None
        return result

    for period in periods:
        if len(prices) >= period:
            try:
                ma = prices.rolling(window=period).mean().iloc[-1]
                result[f'ma{period}'] = float(ma) if pd.notna(ma) and ma > 0 else None
            except (ValueError, TypeError, IndexError):
                result[f'ma{period}'] = None
        else:
            result[f'ma{period}'] = None

    return result


def detect_trend(current_price: float, ma5: Optional[float], ma10: Optional[float],
                 ma20: Optional[float]) -> str:
    """
    Detect trend based on moving average positioning.

    Args:
        current_price: Current stock price
        ma5: 5-day moving average
        ma10: 10-day moving average
        ma20: 20-day moving average

    Returns:
        Trend string: 'upward', 'downward', 'sideways', or 'unknown'
    """
    # Validate current price
    if current_price is None or current_price <= 0:
        return 'unknown'

    # If any key MAs are missing or invalid, default to unknown/sideways
    if (ma5 is None or ma10 is None or ma20 is None or
        ma5 <= 0 or ma10 <= 0 or ma20 <= 0):
        return 'unknown'

    # Check for upward trend: price > MA5 > MA10 > MA20
    if current_price > ma5 > ma10 > ma20:
        return 'upward'

    # Check for downward trend: price < MA5 < MA10 < MA20
    elif current_price < ma5 < ma10 < ma20:
        return 'downward'

    # Otherwise, sideways trend
    else:
        return 'sideways'


def get_trend_label(trend_key: str) -> str:
    """
    Get Polish label for trend.

    Args:
        trend_key: Trend key ('upward', 'downward', 'sideways', 'unknown')

    Returns:
        Polish trend label
    """
    return TREND_LABELS.get(trend_key, 'Nieznany')


def analyze_technical_indicators(historical_prices_df: pd.DataFrame) -> Dict[str, Union[float, str, pd.DataFrame, None]]:
    """
    Analyze technical indicators for a stock.

    Args:
        historical_prices_df: DataFrame with historical price data

    Returns:
        Dictionary containing technical indicators
    """
    # Validate input DataFrame
    if (historical_prices_df is None or
        historical_prices_df.empty or
        'Close' not in historical_prices_df.columns or
        len(historical_prices_df) < 5):
        return {
            'ma5': None,
            'ma10': None,
            'ma20': None,
            'trend': 'unknown',
            'trend_label': get_trend_label('unknown'),
            'historical_prices': None
        }

    # Ensure we have a DatetimeIndex
    if not isinstance(historical_prices_df.index, pd.DatetimeIndex):
        historical_prices_df.index = pd.to_datetime(historical_prices_df.index)

    # Extract closing prices
    close_prices = historical_prices_df['Close']

    # Calculate moving averages
    moving_averages = calculate_moving_averages(close_prices)

    # Get current price (latest close)
    current_price = float(close_prices.iloc[-1])

    # Detect trend
    trend = detect_trend(
        current_price,
        moving_averages.get('ma5'),
        moving_averages.get('ma10'),
        moving_averages.get('ma20')
    )

    # Get Polish trend label
    trend_label = get_trend_label(trend)

    return {
        'ma5': moving_averages.get('ma5'),
        'ma10': moving_averages.get('ma10'),
        'ma20': moving_averages.get('ma20'),
        'trend': trend,
        'trend_label': trend_label,
        'historical_prices': historical_prices_df
    }


def validate_historical_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate historical price data.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"

    if df.empty:
        return False, "DataFrame is empty"

    if 'Close' not in df.columns:
        return False, "Missing 'Close' column"

    if len(df) < 5:
        return False, "Insufficient data points (minimum 5 required)"

    # Check for valid price data
    if df['Close'].isna().all():
        return False, "All Close prices are NaN"

    return True, ""


def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)

    Returns:
        RSI value or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, Optional[float]]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices
        period: Period for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2)

    Returns:
        Dictionary with upper, middle, and lower bands
    """
    if len(prices) < period:
        return {
            'upper': None,
            'middle': None,
            'lower': None
        }

    # Calculate middle band (SMA)
    middle = prices.rolling(window=period).mean()

    # Calculate standard deviation
    std = prices.rolling(window=period).std()

    # Calculate upper and lower bands
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    latest_values = {
        'upper': float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else None,
        'middle': float(middle.iloc[-1]) if pd.notna(middle.iloc[-1]) else None,
        'lower': float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else None
    }

    return latest_values