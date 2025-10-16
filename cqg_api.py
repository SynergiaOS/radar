"""
CQG API Module for GPW Smart Analyzer

This module provides integration with CQG API for high-quality financial data
when available, with graceful fallback to alternative data sources.

Author: GPW Smart Analyzer Team
License: MIT
"""

import pandas as pd
import requests
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from config import USE_CQG_API, CQG_API_URL, CQG_API_KEY, CQG_API_SECRET, CQG_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CQGAPIClient:
    """
    Client for interacting with CQG API for financial data.
    """

    def __init__(self):
        """Initialize CQG API client with configuration."""
        self.base_url = CQG_API_URL
        self.api_key = CQG_API_KEY
        self.api_secret = CQG_API_SECRET
        self.timeout = CQG_TIMEOUT
        self.session = requests.Session()

        # Set up headers for authentication
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'GPW-Smart-Analyzer/1.0'
            })

    def is_configured(self) -> bool:
        """
        Check if CQG API is properly configured.

        Returns:
            True if API key and URL are configured and non-empty
        """
        return (
            USE_CQG_API and
            bool(self.api_key and self.api_key.strip()) and
            bool(self.base_url and self.base_url.strip())
        )

    def get_historical_prices(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data or None on failure
        """
        if not self.is_configured():
            logger.info("CQG API not configured, skipping data fetch")
            return None

        try:
            # Prepare request parameters
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            params = {
                'symbol': ticker,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'interval': 'daily',
                'fields': 'open,high,low,close,volume'
            }

            # Make API request
            url = f"{self.base_url}/api/v1/historical"
            response = self.session.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                return self._format_historical_data(data)
            else:
                logger.warning(f"CQG API request failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"CQG API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in CQG API call: {e}")
            return None

    def _format_historical_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Format raw CQG API data into standardized DataFrame.

        Args:
            raw_data: Raw data from CQG API

        Returns:
            Formatted DataFrame with OHLCV columns
        """
        try:
            # Extract price data from response
            if 'data' not in raw_data or not raw_data['data']:
                logger.warning("No price data in CQG API response")
                return None

            price_data = raw_data['data']

            # Create DataFrame
            df = pd.DataFrame(price_data)

            # Standardize column names
            column_mapping = {
                'date': 'Date',
                'datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }

            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"Missing required columns in CQG data: {missing_columns}")
                return None

            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Set Date as index
            df.set_index('Date', inplace=True)

            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add Volume column if missing
            if 'Volume' not in df.columns:
                df['Volume'] = 0

            # Sort by date
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error formatting CQG data: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test connection to CQG API.

        Returns:
            True if connection is successful
        """
        if not self.is_configured():
            return False

        try:
            # Test with a simple request
            url = f"{self.base_url}/api/v1/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"CQG API connection test failed: {e}")
            return False


def create_cqg_client() -> CQGAPIClient:
    """
    Create and return a CQG API client instance.

    Returns:
        CQGAPIClient instance
    """
    return CQGAPIClient()


def get_fallback_data(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Get fallback data when CQG API is not available.

    This is a placeholder for implementing fallback data sources
    such as Yahoo Finance or other APIs.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data

    Returns:
        DataFrame with OHLCV data or None
    """
    # This function can be extended to use alternative data sources
    # For now, it returns None to indicate no fallback data available
    logger.info(f"No fallback data source implemented for {ticker}")
    return None


def get_historical_prices(ticker: str, days: int = 30, use_cqg: bool = True) -> Optional[pd.DataFrame]:
    """
    Get historical prices with CQG API and fallback logic.

    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data
        use_cqg: Whether to try CQG API first

    Returns:
        DataFrame with OHLCV data or None
    """
    if use_cqg and USE_CQG_API:
        client = create_cqg_client()
        if client.is_configured():
            data = client.get_historical_prices(ticker, days)
            if data is not None:
                logger.info(f"Successfully retrieved {len(data)} days of CQG data for {ticker}")
                return data
            else:
                logger.warning(f"CQG API failed for {ticker}, trying fallback")

    # Try fallback data source
    fallback_data = get_fallback_data(ticker, days)
    if fallback_data is not None:
        logger.info(f"Using fallback data for {ticker}")
        return fallback_data

    logger.warning(f"No data available for {ticker}")
    return None


# Create a global client instance for reuse
_cqg_client = None

def get_cqg_client() -> CQGAPIClient:
    """
    Get or create a global CQG client instance.

    Returns:
        CQGAPIClient instance
    """
    global _cqg_client
    if _cqg_client is None:
        _cqg_client = create_cqg_client()
    return _cqg_client