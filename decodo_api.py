"""
Decodo API integration module for financial data scraping.
Replaces yfinance functionality with Decodo's professional scraping service.
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from config import DECODO_API_URL, DECODO_API_AUTH, DECODO_TIMEOUT, FINANCIAL_SOURCES


class DecodoAPIClient:
    """Client for interacting with Decodo's scraping API."""

    def __init__(self):
        self.api_url = DECODO_API_URL
        self.auth_header = {'Authorization': f'Basic {DECODO_API_AUTH}'}
        self.timeout = DECODO_TIMEOUT

    def scrape_url(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Scrape a single URL using Decodo API.

        Args:
            url: URL to scrape
            **kwargs: Additional parameters for scraping

        Returns:
            Scraped data as dictionary or None if failed
        """
        try:
            payload = {
                'url': url,
                **kwargs
            }

            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                **self.auth_header
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Decodo API returned status {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error making request to Decodo API: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {str(e)}")
            return None

    def scrape_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Scrape comprehensive financial data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'PKN.WA')

        Returns:
            Dictionary containing financial metrics or None if failed
        """
        # Construct Yahoo Finance URL for the ticker
        yahoo_url = f"https://finance.yahoo.com/quote/{ticker}"

        # Scrape Yahoo Finance page
        data = self.scrape_url(yahoo_url)

        if data:
            return self._parse_financial_data(data, ticker)
        else:
            return None

    def scrape_multiple_tickers(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape financial data for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of financial data dictionaries
        """
        results = []

        for ticker in tickers:
            print(f"Scraping {ticker} with Decodo API...", end=' ')
            data = self.scrape_financial_data(ticker)

            if data:
                results.append(data)
                print("✓")
            else:
                print("✗")

        return results

    def _parse_financial_data(self, raw_data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Parse raw scraped data into structured financial metrics.

        Args:
            raw_data: Raw data from Decodo API
            ticker: Ticker symbol

        Returns:
            Structured financial data dictionary
        """
        # This is a placeholder for data parsing logic
        # In a real implementation, you would parse the HTML/JSON content
        # extracted by Decodo to extract financial metrics

        financial_data = {
            'ticker': ticker,
            'name': None,  # Extract from scraped data
            'current_price': None,
            'eps': None,
            'pe_ratio': None,
            'net_income': None,
            'equity': None,
            'roe': None,
            'quarter_end': None,
            'profitable': False,
            'data_source': 'decodo'
        }

        # TODO: Implement actual parsing logic based on Decodo's response format
        # This would involve:
        # 1. Parsing HTML content to find financial tables
        # 2. Extracting specific metrics (P/E, EPS, etc.)
        # 3. Calculating derived metrics (ROE)
        # 4. Handling missing data gracefully

        return financial_data

    def get_company_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company profile data or None if failed
        """
        profile_url = f"https://finance.yahoo.com/quote/{ticker}/profile"

        data = self.scrape_url(profile_url)
        if data:
            # Parse profile data
            return {
                'ticker': ticker,
                'company_name': None,  # Extract from data
                'sector': None,
                'industry': None,
                'market_cap': None,
                'employees': None,
                'description': None
            }

        return None


def test_decodo_api():
    """Test function to verify Decodo API integration."""
    client = DecodoAPIClient()

    # Test with a sample URL
    print("Testing Decodo API with sample URL...")
    test_url = "https://finance.yahoo.com/quote/PKN.WA"
    result = client.scrape_url(test_url)

    if result:
        print("✅ Decodo API test successful!")
        print(f"Response keys: {list(result.keys())}")
        return True
    else:
        print("❌ Decodo API test failed!")
        return False


if __name__ == "__main__":
    test_decodo_api()