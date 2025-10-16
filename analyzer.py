import yfinance as yf
import pandas as pd
from config import PROFITABILITY_THRESHOLD, USE_DECODO_API
from decodo_api import DecodoAPIClient


def analyze_company_fundamentals(ticker: str) -> dict | None:
    """
    Analyze company fundamentals including profitability (ROE) and valuation (P/E ratio).

    Args:
        ticker: Stock ticker symbol (e.g., 'PKN.WA')

    Returns:
        Dictionary with analysis results or None if data unavailable:
        {
            'ticker': str,
            'name': str,
            'net_income': float,
            'equity': float | None,
            'roe': float | None,
            'current_price': float | None,
            'eps': float | None,
            'pe_ratio': float | None,
            'profitable': bool,
            'quarter_end': str,
            'data_source': str  # 'decodo' or 'yfinance'
        }

    Note:
        ROE (Return on Equity) is calculated as (Net Income / Shareholders' Equity) × 100%
        P/E ratio (Price-to-Earnings) is calculated as Current Price / Trailing EPS
        Both ROE and P/E may be None if required data is unavailable.
    """
    try:
        # Try Decodo API first if enabled, otherwise use yfinance
        if USE_DECODO_API:
            print(f"Using Decodo API for {ticker}...", end=' ')
            decodo_client = DecodoAPIClient()
            result = decodo_client.scrape_financial_data(ticker)

            if result:
                print("✓")
                return result
            else:
                print("✗ Falling back to yfinance...")
                return _analyze_with_yfinance(ticker)
        else:
            return _analyze_with_yfinance(ticker)
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None


def _analyze_with_yfinance(ticker: str) -> dict | None:
    """
    Analyze company fundamentals using yfinance (fallback method).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with analysis results or None if data unavailable
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)

        # Get quarterly financial data
        quarterly_financials = stock.quarterly_financials

        if quarterly_financials.empty:
            print(f"Warning: No quarterly financial data available for {ticker}")
            return None

        # Get quarterly balance sheet data
        quarterly_balance_sheet = stock.quarterly_balance_sheet

        # Find the most recent common column between financials and balance sheet
        latest_col = None
        if not quarterly_balance_sheet.empty:
            common_cols = set(quarterly_financials.columns) & set(quarterly_balance_sheet.columns)
            if common_cols:
                latest_col = max(common_cols)  # Get the most recent date
            else:
                latest_col = quarterly_financials.columns[0]  # Fall back to financials
        else:
            latest_col = quarterly_financials.columns[0]  # Fall back to financials

        # Get the latest quarter data
        latest_quarter = quarterly_financials[latest_col]
        latest_balance_quarter = quarterly_balance_sheet[latest_col] if not quarterly_balance_sheet.empty and latest_col in quarterly_balance_sheet.columns else None

        # Extract net income
        net_income = None
        if 'Net Income' in latest_quarter.index:
            net_income = latest_quarter['Net Income']
        elif 'Net Income Continuing Operations' in latest_quarter.index:
            net_income = latest_quarter['Net Income Continuing Operations']
        else:
            # Look for similar net income related rows
            net_income_rows = [idx for idx in latest_quarter.index if 'net income' in idx.lower()]
            if net_income_rows:
                net_income = latest_quarter[net_income_rows[0]]
            else:
                print(f"Warning: Could not find net income data for {ticker}")
                return None

        if net_income is None or pd.isna(net_income):
            print(f"Warning: Net income data is missing for {ticker}")
            return None

        # Extract shareholders equity from balance sheet with robust fallbacks
        equity = None
        if latest_balance_quarter is not None:
            equity_field_names = [
                'Total Stockholder Equity',
                'Stockholders Equity',
                'Total Equity',
                'Shareholders Equity',
                'Total Shareholders Equity',
                'Total Stockholders Equity',
                'Total Stockholders\' Equity',
                'Total Stockholder\'s Equity',
                'Shareholder Equity',
                'Total Shareholder Equity'
            ]

            for field_name in equity_field_names:
                if field_name in latest_balance_quarter.index:
                    try:
                        equity_value = latest_balance_quarter[field_name]
                        if not pd.isna(equity_value):
                            equity = float(equity_value)
                            break
                    except (ValueError, TypeError):
                        continue

            if equity is not None and equity <= 0:
                print(f"Warning: Equity non-positive for {ticker}")
                equity = None
            elif equity is None:
                print(f"Warning: Equity data not found for {ticker}")
        else:
            print(f"Warning: No balance sheet data available for {ticker}")

        # Get company name
        company_name = stock.info.get('longName', ticker)

        # Get quarter end date
        quarter_end = latest_quarter.name.strftime('%Y-%m-%d') if hasattr(latest_quarter.name, 'strftime') else str(latest_quarter.name)

        # Fetch current stock price
        current_price = None
        try:
            # Try to get current price from info
            current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            if current_price is None or current_price <= 0:
                # Fallback to historical data
                hist_data = stock.history(period='1d')
                if not hist_data.empty:
                    current_price = hist_data['Close'].iloc[-1]

            if current_price and current_price > 0:
                current_price = float(current_price)
            else:
                current_price = None
                print(f"Warning: Could not fetch valid current price for {ticker}")
        except Exception as e:
            print(f"Warning: Error fetching price data for {ticker}: {str(e)}")
            current_price = None

        # Fetch trailing EPS (Earnings Per Share)
        eps = None
        try:
            # Try multiple EPS field names
            eps_field_names = ['trailingEps', 'epsTrailingTwelveMonths', 'trailingEPS', 'eps']

            for field_name in eps_field_names:
                eps_value = stock.info.get(field_name)
                if eps_value is not None and not pd.isna(eps_value) and eps_value > 0:
                    eps = float(eps_value)
                    break

            if eps is None or eps <= 0:
                eps = None
                print(f"Warning: Could not fetch valid EPS data for {ticker}")
        except Exception as e:
            print(f"Warning: Error fetching EPS data for {ticker}: {str(e)}")
            eps = None

        # Calculate P/E ratio
        pe_ratio = None
        if current_price is not None and eps is not None and eps > 0:
            pe_ratio = current_price / eps
            # Validate P/E ratio - flag unusually high values
            if pe_ratio > 1000:
                print(f"Warning: Unusually high P/E ratio ({pe_ratio:.1f}) for {ticker} - possible data quality issue")
        elif current_price is not None and eps is not None:
            print(f"Warning: P/E ratio unavailable for {ticker} (price: {current_price}, EPS: {eps})")

        # Calculate ROE (Return on Equity) with numeric validation
        roe = None
        try:
            net_income_float = float(net_income)
            if equity is not None and equity > 0:
                roe = (net_income_float / equity) * 100
        except (ValueError, TypeError):
            print(f"Warning: Invalid net income data for {ticker}")
            return None

        # Determine profitability using the validated float value
        profitable = net_income_float > PROFITABILITY_THRESHOLD

        return {
            'ticker': ticker,
            'name': company_name,
            'net_income': net_income_float,
            'equity': equity,
            'roe': float(roe) if roe is not None else None,
            'current_price': float(current_price) if current_price is not None else None,
            'eps': float(eps) if eps is not None else None,
            'pe_ratio': float(pe_ratio) if pe_ratio is not None else None,
            'profitable': profitable,
            'quarter_end': quarter_end,
            'data_source': 'yfinance'
        }

    except Exception as e:
        print(f"Error analyzing {ticker} with yfinance: {str(e)}")
        return None