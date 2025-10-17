import yfinance as yf
import pandas as pd
import numpy as np
import re
import time
from typing import Optional, Dict, Any
from config import (
    PROFITABILITY_THRESHOLD, USE_DECODO_API, TREND_DETECTION_ENABLED,
    MA_HISTORY_DAYS, USE_CQG_API, PB_THRESHOLD, ENABLE_PB_FILTER
)
from decodo_api import DecodoAPIClient
import technical_analysis
import cqg_api
import logging

logger = logging.getLogger(__name__)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 0.5):
    """
    Retry helper with exponential backoff for transient error handling.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds

    Returns:
        Function result or None if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            result = func()
            if result is not None:
                return result
        except Exception as e:
            if attempt == max_retries:
                print(f"Max retries exceeded for function {func.__name__}: {str(e)}")
                return None

            # Calculate exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + np.random.uniform(0, 0.1)
            print(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s delay...")
            time.sleep(delay)

    return None


def analyze_company_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Analyze company fundamentals including profitability (ROE), valuation (P/E, P/B),
    and technical indicators (moving averages, trend).

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
            'book_value_per_share': float | None,
            'pb_ratio': float | None,
            'ma5': float | None,
            'ma10': float | None,
            'ma20': float | None,
            'trend': str | None,
            'trend_label': str | None,
            'historical_prices': pd.DataFrame | None,
            'profitable': bool,
            'quarter_end': str,
            'data_source': str  # 'decodo' or 'yfinance'
        }

    Note:
        ROE (Return on Equity) is calculated as (Net Income / Shareholders' Equity) × 100%
        P/E ratio (Price-to-Earnings) is calculated as Current Price / Trailing EPS
        P/B ratio (Price-to-Book) is calculated as Current Price / Book Value per Share
        Moving averages and trend are calculated when TREND_DETECTION_ENABLED is True
        All financial ratios may be None if required data is unavailable.
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


def _analyze_with_yfinance(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Analyze company fundamentals using yfinance (fallback method).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with analysis results or None if data unavailable
    """
    try:
        # Create ticker object with retry
        def get_ticker():
            return yf.Ticker(ticker)

        stock = retry_with_backoff(get_ticker)
        if stock is None:
            print(f"Warning: Failed to create ticker object for {ticker}")
            return None

        # Get quarterly financial data with retry
        def get_financials():
            return stock.quarterly_financials

        quarterly_financials = retry_with_backoff(get_financials)
        if quarterly_financials is None or quarterly_financials.empty:
            print(f"Warning: No quarterly financial data available for {ticker}")
            return None

        # Get quarterly balance sheet data with retry
        def get_balance_sheet():
            return stock.quarterly_balance_sheet

        quarterly_balance_sheet = retry_with_backoff(get_balance_sheet)

        # Deterministic latest quarter selection from quarterly_financials
        fin_cols = list(quarterly_financials.columns)
        if not fin_cols:
            print(f"Warning: No columns in quarterly financials for {ticker}")
            return None

        # Determine latest_fin_col deterministically
        try:
            # Check if columns are datetime-like
            first_col = fin_cols[0]
            if hasattr(first_col, 'strftime') or hasattr(first_col, 'toordinal'):
                latest_fin_col = max(fin_cols)
            else:
                # Parse to datetime for deterministic selection
                parsed_fin = pd.to_datetime(fin_cols, errors='coerce')
                valid_dates = parsed_fin[~pd.isna(parsed_fin)]
                if valid_dates.empty:
                    print(f"Warning: Could not parse any dates from quarterly financials columns for {ticker}")
                    return None
                latest_fin_col = fin_cols[valid_dates.idxmax()]
        except Exception as e:
            print(f"Warning: Error determining latest financial column for {ticker}: {e}")
            return None

        # Set latest_quarter and quarter_end from latest_fin_col
        latest_quarter = quarterly_financials[latest_fin_col]
        if hasattr(latest_fin_col, 'strftime'):
            quarter_end = latest_fin_col.strftime('%Y-%m-%d')
        else:
            quarter_end = str(latest_fin_col)

        # Balance sheet alignment with deterministic fallback
        latest_balance_quarter = None
        if quarterly_balance_sheet.empty:
            print(f"Warning: No balance sheet data available for {ticker}")
        else:
            # Try to align with latest_fin_col
            if latest_fin_col in quarterly_balance_sheet.columns:
                latest_balance_quarter = quarterly_balance_sheet[latest_fin_col]
            else:
                # Deterministic alternative: choose nearest prior or latest from balance sheet
                try:
                    bs_cols = list(quarterly_balance_sheet.columns)
                    if hasattr(bs_cols[0], 'strftime') or hasattr(bs_cols[0], 'toordinal'):
                        # Both datetime-like - find nearest prior or latest
                        parsed_bs = pd.to_datetime(bs_cols, errors='coerce')
                        valid_bs_dates = parsed_bs[~pd.isna(parsed_bs)]

                        if not valid_bs_dates.empty:
                            if hasattr(latest_fin_col, 'strftime') or hasattr(latest_fin_col, 'toordinal'):
                                # Both datetime-like - find nearest prior
                                latest_fin_datetime = pd.to_datetime(latest_fin_col)
                                prior_dates = valid_bs_dates[valid_bs_dates <= latest_fin_datetime]
                                if not prior_dates.empty:
                                    selected_bs_date = prior_dates.idxmax()
                                else:
                                    selected_bs_date = valid_bs_dates.idxmax()
                                    print(f"Warning: Balance sheet dates after financial date for {ticker}; using latest BS date")
                            else:
                                # Financial date not datetime-like - use latest BS
                                selected_bs_date = valid_bs_dates.idxmax()

                            latest_balance_quarter = quarterly_balance_sheet[bs_cols[valid_bs_dates.index.get_loc(selected_bs_date)]]
                            print(f"Warning: Balance sheet not aligned for {ticker}; using {selected_bs_date}")
                        else:
                            print(f"Warning: Could not parse any dates from balance sheet columns for {ticker}")
                    else:
                        # Use deterministic selection from balance sheet columns
                        parsed_bs = pd.to_datetime(bs_cols, errors='coerce')
                        valid_bs_dates = parsed_bs[~pd.isna(parsed_bs)]
                        if not valid_bs_dates.empty:
                            latest_bs_col = bs_cols[valid_bs_dates.idxmax()]
                            latest_balance_quarter = quarterly_balance_sheet[latest_bs_col]
                            print(f"Warning: Balance sheet not aligned for {ticker}; using latest BS column {latest_bs_col}")
                        else:
                            print(f"Warning: Could not parse balance sheet dates for {ticker}")
                except Exception as e:
                    print(f"Warning: Error aligning balance sheet for {ticker}: {e}")

        # Extract net income with prioritized labels and regex fallback
        net_income = None
        net_income_labels = [
            'Net Income',
            'Net Income Continuing Operations',
            'Net Income from Continuing Operations',
            'Net Income Applicable to Common Shares',
            'Net Income After Tax',
            'Consolidated Net Income'
        ]

        # Try prioritized labels first
        for label in net_income_labels:
            if label in latest_quarter.index:
                net_income = latest_quarter[label]
                break

        # If still not found, use regex fallback anchored at start of label
        if net_income is None:
            for idx in latest_quarter.index:
                idx_str = str(idx).strip()
                # Regex anchored at start: net income (case insensitive)
                if re.match(r'^\s*net\s+income', idx_str, re.IGNORECASE):
                    net_income = latest_quarter[idx]
                    break

        if net_income is None:
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

        # Get company name with retry
        def get_company_info():
            return stock.info

        company_info = retry_with_backoff(get_company_info)
        company_name = company_info.get('longName', ticker) if company_info else ticker

        # Fetch current stock price with retry
        current_price = None
        def get_current_price():
            price = company_info.get('currentPrice') or company_info.get('regularMarketPrice')
            if price is None or price <= 0:
                # Fallback to historical data
                hist_data = stock.history(period='1d')
                if not hist_data.empty:
                    price = hist_data['Close'].iloc[-1]
            return price

        try:
            current_price = retry_with_backoff(get_current_price)
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

        # Calculate Book Value per Share and P/B ratio
        book_value_per_share = None
        pb_ratio = None

        try:
            # Try to get book value per share directly from stock.info
            book_value_direct = company_info.get('bookValue') if company_info else None
            if book_value_direct is not None and book_value_direct > 0:
                book_value_per_share = float(book_value_direct)
            else:
                # Calculate from equity and shares outstanding
                if equity is not None and equity > 0:
                    # Try multiple shares outstanding fields
                    shares_fields = [
                        'sharesOutstanding',
                        'impliedSharesOutstanding',
                        'commonStockSharesOutstanding',
                        'shareOutstanding'
                    ]

                    shares_outstanding = None
                    for field in shares_fields:
                        shares_value = company_info.get(field) if company_info else None
                        if shares_value is not None and shares_value > 0:
                            shares_outstanding = float(shares_value)
                            break

                    if shares_outstanding is not None and shares_outstanding > 0:
                        book_value_per_share = equity / shares_outstanding
                    else:
                        print(f"Warning: Could not find valid shares outstanding data for {ticker}")
                else:
                    print(f"Warning: No valid equity data for P/B calculation for {ticker}")

            # Calculate P/B ratio if book value per share is available and positive
            if (book_value_per_share is not None and book_value_per_share > 0 and
                current_price is not None and current_price > 0):
                pb_ratio = current_price / book_value_per_share
            elif book_value_per_share is not None:
                print(f"Warning: Cannot calculate P/B ratio for {ticker} (book value: {book_value_per_share:.2f}, price: {current_price})")
            else:
                print(f"Warning: No valid book value data for P/B calculation for {ticker}")

        except (ValueError, TypeError, ZeroDivisionError) as e:
            print(f"Warning: Error calculating P/B ratio for {ticker}: {str(e)}")
            book_value_per_share = None
            pb_ratio = None

        # Technical Analysis
        technical_indicators = {
            'ma5': None,
            'ma10': None,
            'ma20': None,
            'trend': 'unknown',
            'trend_label': technical_analysis.get_trend_label('unknown'),
            'historical_prices': None
        }

        if TREND_DETECTION_ENABLED:
            try:
                historical_prices = None

                # Try CQG API first if enabled
                if USE_CQG_API:
                    try:
                        cqg_client = cqg_api.create_cqg_client()
                        if cqg_client.is_configured():
                            historical_prices = cqg_client.get_historical_prices(ticker, MA_HISTORY_DAYS)
                            if historical_prices is not None:
                                logger.info(f"Retrieved {len(historical_prices)} days from CQG for {ticker}")
                            else:
                                logger.warning(f"CQG API returned no data for {ticker}")
                        else:
                            logger.info(f"CQG API not configured for {ticker}")
                    except Exception as e:
                        logger.warning(f"CQG API error for {ticker}: {e}")

                # Fallback to yfinance if CQG failed or is disabled
                if historical_prices is None:
                    try:
                        hist_data = stock.history(period=f"{MA_HISTORY_DAYS}d")
                        if not hist_data.empty:
                            historical_prices = hist_data
                            logger.info(f"Retrieved {len(historical_prices)} days from yfinance for {ticker}")
                        else:
                            logger.warning(f"No historical data available from yfinance for {ticker}")
                    except Exception as e:
                        logger.warning(f"Error fetching historical data from yfinance for {ticker}: {e}")

                # Analyze technical indicators if we have historical data
                if historical_prices is not None and not historical_prices.empty:
                    technical_indicators = technical_analysis.analyze_technical_indicators(historical_prices)
                    if technical_indicators:
                        logger.info(f"Technical analysis completed for {ticker}: trend={technical_indicators.get('trend')}")
                    else:
                        logger.warning(f"Technical analysis failed for {ticker}")
                else:
                    logger.warning(f"No historical data available for technical analysis of {ticker}")

            except Exception as e:
                logger.error(f"Error in technical analysis for {ticker}: {e}")
                # Continue with None values for technical indicators
                technical_indicators = {
                    'ma5': None,
                    'ma10': None,
                    'ma20': None,
                    'trend': 'unknown',
                    'trend_label': technical_analysis.get_trend_label('unknown'),
                    'historical_prices': None
                }

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
            'book_value_per_share': float(book_value_per_share) if book_value_per_share is not None else None,
            'pb_ratio': float(pb_ratio) if pb_ratio is not None else None,
            'ma5': technical_indicators.get('ma5'),
            'ma10': technical_indicators.get('ma10'),
            'ma20': technical_indicators.get('ma20'),
            'trend': technical_indicators.get('trend'),
            'trend_label': technical_indicators.get('trend_label'),
            'historical_prices': technical_indicators.get('historical_prices'),
            'profitable': profitable,
            'quarter_end': quarter_end,
            'data_source': 'yfinance'
        }

    except Exception as e:
        print(f"Error analyzing {ticker} with yfinance: {str(e)}")
        return None