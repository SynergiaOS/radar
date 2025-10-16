import yfinance as yf
import pandas as pd
from config import PROFITABILITY_THRESHOLD


def analyze_profitability(ticker: str) -> dict | None:
    """
    Analyze profitability of a company based on its latest quarterly net income.

    Args:
        ticker: Stock ticker symbol (e.g., 'PKN.WA')

    Returns:
        Dictionary with analysis results or None if data unavailable:
        {
            'ticker': str,
            'name': str,
            'net_income': float,
            'profitable': bool,
            'quarter_end': str
        }
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)

        # Get quarterly financial data
        quarterly_financials = stock.quarterly_financials

        if quarterly_financials.empty:
            print(f"Warning: No quarterly financial data available for {ticker}")
            return None

        # Get the latest quarter (first column)
        latest_quarter = quarterly_financials.iloc[:, 0]

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

        # Get company name
        company_name = stock.info.get('longName', ticker)

        # Get quarter end date
        quarter_end = latest_quarter.name.strftime('%Y-%m-%d') if hasattr(latest_quarter.name, 'strftime') else str(latest_quarter.name)

        # Determine profitability
        profitable = net_income > PROFITABILITY_THRESHOLD

        return {
            'ticker': ticker,
            'name': company_name,
            'net_income': float(net_income),
            'profitable': profitable,
            'quarter_end': quarter_end
        }

    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None