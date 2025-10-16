# WIG30 ticker symbols as listed on Warsaw Stock Exchange
WIG30_TICKERS = [
    'PKN.WA', 'PKO.WA', 'SPL.WA', 'PEO.WA', 'PZU.WA', 'DNP.WA', 'MBK.WA',
    'KGH.WA', 'ALE.WA', 'LPP.WA', 'CDR.WA', 'PGE.WA', 'ZAB.WA', 'MIL.WA',
    'ACP.WA', 'PCO.WA', 'TPE.WA', 'ALR.WA', 'BDX.WA', 'CCC.WA', 'OPL.WA',
    'CPS.WA', 'KTY.WA', 'KRU.WA', 'XTB.WA', 'JSW.WA', 'SNT.WA', 'RBW.WA',
    'TXT.WA', '11B.WA'
]

# WIG20 ticker symbols - blue-chip index of the 20 largest and most liquid companies
WIG20_TICKERS = [
    'PKN.WA', 'PKO.WA', 'PEO.WA', 'PZU.WA', 'KGH.WA', 'DNP.WA', 'CDR.WA',
    'ALE.WA', 'LPP.WA', 'PGE.WA', 'SPL.WA', 'MBK.WA', 'ALR.WA', 'KRU.WA',
    'JSW.WA', 'OPL.WA', 'CPS.WA', 'TPE.WA', 'KTY.WA', 'BDX.WA'
]

# Index selection configuration
ACTIVE_INDEX = 'WIG30'  # Set to 'WIG30' or 'WIG20' to choose which index to analyze

# Output configuration
OUTPUT_CSV_FILE = 'wig30_analysis_pe_threshold.csv'
WIG20_OUTPUT_CSV_FILE = 'wig20_analysis_pe_threshold.csv'
LEGACY_CSV_FILE = 'wig30_analysis.csv'  # For backward compatibility

# Display configuration
TOP_N_DISPLAY = 10  # Number of top companies to display in console output

# ROE (Return on Equity) filtering - measures profitability efficiency
ROE_THRESHOLD = 10.0  # Minimum ROE percentage for filtering - companies below this threshold will be excluded
# Calculated as: (Net Income / Shareholders' Equity) × 100%

# P/E ratio filtering - measures stock valuation
PE_THRESHOLD = 15.0  # Maximum P/E ratio for value investing filter (companies with P/E above this are considered overvalued)
MIN_PE_THRESHOLD = 0.0  # Minimum P/E threshold to exclude companies with negative or zero P/E
ENABLE_PE_FILTER = True  # Boolean flag to enable/disable P/E filtering

# Dual filtering strategy - combines profitability and valuation
ENABLE_DUAL_FILTER = True  # Enable simultaneous ROE and P/E filtering
# When enabled, only companies meeting BOTH criteria (ROE ≥ ROE_THRESHOLD AND P/E ≤ PE_THRESHOLD) will be displayed

# Price-to-Earnings (P/E) ratio measures stock valuation by comparing current stock price to earnings per share (EPS)
# Formula: P/E = Current Price / Trailing EPS (TTM)
# Lower P/E suggests undervaluation. Typical value investing threshold is P/E ≤ 15.
# Combining ROE ≥ 10% (profitability) with P/E ≤ 15 (valuation) identifies fundamentally strong and undervalued companies.

# Decodo API Configuration
DECODO_API_URL = 'https://scraper-api.decodo.com/v2/scrape'
DECODO_API_AUTH = 'VTAwMDAzMTQ4NTg6UF9fMWYwMzY1OThiMDVjMTNkZTA0YjJkYzkyOThlNDBiZjBm'
DECODO_TIMEOUT = 30  # Request timeout in seconds
USE_DECODO_API = False  # Toggle between yfinance (False) and Decodo (True)

# Financial data sources to scrape
FINANCIAL_SOURCES = {
    'yahoo_finance': 'https://finance.yahoo.com',
    'bankier_pl': 'https://www.bankier.pl',
    'money_pl': 'https://www.money.pl'
}

# Profitability threshold (net income in local currency)
PROFITABILITY_THRESHOLD = 0