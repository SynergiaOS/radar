"""
Configuration Module for WIG30/WIG20 Trading System

This module contains all configuration settings for the advanced trading system,
including risk management parameters, market regime detection thresholds,
and API credentials.

Author: Advanced Trading System Team
License: MIT
"""

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
OUTPUT_CSV_FILE = 'wig30_roe_sorted.csv'  # Primary output - ROE-sorted analysis
WIG20_OUTPUT_CSV_FILE = 'wig20_roe_sorted.csv'  # WIG20 version - ROE-sorted
LEGACY_CSV_FILE = 'wig30_analysis.csv'  # For backward compatibility
DEPRECATED_PE_CSV_FILE = 'wig30_analysis_pe_threshold.csv'  # Deprecated filename for compatibility

# Display configuration
TOP_N_DISPLAY = 10  # Number of top companies to display in console output

# ROE (Return on Equity) filtering - measures profitability efficiency
ROE_THRESHOLD = 8.0  # Minimum ROE percentage for filtering - companies below this threshold will be excluded (more realistic threshold)
# Calculated as: (Net Income / Shareholders' Equity) × 100%

# P/E ratio filtering - measures stock valuation
PE_THRESHOLD = 20.0  # Maximum P/E ratio for value investing filter (companies with P/E above this are considered overvalued - more flexible)
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

# Risk Management Configuration
MAX_POSITION_SIZE_PCT = 0.25  # Maximum position size as % of capital
MAX_PORTFOLIO_HEAT = 20.0  # Maximum portfolio heat in %
MIN_RISK_REWARD_RATIO = 1.5  # Minimum risk/reward ratio

# xtB API Configuration (for future integration)
XTB_CLIENT_ID = "demo_user"
XTB_CLIENT_SECRET = "demo_password"
XTB_DEMO_MODE = True
XTB_WEBSOCKET_URL_DEMO = "wss://ws.xtb.com/demo"
XTB_WEBSOCKET_URL_LIVE = "wss://ws.xtb.com/live"
ENABLE_XTB_INTEGRATION = False
XTB_RETRY_ATTEMPTS = 3
XTB_TIMEOUT_SECONDS = 30

# Enhanced Data Integration Configuration
POLISH_STOCK_TICKERS = WIG30_TICKERS  # Default to WIG30 tickers
RISK_PER_TRADE_PCT = 0.02  # Default 2% risk per trade

# Market Regime Detection Configuration
ADX_THRESHOLD_WEAK = 20.0
ADX_THRESHOLD_STRONG = 40.0
ADX_THRESHOLD_VERY_STRONG = 60.0
ENABLE_REGIME_FILTER = True

# Data Sources Configuration
GPW_TICKERS = WIG30_TICKERS  # Polish Stock Exchange tickers alias

# Backtesting Configuration
BACKTEST_INITIAL_CAPITAL = 100000.0
BACKTEST_COMMISSION_RATE = 0.001
BACKTEST_SLIPPAGE_PCT = 0.0005
BACKTEST_START_DATE = "2022-01-01"

# Data Cache Configuration
DATA_CACHE_DAYS = 30
DATABASE_PATH = "realtime_data.db"

# Kelly Criterion Configuration
KELLY_FRACTION = 0.25  # Conservative Kelly fraction
TRAILING_STOP_ATR_MULTIPLIER = 2.0

# P/B Valuation Configuration
PB_THRESHOLD = 3.0  # Increased threshold for more flexibility
ENABLE_PB_FILTER = False  # Disable strict P/B filtering
ENABLE_TRIPLE_FILTER = False  # Disable triple filter to show more stocks

# Technical Analysis Configuration
MA_PERIODS = [5, 10, 20]
MA_HISTORY_DAYS = 30
TREND_DETECTION_ENABLED = True
TREND_LABELS = {
    'upward': 'Wzrostowy',
    'downward': 'Spadkowy',
    'sideways': 'Boczny'
}

# CQG API Configuration
USE_CQG_API = False
CQG_API_URL = ""
CQG_API_KEY = ""
CQG_API_SECRET = ""
CQG_TIMEOUT = 30

# Correlation Analysis Configuration
ENABLE_CORRELATION_ANALYSIS = True
CORRELATION_METRICS = ['roe', 'pb_ratio']

# Chart Generation Configuration
ENABLE_CHART_GENERATION = True
CHART_OUTPUT_DIR = 'charts'
CHART_MAX_STOCKS = 5
CHART_COLORS = {
    'price': '#1E3A8A',
    'ma5': '#22C55E',
    'ma10': '#F97316',
    'ma20': '#EF4444'
}

# Updated Output Filenames
OUTPUT_CSV_FILE = 'wig30_analysis_cqg_ma_viz.csv'
WIG20_OUTPUT_CSV_FILE = 'wig20_analysis_cqg_ma_viz.csv'