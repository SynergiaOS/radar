#!/usr/bin/env python3
"""
Stooq.pl Integration Module
Historical data integration for Polish market data from Stooq.pl
Provides comprehensive historical data fetching and caching
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import os
from config import GPW_TICKERS, DATA_CACHE_DAYS, DATABASE_PATH
from trading_chart_service import chart_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StooqData:
    """Historical data from Stooq.pl"""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class StooqDividend:
    """Dividend data from Stooq.pl"""
    symbol: str
    ex_date: datetime
    payment_date: datetime
    amount: float
    currency: str

@dataclass
class StooqSplit:
    """Stock split data from Stooq.pl"""
    symbol: str
    ex_date: datetime
    ratio: float
    description: str

class StooqClient:
    """Stooq.pl API client for historical data"""

    def __init__(self):
        self.base_url = "https://stooq.pl/q"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache_dir = "stooq_cache"
        self.db_path = os.path.join(DATABASE_PATH, "stooq_data.db")
        self._ensure_cache_dir()
        self._init_database()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for caching"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS dividends (
                    symbol TEXT,
                    ex_date DATE,
                    payment_date DATE,
                    amount REAL,
                    currency TEXT,
                    PRIMARY KEY (symbol, ex_date)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS splits (
                    symbol TEXT,
                    ex_date DATE,
                    ratio REAL,
                    description TEXT,
                    PRIMARY KEY (symbol, ex_date)
                )
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_historical_symbol_date
                ON historical_data(symbol, date)
            ''')

    def _get_stooq_symbol(self, ticker: str) -> str:
        """Convert ticker to Stooq.pl format"""
        # Polish stocks on GPW
        if ticker.endswith('.WA'):
            return ticker
        elif '.' not in ticker:
            return f"{ticker}.WA"
        return ticker

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from Stooq.pl"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                return datetime.strptime(date_str, '%Y%m%d')

    def _clean_price(self, price_str: str) -> float:
        """Clean and parse price string"""
        if pd.isna(price_str) or price_str == '' or price_str == '-':
            return 0.0
        try:
            # Remove commas and convert to float
            price_str = str(price_str).replace(',', '').replace(' ', '')
            return float(price_str)
        except (ValueError, TypeError):
            return 0.0

    def _clean_volume(self, volume_str: str) -> int:
        """Clean and parse volume string"""
        if pd.isna(volume_str) or volume_str == '' or volume_str == '-':
            return 0
        try:
            volume_str = str(volume_str).replace(',', '').replace(' ', '').replace('K', '000').replace('M', '000000')
            return int(float(volume_str))
        except (ValueError, TypeError):
            return 0

    async def get_historical_data(self, ticker: str,
                                start_date: datetime = None,
                                end_date: datetime = None,
                                use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical data for a ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            symbol = self._get_stooq_symbol(ticker)

            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)

            # Check cache first
            if use_cache:
                cached_data = self._get_cached_data(symbol, start_date, end_date)
                if not cached_data.empty:
                    # Check if we have recent data
                    latest_date = cached_data['date'].max()
                    if (end_date - latest_date).days <= 1:
                        logger.info(f"Using cached data for {ticker}")
                        return cached_data
                    else:
                        # Need to fetch missing data
                        start_date = latest_date + timedelta(days=1)

            # Fetch from Stooq.pl
            fresh_data = await self._fetch_from_stooq(symbol, start_date, end_date)

            if not fresh_data.empty:
                # Cache the fresh data
                self._cache_data(fresh_data)

                # Combine with cached data if available
                if use_cache and 'cached_data' in locals():
                    all_data = pd.concat([cached_data, fresh_data])
                    all_data = all_data.drop_duplicates(subset=['symbol', 'date']).sort_values('date')
                    return all_data
                else:
                    return fresh_data

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    async def _fetch_from_stooq(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data directly from Stooq.pl"""
        try:
            # Stooq.pl historical data URL
            url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV data
            data_lines = response.text.strip().split('\n')
            if len(data_lines) < 2:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Skip header and parse data
            records = []
            for line in data_lines[1:]:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            date = self._parse_date(parts[0])

                            # Skip if outside our date range
                            if date < start_date or date > end_date:
                                continue

                            record = {
                                'symbol': symbol,
                                'date': date,
                                'open': self._clean_price(parts[1]),
                                'high': self._clean_price(parts[2]),
                                'low': self._clean_price(parts[3]),
                                'close': self._clean_price(parts[4]),
                                'volume': self._clean_volume(parts[5])
                            }
                            records.append(record)

                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing line for {symbol}: {line}, error: {e}")
                            continue

            if records:
                df = pd.DataFrame(records)
                logger.info(f"Fetched {len(df)} records for {symbol} from {start_date.date()} to {end_date.date()}")
                return df
            else:
                logger.warning(f"No valid data found for {symbol} in specified date range")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error parsing data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_cached_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get cached data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT symbol, date, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                '''
                df = pd.read_sql_query(query, conn, params=(symbol, start_date.date(), end_date.date()))
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            logger.error(f"Error reading cached data: {e}")
            return pd.DataFrame()

    def _cache_data(self, df: pd.DataFrame):
        """Cache data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql('historical_data', conn, if_exists='append', index=False, method='multi')
        except Exception as e:
            logger.error(f"Error caching data: {e}")

    async def get_dividends(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> List[StooqDividend]:
        """Get dividend data for a ticker"""
        try:
            symbol = self._get_stooq_symbol(ticker)

            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365*2)  # 2 years of dividends

            # Check cache first
            cached_dividends = self._get_cached_dividends(symbol, start_date, end_date)
            if cached_dividends:
                return cached_dividends

            # Stooq.pl dividend URL
            url = f"https://stooq.pl/q/d/s/?s={symbol}"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse dividend data
            dividends = []
            data_lines = response.text.strip().split('\n')

            for line in data_lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            dividend = StooqDividend(
                                symbol=symbol,
                                ex_date=self._parse_date(parts[0]),
                                payment_date=self._parse_date(parts[1]),
                                amount=self._clean_price(parts[2]),
                                currency=parts[3].strip()
                            )

                            if start_date <= dividend.ex_date <= end_date:
                                dividends.append(dividend)

                        except (ValueError, IndexError):
                            continue

            # Cache dividends
            if dividends:
                self._cache_dividends(dividends)

            return dividends

        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker}: {e}")
            return []

    async def get_splits(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> List[StooqSplit]:
        """Get stock split data for a ticker"""
        try:
            symbol = self._get_stooq_symbol(ticker)

            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365*5)  # 5 years of splits

            # Check cache first
            cached_splits = self._get_cached_splits(symbol, start_date, end_date)
            if cached_splits:
                return cached_splits

            # For now, return empty list as Stooq.pl split data format needs investigation
            # This is a placeholder for future implementation
            return []

        except Exception as e:
            logger.error(f"Error fetching splits for {ticker}: {e}")
            return []

    def _get_cached_dividends(self, symbol: str, start_date: datetime, end_date: datetime) -> List[StooqDividend]:
        """Get cached dividend data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT symbol, ex_date, payment_date, amount, currency
                    FROM dividends
                    WHERE symbol = ? AND ex_date BETWEEN ? AND ?
                    ORDER BY ex_date
                '''
                cursor = conn.execute(query, (symbol, start_date.date(), end_date.date()))
                dividends = []
                for row in cursor.fetchall():
                    dividend = StooqDividend(
                        symbol=row[0],
                        ex_date=datetime.strptime(row[1], '%Y-%m-%d'),
                        payment_date=datetime.strptime(row[2], '%Y-%m-%d'),
                        amount=row[3],
                        currency=row[4]
                    )
                    dividends.append(dividend)
                return dividends
        except Exception as e:
            logger.error(f"Error reading cached dividends: {e}")
            return []

    def _cache_dividends(self, dividends: List[StooqDividend]):
        """Cache dividend data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for dividend in dividends:
                    conn.execute('''
                        INSERT OR REPLACE INTO dividends
                        (symbol, ex_date, payment_date, amount, currency)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        dividend.symbol,
                        dividend.ex_date.date(),
                        dividend.payment_date.date(),
                        dividend.amount,
                        dividend.currency
                    ))
        except Exception as e:
            logger.error(f"Error caching dividends: {e}")

    def _get_cached_splits(self, symbol: str, start_date: datetime, end_date: datetime) -> List[StooqSplit]:
        """Get cached split data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT symbol, ex_date, ratio, description
                    FROM splits
                    WHERE symbol = ? AND ex_date BETWEEN ? AND ?
                    ORDER BY ex_date
                '''
                cursor = conn.execute(query, (symbol, start_date.date(), end_date.date()))
                splits = []
                for row in cursor.fetchall():
                    split = StooqSplit(
                        symbol=row[0],
                        ex_date=datetime.strptime(row[1], '%Y-%m-%d'),
                        ratio=row[2],
                        description=row[3]
                    )
                    splits.append(split)
                return splits
        except Exception as e:
            logger.error(f"Error reading cached splits: {e}")
            return []

    async def update_multiple_symbols(self, tickers: List[str], days_back: int = DATA_CACHE_DAYS) -> Dict[str, pd.DataFrame]:
        """Update historical data for multiple symbols"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for ticker in tickers:
            try:
                logger.info(f"Updating data for {ticker}")
                data = await self.get_historical_data(ticker, start_date, end_date)
                if not data.empty:
                    results[ticker] = data
                    logger.info(f"Updated {len(data)} records for {ticker}")
                else:
                    logger.warning(f"No data found for {ticker}")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error updating {ticker}: {e}")

        logger.info(f"Updated historical data for {len(results)} symbols")
        return results

    async def get_market_data_summary(self, tickers: List[str] = None) -> Dict[str, Any]:
        """Get summary of market data for multiple symbols"""
        try:
            if tickers is None:
                tickers = GPW_TICKERS

            summary = {
                'timestamp': datetime.now(),
                'symbols': {},
                'market_stats': {
                    'total_symbols': 0,
                    'data_available': 0,
                    'last_updated': None
                }
            }

            for ticker in tickers:
                try:
                    symbol = self._get_stooq_symbol(ticker)

                    # Get latest data from cache
                    with sqlite3.connect(self.db_path) as conn:
                        query = '''
                            SELECT date, close, volume
                            FROM historical_data
                            WHERE symbol = ?
                            ORDER BY date DESC
                            LIMIT 1
                        '''
                        cursor = conn.execute(query, (symbol,))
                        row = cursor.fetchone()

                        if row:
                            summary['symbols'][ticker] = {
                                'symbol': symbol,
                                'last_date': datetime.strptime(row[0], '%Y-%m-%d'),
                                'last_price': row[1],
                                'last_volume': row[2],
                                'data_available': True
                            }
                            summary['market_stats']['data_available'] += 1
                        else:
                            summary['symbols'][ticker] = {
                                'symbol': symbol,
                                'data_available': False
                            }

                        summary['market_stats']['total_symbols'] += 1

                except Exception as e:
                    logger.error(f"Error getting summary for {ticker}: {e}")
                    summary['symbols'][ticker] = {'data_available': False, 'error': str(e)}

            return summary

        except Exception as e:
            logger.error(f"Error getting market data summary: {e}")
            return {}

    def clear_cache(self, older_than_days: int = None):
        """Clear cached data"""
        try:
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM historical_data WHERE date < ?', (cutoff_date.date(),))
                    conn.execute('DELETE FROM dividends WHERE ex_date < ?', (cutoff_date.date(),))
                    conn.execute('DELETE FROM splits WHERE ex_date < ?', (cutoff_date.date(),))
                logger.info(f"Cleared cache older than {older_than_days} days")
            else:
                # Clear entire cache
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM historical_data')
                    conn.execute('DELETE FROM dividends')
                    conn.execute('DELETE FROM splits')
                logger.info("Cleared entire cache")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class StooqDataManager:
    """Manager for Stooq.pl historical data operations"""

    def __init__(self):
        self.client = StooqClient()
        self.gpw_symbols = GPW_TICKERS

    async def initialize(self) -> bool:
        """Initialize Stooq data manager"""
        try:
            # Test connection with a sample symbol
            test_data = await self.client.get_historical_data('PEO.WA',
                                                           start_date=datetime.now() - timedelta(days=7))
            if not test_data.empty:
                logger.info("Stooq.pl data manager initialized successfully")
                return True
            else:
                logger.error("Failed to connect to Stooq.pl")
                return False
        except Exception as e:
            logger.error(f"Error initializing Stooq data manager: {e}")
            return False

    async def update_historical_data(self, tickers: List[str] = None, days_back: int = DATA_CACHE_DAYS):
        """Update historical data for specified tickers"""
        if tickers is None:
            tickers = self.gpw_symbols

        logger.info(f"Updating historical data for {len(tickers)} symbols")
        return await self.client.update_multiple_symbols(tickers, days_back)

    async def get_technical_data(self, ticker: str,
                               start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """Get historical data with technical indicators"""
        try:
            # Get OHLCV data
            df = await self.client.get_historical_data(ticker, start_date, end_date)

            if df.empty:
                return df

            # Calculate technical indicators
            df = self._calculate_indicators(df)
            return df

        except Exception as e:
            logger.error(f"Error getting technical data for {ticker}: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()

            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview with latest data"""
        return await self.client.get_market_data_summary(self.gpw_symbols)


# Global Stooq data manager instance
stooq_manager = StooqDataManager()


async def initialize_stooq() -> bool:
    """Initialize Stooq integration"""
    return await stooq_manager.initialize()


async def update_historical_database():
    """Update historical database with latest data"""
    return await stooq_manager.update_historical_data()


async def get_technical_analysis_data(ticker: str,
                                    start_date: datetime = None,
                                    end_date: datetime = None) -> pd.DataFrame:
    """Get technical analysis data for a ticker"""
    return await stooq_manager.get_technical_data(ticker, start_date, end_date)


if __name__ == "__main__":
    async def main():
        """Example usage"""
        # Initialize Stooq
        if await initialize_stooq():
            print("Stooq initialized successfully")

            # Get historical data for a symbol
            start_date = datetime.now() - timedelta(days=30)
            data = await get_technical_analysis_data('PEO.WA', start_date)

            if not data.empty:
                print(f"Retrieved {len(data)} records for PEO.WA")
                print(f"Date range: {data['date'].min()} to {data['date'].max()}")
                print(f"Latest price: {data['close'].iloc[-1]:.2f}")
                print(f"Latest RSI: {data['rsi'].iloc[-1]:.2f}")
            else:
                print("No data found for PEO.WA")

            # Update multiple symbols
            tickers = ['PEO.WA', 'PKO.WA', 'KGH.WA']
            results = await stooq_manager.update_historical_data(tickers, days_back=90)
            print(f"Updated data for {len(results)} symbols")

        else:
            print("Failed to initialize Stooq")

    import asyncio
    asyncio.run(main())