#!/usr/bin/env python3
"""
Real-Time Data Integration System
Connects to multiple real-time data sources for live trading analysis
"""

import requests
import json
import time
import websocket
import threading
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import os
import asyncio
import aiohttp

class RealTimeDataIntegration:
    """Advanced real-time data integration system."""

    def __init__(self):
        self.connections = {}
        self.data_callbacks = []
        self.price_cache = {}
        self.last_update = {}
        self.database_path = 'realtime_data.db'
        self.running = False

        # Initialize database
        self.init_database()

        # Configuration
        self.config = {
            'alpha_vantage': {
                'enabled': False,
                'api_key': '',
                'symbols': ['TXT.WA', 'XTB.WA', 'PKN.WA'],
                'interval': '1min'
            },
            'finnhub': {
                'enabled': False,
                'api_key': '',
                'symbols': ['TXT.WA', 'XTB.WA', 'PKN.WA']
            },
            'websocket_feeds': {
                'enabled': False,
                'polygon': {'api_key': '', 'tickers': ['TXT:WA', 'XTB:PL']},
                'finnhub': {'api_key': '', 'symbols': ['TXT.WA', 'XTB.WA']}
            },
            'local_feeds': {
                'enabled': True,
                'file_paths': ['data/current_prices.csv', 'data/live_prices.json'],
                'update_interval': 5
            }
        }

    def init_database(self):
        """Initialize SQLite database for storing real-time data."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                volume_ratio REAL,
                volatility REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_data(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals_log(ticker)')

        conn.commit()
        conn.close()

    def load_config(self, config_file='realtime_config.json'):
        """Load configuration from file."""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                self.config.update(user_config)
                print(f"✅ Configuration loaded from {config_file}")
            except Exception as e:
                print(f"❌ Error loading config: {str(e)}")
        else:
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"📝 Created default config: {config_file}")

    def add_data_callback(self, callback: Callable):
        """Add callback function for new data."""
        self.data_callbacks.append(callback)

    def store_price_data(self, ticker: str, price: float, volume: int = 0, source: str = 'unknown'):
        """Store price data in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO price_data (ticker, price, volume, source)
                VALUES (?, ?, ?, ?)
            ''', (ticker, price, volume, source))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ Error storing price data: {str(e)}")

    def store_signal_data(self, ticker: str, action: str, confidence: float, score: float, source: str):
        """Store signal data in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO signals_log (ticker, action, confidence, score, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, action, confidence, score, source))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"❌ Error storing signal data: {str(e)}")

    def get_latest_prices(self, tickers: List[str] = None) -> Dict:
        """Get latest prices from database."""
        try:
            conn = sqlite3.connect(self.database_path)

            if tickers:
                placeholders = ','.join(['?' for _ in tickers])
                query = f'''
                    SELECT ticker, price, timestamp FROM price_data
                    WHERE ticker IN ({placeholders})
                    AND timestamp = (
                        SELECT MAX(timestamp) FROM price_data p2
                        WHERE p2.ticker = price_data.ticker
                    )
                '''
                df = pd.read_sql_query(query, conn, params=tickers)
            else:
                query = '''
                    SELECT ticker, price, timestamp FROM price_data p1
                    WHERE timestamp = (
                        SELECT MAX(timestamp) FROM price_data p2
                        WHERE p2.ticker = p1.ticker
                    )
                '''
                df = pd.read_sql_query(query, conn)

            conn.close()

            result = {}
            for _, row in df.iterrows():
                result[row['ticker']] = {
                    'price': row['price'],
                    'timestamp': row['timestamp']
                }

            return result

        except Exception as e:
            print(f"❌ Error getting latest prices: {str(e)}")
            return {}

    async def fetch_alpha_vantage_data(self):
        """Fetch real-time data from Alpha Vantage."""
        if not self.config['alpha_vantage']['enabled']:
            return

        api_key = self.config['alpha_vantage']['api_key']
        symbols = self.config['alpha_vantage']['symbols']

        if not api_key:
            print("⚠️  Alpha Vantage API key not configured")
            return

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Convert Polish format to Alpha Vantage format
                    av_symbol = symbol.replace('.WA', '.WA')
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_symbol}&apikey={api_key}"

                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'Global Quote' in data:
                                quote = data['Global Quote']
                                price = float(quote['05. price'])
                                volume = int(quote['06. volume'].replace(',', ''))

                                # Store and notify
                                self.store_price_data(symbol, price, volume, 'alpha_vantage')
                                self.price_cache[symbol] = {
                                    'price': price,
                                    'volume': volume,
                                    'timestamp': datetime.now(),
                                    'source': 'alpha_vantage'
                                }

                                # Notify callbacks
                                for callback in self.data_callbacks:
                                    callback(symbol, price, 'alpha_vantage')

                                print(f"✅ Alpha Vantage: {symbol} @ {price} PLN")

                except Exception as e:
                    print(f"❌ Alpha Vantage error for {symbol}: {str(e)}")

                # Rate limiting
                await asyncio.sleep(12)  # Alpha Vantage free tier: 5 calls per minute

    async def fetch_finnhub_data(self):
        """Fetch real-time data from Finnhub."""
        if not self.config['finnhub']['enabled']:
            return

        api_key = self.config['finnhub']['api_key']
        symbols = self.config['finnhub']['symbols']

        if not api_key:
            print("⚠️  Finnhub API key not configured")
            return

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Convert to Finnhub format
                    fh_symbol = symbol.replace('.WA', '.WA')
                    url = f"https://finnhub.io/api/v1/quote?symbol={fh_symbol}&token={api_key}"

                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'c' in data and data['c'] > 0:
                                price = data['c']
                                volume = data.get('vol', 0)

                                self.store_price_data(symbol, price, volume, 'finnhub')
                                self.price_cache[symbol] = {
                                    'price': price,
                                    'volume': volume,
                                    'timestamp': datetime.now(),
                                    'source': 'finnhub'
                                }

                                for callback in self.data_callbacks:
                                    callback(symbol, price, 'finnhub')

                                print(f"✅ Finnhub: {symbol} @ {price} PLN")

                except Exception as e:
                    print(f"❌ Finnhub error for {symbol}: {str(e)}")

    def monitor_local_files(self):
        """Monitor local files for price updates."""
        if not self.config['local_feeds']['enabled']:
            return

        print("📁 Starting local file monitoring...")

        while self.running:
            try:
                # Monitor CSV file
                csv_path = 'data/current_prices.csv'
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    current_time = datetime.now()

                    for _, row in df.iterrows():
                        ticker = row['ticker']
                        price = float(row['price'])
                        volume = int(row.get('volume', 0))

                        # Check if this is new data
                        last_update = self.last_update.get(ticker, datetime.min)
                        if (current_time - last_update).seconds >= self.config['local_feeds']['update_interval']:
                            self.store_price_data(ticker, price, volume, 'local_file')
                            self.price_cache[ticker] = {
                                'price': price,
                                'volume': volume,
                                'timestamp': current_time,
                                'source': 'local_file'
                            }

                            for callback in self.data_callbacks:
                                callback(ticker, price, 'local_file')

                            self.last_update[ticker] = current_time

                # Monitor JSON file
                json_path = 'data/live_prices.json'
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    for ticker, info in data.items():
                        if 'price' in info:
                            price = float(info['price'])
                            volume = int(info.get('volume', 0))

                            last_update = self.last_update.get(ticker, datetime.min)
                            current_time = datetime.now()

                            if (current_time - last_update).seconds >= self.config['local_feeds']['update_interval']:
                                self.store_price_data(ticker, price, volume, 'local_json')
                                self.price_cache[ticker] = {
                                    'price': price,
                                    'volume': volume,
                                    'timestamp': current_time,
                                    'source': 'local_json'
                                }

                                for callback in self.data_callbacks:
                                    callback(ticker, price, 'local_json')

                                self.last_update[ticker] = current_time

            except Exception as e:
                print(f"❌ Local file monitoring error: {str(e)}")

            # Wait before next check
            time.sleep(self.config['local_feeds']['update_interval'])

    def simulate_real_time_data(self):
        """Simulate real-time data for testing."""
        print("🎲 Starting real-time data simulation...")

        base_prices = {
            'TXT.WA': 51.47,
            'XTB.WA': 68.06,
            'PKN.WA': 89.42,
            'PKO.WA': 73.96,
            'PZU.WA': 56.19
        }

        while self.running:
            try:
                current_time = datetime.now()

                for ticker, base_price in base_prices.items():
                    # Generate realistic price movement
                    change_pct = np.random.normal(0, 0.005)  # 0.5% std dev
                    new_price = base_price * (1 + change_pct)
                    volume = int(np.random.normal(1000000, 200000))

                    # Store and cache
                    self.store_price_data(ticker, new_price, volume, 'simulation')
                    self.price_cache[ticker] = {
                        'price': new_price,
                        'volume': volume,
                        'timestamp': current_time,
                        'source': 'simulation'
                    }

                    # Notify callbacks
                    for callback in self.data_callbacks:
                        callback(ticker, new_price, 'simulation')

                    print(f"🎲 {ticker}: {new_price:.2f} PLN ({change_pct:+.2%})")

                # Wait before next update
                time.sleep(5)

            except Exception as e:
                print(f"❌ Simulation error: {str(e)}")
                time.sleep(5)

    def start_websocket_connection(self):
        """Start WebSocket connection for real-time feeds."""
        # WebSocket implementation would go here
        # This is a placeholder for WebSocket functionality
        pass

    async def start_real_time_collection(self):
        """Start all real-time data collection methods."""
        print("🚀 Starting real-time data collection...")
        self.running = True

        # Start file monitoring in background thread
        file_thread = threading.Thread(target=self.monitor_local_files, daemon=True)
        file_thread.start()

        # Start simulation in background thread
        sim_thread = threading.Thread(target=self.simulate_real_time_data, daemon=True)
        sim_thread.start()

        # Start API data collection
        tasks = []
        if self.config['alpha_vantage']['enabled']:
            tasks.append(self.fetch_alpha_vantage_data())

        if self.config['finnhub']['enabled']:
            tasks.append(self.fetch_finnhub_data())

        # Run API tasks
        if tasks:
            await asyncio.gather(*tasks)

    def stop_real_time_collection(self):
        """Stop real-time data collection."""
        print("🛑 Stopping real-time data collection...")
        self.running = False

    def get_price_history(self, ticker: str, hours: int = 24) -> pd.DataFrame:
        """Get price history for a ticker."""
        try:
            conn = sqlite3.connect(self.database_path)

            query = '''
                SELECT price, volume, timestamp FROM price_data
                WHERE ticker = ? AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours)

            df = pd.read_sql_query(query, conn, params=[ticker])
            conn.close()

            return df

        except Exception as e:
            print(f"❌ Error getting price history: {str(e)}")
            return pd.DataFrame()

    def export_data(self, filename: str = 'realtime_export.csv'):
        """Export real-time data to CSV."""
        try:
            conn = sqlite3.connect(self.database_path)

            # Get all price data
            query = '''
                SELECT ticker, price, volume, timestamp, source
                FROM price_data
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            '''

            df = pd.read_sql_query(query, conn)
            conn.close()

            df.to_csv(filename, index=False)
            print(f"✅ Data exported to {filename}")

        except Exception as e:
            print(f"❌ Error exporting data: {str(e)}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of collected data."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Total records
            cursor.execute('SELECT COUNT(*) FROM price_data WHERE timestamp >= datetime("now", "-1 hour")')
            total_records = cursor.fetchone()[0]

            # Unique tickers
            cursor.execute('SELECT COUNT(DISTINCT ticker) FROM price_data WHERE timestamp >= datetime("now", "-1 hour")')
            unique_tickers = cursor.fetchone()[0]

            # Latest update
            cursor.execute('SELECT MAX(timestamp) FROM price_data')
            latest_update = cursor.fetchone()[0]

            conn.close()

            return {
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'latest_update': latest_update,
                'cached_prices': len(self.price_cache),
                'monitoring_active': self.running
            }

        except Exception as e:
            print(f"❌ Error getting summary stats: {str(e)}")
            return {}


def main():
    """Main function to test real-time integration."""
    print("📡 REAL-TIME DATA INTEGRATION SYSTEM")
    print("=" * 50)

    system = RealTimeDataIntegration()
    system.load_config()

    # Add sample callback
    def price_callback(ticker, price, source):
        print(f"📊 Callback: {ticker} @ {price:.2f} PLN from {source}")

    system.add_data_callback(price_callback)

    try:
        # Start real-time collection
        print("🚀 Starting real-time data collection...")
        asyncio.run(system.start_real_time_collection())

        # Run for a limited time for demo
        time.sleep(30)

        # Show stats
        stats = system.get_summary_stats()
        print(f"\n📊 Summary Stats:")
        print(f"   Total records: {stats['total_records']}")
        print(f"   Unique tickers: {stats['unique_tickers']}")
        print(f"   Cached prices: {stats['cached_prices']}")
        print(f"   Latest update: {stats['latest_update']}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping real-time collection...")
        system.stop_real_time_collection()

    # Export data
    system.export_data()

    print("🏁 Real-time integration test complete!")


if __name__ == "__main__":
    main()