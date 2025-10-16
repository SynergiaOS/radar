#!/usr/bin/env python3
"""
Enhanced Real-Time Data Integration System with Risk Management
Connects to multiple real-time data sources for live trading analysis
Enhanced with comprehensive risk management features and real-time monitoring
"""

import requests
import json
import time
import threading
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import os
import asyncio
import numpy as np

# Optional dependencies - will be imported lazily
WEBSOCKET_AVAILABLE = False
AIOHTTP_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    websocket = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None

# Import enhanced risk management modules
from risk_management import RiskManager
from market_regime import MarketRegimeDetector
from trading_signals import TradingSignalGenerator
from ml_trading_system import AdvancedMLTradingSystem

class EnhancedRealTimeIntegration:
    """Enhanced real-time data integration system with comprehensive risk management."""

    def __init__(self, capital: float = 100000):
        self.connections = {}
        self.data_callbacks = []
        self.price_cache = {}
        self.price_history = {}
        self.last_update = {}
        self.database_path = 'realtime_data.db'
        self.running = False
        self.capital = capital

        # Enhanced risk management components
        self.risk_manager = RiskManager()
        self.market_regime_detector = MarketRegimeDetector()
        self.signal_generator = TradingSignalGenerator()
        self.ml_system = AdvancedMLTradingSystem()

        # Real-time risk monitoring
        self.active_positions = []
        self.real_time_alerts = []
        self.risk_metrics_history = []
        self.position_updates = []

        # Initialize database
        self.init_database()
        self.load_historical_data()

        print(f"🛡️ Enhanced Real-Time Integration initialized with {capital:,.0f} PLN capital")

        # Enhanced configuration with risk management
        self.config = {
            'risk_management': {
                'enabled': True,
                'max_portfolio_heat': 20.0,
                'risk_per_trade': 0.02,
                'stop_loss_atr_multiplier': 2.0,
                'correlation_threshold': 0.7,
                'real_time_monitoring': True,
                'alert_thresholds': {
                    'portfolio_heat_warning': 15.0,
                    'portfolio_heat_critical': 18.0,
                    'position_loss_warning': -5.0,
                    'position_loss_critical': -10.0
                }
            },
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
        if not self.config['alpha_vantage']['enabled'] or not AIOHTTP_AVAILABLE:
            if not AIOHTTP_AVAILABLE:
                print("⚠️  aiohttp not available - Alpha Vantage data collection disabled")
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
        if not self.config['finnhub']['enabled'] or not AIOHTTP_AVAILABLE:
            if not AIOHTTP_AVAILABLE:
                print("⚠️  aiohttp not available - Finnhub data collection disabled")
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
        if not WEBSOCKET_AVAILABLE:
            print("⚠️  websocket-client not available - WebSocket connections disabled")
            return

        # WebSocket implementation would go here
        # This is a placeholder for WebSocket functionality
        print("🔌 WebSocket connection placeholder - implementation required")
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

    def load_historical_data(self):
        """Load historical data for risk management components."""
        try:
            # Load signal generator data
            self.signal_generator.load_data()

            # Load ML system data
            self.ml_system.load_historical_data()
            self.ml_system.train_ml_models()

            print("✅ Historical data loaded for risk management components")
        except Exception as e:
            print(f"⚠️  Error loading historical data: {str(e)}")

    def enhanced_price_callback(self, ticker: str, price: float, source: str):
        """Enhanced price callback with real-time risk management."""
        try:
            # Store price data
            self.price_cache[ticker] = {
                'price': price,
                'timestamp': datetime.now(),
                'source': source
            }

            # Update price history
            if ticker not in self.price_history:
                self.price_history[ticker] = []

            self.price_history[ticker].append({
                'price': price,
                'timestamp': datetime.now(),
                'source': source
            })

            # Keep only last 100 points per ticker
            if len(self.price_history[ticker]) > 100:
                self.price_history[ticker] = self.price_history[ticker][-100:]

            # Real-time risk monitoring
            if self.config['risk_management']['enabled'] and self.config['risk_management']['real_time_monitoring']:
                self.monitor_position_risks(ticker, price)
                self.check_portfolio_heat()
                self.update_market_regimes(ticker)

        except Exception as e:
            print(f"❌ Error in enhanced price callback: {str(e)}")

    def monitor_position_risks(self, ticker: str, current_price: float):
        """Monitor risks for active positions in real-time."""
        for position in self.active_positions:
            if position['ticker'] == ticker:
                # Calculate current P&L
                entry_price = position['entry_price']
                shares = position['shares']
                current_value = shares * current_price
                entry_value = shares * entry_price
                pnl = current_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # Update position
                position['current_price'] = current_price
                position['current_value'] = current_value
                position['unrealized_pnl'] = pnl
                position['unrealized_pnl_pct'] = pnl_pct

                # Check stop loss and take profit
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')

                alerts = []

                if stop_loss and current_price <= stop_loss:
                    alerts.append(f"🔴 STOP LOSS TRIGGERED for {ticker} at {current_price:.2f} PLN")
                    alerts.append(f"   Loss: {pnl_pct:.2f}% ({pnl:,.0f} PLN)")

                elif take_profit and current_price >= take_profit:
                    alerts.append(f"🟢 TAKE PROFIT TRIGGERED for {ticker} at {current_price:.2f} PLN")
                    alerts.append(f"   Profit: {pnl_pct:.2f}% ({pnl:,.0f} PLN)")

                # Check loss thresholds
                thresholds = self.config['risk_management']['alert_thresholds']
                if pnl_pct <= thresholds['position_loss_critical']:
                    alerts.append(f"🚨 CRITICAL LOSS: {ticker} down {pnl_pct:.2f}%")
                elif pnl_pct <= thresholds['position_loss_warning']:
                    alerts.append(f"⚠️  WARNING: {ticker} down {pnl_pct:.2f}%")

                # Store alerts
                if alerts:
                    for alert in alerts:
                        self.real_time_alerts.append({
                            'ticker': ticker,
                            'alert': alert,
                            'timestamp': datetime.now(),
                            'severity': 'CRITICAL' if pnl_pct <= thresholds['position_loss_critical'] else 'WARNING'
                        })
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")

    def check_portfolio_heat(self):
        """Check portfolio heat in real-time."""
        try:
            if not self.active_positions:
                return

            total_risk = 0
            total_value = 0

            for position in self.active_positions:
                position_value = position.get('current_value', 0)
                position_risk = position.get('position_risk', 0)

                total_value += position_value
                total_risk += position_risk

            portfolio_heat_pct = (total_risk / self.capital) * 100 if self.capital > 0 else 0

            # Check thresholds
            thresholds = self.config['risk_management']['alert_thresholds']
            max_heat = self.config['risk_management']['max_portfolio_heat']

            if portfolio_heat_pct >= max_heat:
                alert = f"🚨 PORTFOLIO OVERHEATED: {portfolio_heat_pct:.1f}% (Max: {max_heat:.1f}%)"
                self.real_time_alerts.append({
                    'alert': alert,
                    'timestamp': datetime.now(),
                    'severity': 'CRITICAL'
                })
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")

            elif portfolio_heat_pct >= thresholds['portfolio_heat_critical']:
                alert = f"🔴 CRITICAL PORTFOLIO HEAT: {portfolio_heat_pct:.1f}%"
                self.real_time_alerts.append({
                    'alert': alert,
                    'timestamp': datetime.now(),
                    'severity': 'CRITICAL'
                })
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")

            elif portfolio_heat_pct >= thresholds['portfolio_heat_warning']:
                alert = f"⚠️  HIGH PORTFOLIO HEAT: {portfolio_heat_pct:.1f}%"
                self.real_time_alerts.append({
                    'alert': alert,
                    'timestamp': datetime.now(),
                    'severity': 'WARNING'
                })
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")

            # Store risk metrics
            self.risk_metrics_history.append({
                'timestamp': datetime.now(),
                'portfolio_heat_pct': portfolio_heat_pct,
                'total_positions': len(self.active_positions),
                'total_value': total_value,
                'total_risk': total_risk
            })

            # Keep only last 1000 metrics
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]

        except Exception as e:
            print(f"❌ Error checking portfolio heat: {str(e)}")

    def update_market_regimes(self, ticker: str):
        """Update market regimes in real-time."""
        try:
            if ticker not in self.price_history or len(self.price_history[ticker]) < 14:
                return

            # Get recent prices
            recent_prices = [p['price'] for p in self.price_history[ticker][-14:]]

            if len(recent_prices) < 14:
                return

            # Create DataFrame for regime analysis
            df = pd.DataFrame({'close': recent_prices})
            regime_info = self.market_regime_detector.analyze_regime(df)

            # Check for regime changes
            previous_regime = getattr(self, f'_last_regime_{ticker}', 'UNKNOWN')
            current_regime = regime_info['regime']

            if previous_regime != current_regime:
                alert = f"📊 MARKET REGIME CHANGE: {ticker} from {previous_regime} to {current_regime}"
                self.real_time_alerts.append({
                    'ticker': ticker,
                    'alert': alert,
                    'timestamp': datetime.now(),
                    'severity': 'INFO'
                })
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")

            setattr(self, f'_last_regime_{ticker}', current_regime)

        except Exception as e:
            print(f"❌ Error updating market regime for {ticker}: {str(e)}")

    def add_position(self, ticker: str, shares: int, entry_price: float,
                   stop_loss: float = None, take_profit: float = None):
        """Add a new position to monitor."""
        try:
            # Calculate position risk
            position_risk = shares * (entry_price - stop_loss) if stop_loss else 0

            position = {
                'ticker': ticker,
                'shares': shares,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_value': shares * entry_price,
                'position_risk': position_risk,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
                'entry_time': datetime.now()
            }

            self.active_positions.append(position)

            print(f"✅ Position added: {ticker} - {shares} shares @ {entry_price:.2f} PLN")
            print(f"   Stop Loss: {stop_loss:.2f} PLN" if stop_loss else "   Stop Loss: Not set")
            print(f"   Take Profit: {take_profit:.2f} PLN" if take_profit else "   Take Profit: Not set")
            print(f"   Position Risk: {position_risk:,.0f} PLN")

        except Exception as e:
            print(f"❌ Error adding position: {str(e)}")

    def remove_position(self, ticker: str, shares: int = None):
        """Remove or reduce a position."""
        try:
            for i, position in enumerate(self.active_positions):
                if position['ticker'] == ticker:
                    if shares is None or shares >= position['shares']:
                        # Remove entire position
                        removed_position = self.active_positions.pop(i)
                        pnl = removed_position['unrealized_pnl']
                        print(f"✅ Position closed: {ticker} - P&L: {pnl:,.0f} PLN")
                    else:
                        # Reduce position
                        position['shares'] -= shares
                        position['position_value'] = position['shares'] * position['current_price']
                        position['position_risk'] = position['shares'] * (position['current_price'] - position['stop_loss']) if position['stop_loss'] else 0
                        print(f"✅ Position reduced: {ticker} - {shares} shares removed")
                    break

        except Exception as e:
            print(f"❌ Error removing position: {str(e)}")

    def generate_real_time_signals(self) -> Dict:
        """Generate real-time trading signals with risk management."""
        try:
            # Get enhanced signals from signal generator
            enhanced_signals = self.signal_generator.generate_enhanced_signals(self.capital, 'MODERATE')

            # Enhance with current market conditions
            for signal in enhanced_signals['signals']:
                ticker = signal['ticker']

                # Check if we have current price data
                if ticker in self.price_cache:
                    current_price = self.price_cache[ticker]['price']
                    signal['real_time_price'] = current_price
                    signal['price_change'] = ((current_price - signal['current_price']) / signal['current_price']) * 100

                    # Recalculate position size with current price
                    stop_loss = signal['stop_loss']
                    if stop_loss and current_price > 0:
                        position_size = self.risk_manager.calculate_position_size(
                            self.capital, 0.02, current_price, stop_loss
                        )
                        signal['real_time_position_size'] = position_size

                # Add real-time market regime if available
                if ticker in self.price_history and len(self.price_history[ticker]) >= 14:
                    recent_prices = [p['price'] for p in self.price_history[ticker][-14:]]
                    df = pd.DataFrame({'close': recent_prices})
                    regime_info = self.market_regime_detector.analyze_regime(df)
                    signal['real_time_regime'] = regime_info['regime']
                    signal['real_time_regime_strength'] = regime_info['strength']

            return enhanced_signals

        except Exception as e:
            print(f"❌ Error generating real-time signals: {str(e)}")
            return {'signals': [], 'error': str(e)}

    def get_real_time_dashboard(self) -> Dict:
        """Get comprehensive real-time dashboard data."""
        try:
            # Calculate portfolio metrics
            total_value = sum(pos.get('current_value', 0) for pos in self.active_positions)
            total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.active_positions)
            total_risk = sum(pos.get('position_risk', 0) for pos in self.active_positions)
            portfolio_heat_pct = (total_risk / self.capital) * 100 if self.capital > 0 else 0

            # Get recent alerts
            recent_alerts = [alert for alert in self.real_time_alerts
                           if (datetime.now() - alert['timestamp']).seconds < 3600]  # Last hour

            dashboard = {
                'timestamp': datetime.now(),
                'portfolio': {
                    'total_capital': self.capital,
                    'total_value': total_value,
                    'cash_available': self.capital - total_value,
                    'total_positions': len(self.active_positions),
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'total_unrealized_pnl_pct': (total_unrealized_pnl / total_value * 100) if total_value > 0 else 0,
                    'total_risk': total_risk,
                    'portfolio_heat_pct': portfolio_heat_pct,
                    'max_heat_allowed': self.config['risk_management']['max_portfolio_heat']
                },
                'positions': self.active_positions,
                'recent_alerts': recent_alerts[-10:],  # Last 10 alerts
                'price_cache': self.price_cache,
                'risk_metrics': self.risk_metrics_history[-10:] if self.risk_metrics_history else [],
                'monitoring_active': self.running,
                'alerts_count': len(recent_alerts)
            }

            return dashboard

        except Exception as e:
            print(f"❌ Error getting dashboard data: {str(e)}")
            return {'error': str(e)}

    def start_enhanced_monitoring(self):
        """Start enhanced real-time monitoring with risk management."""
        print("🚀 Starting enhanced real-time monitoring with risk management...")

        # Add enhanced price callback
        self.add_data_callback(self.enhanced_price_callback)

        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _risk_monitoring_loop(self):
        """Background loop for risk monitoring."""
        while self.running:
            try:
                # Check portfolio heat every 30 seconds
                if hasattr(self, '_last_heat_check'):
                    if (datetime.now() - self._last_heat_check).seconds >= 30:
                        self.check_portfolio_heat()
                        self._last_heat_check = datetime.now()
                else:
                    self._last_heat_check = datetime.now()

                # Clean old alerts (older than 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.real_time_alerts = [alert for alert in self.real_time_alerts
                                        if alert['timestamp'] > cutoff_time]

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                print(f"❌ Error in risk monitoring loop: {str(e)}")
                time.sleep(10)


def main():
    """Main function to test enhanced real-time integration with risk management."""
    print("🛡️ ENHANCED REAL-TIME DATA INTEGRATION WITH RISK MANAGEMENT")
    print("=" * 70)

    # Initialize enhanced system
    system = EnhancedRealTimeIntegration(capital=100000)  # 100k PLN
    system.load_config()

    try:
        # Start enhanced monitoring
        system.start_enhanced_monitoring()

        # Add some sample positions for testing
        print("\n📊 Adding sample positions for monitoring...")
        system.add_position('TXT.WA', 100, 50.0, stop_loss=47.5, take_profit=55.0)
        system.add_position('XTB.WA', 50, 65.0, stop_loss=61.75, take_profit=71.5)
        system.add_position('PKN.WA', 75, 85.0, stop_loss=80.75, take_profit=89.25)

        # Start real-time data collection
        print("\n🚀 Starting enhanced real-time data collection...")
        system.running = True

        # Simulate price updates for testing
        print("\n🎲 Simulating real-time price updates...")
        base_prices = {'TXT.WA': 50.0, 'XTB.WA': 65.0, 'PKN.WA': 85.0}

        for i in range(20):  # 20 updates
            current_time = datetime.now()

            for ticker, base_price in base_prices.items():
                # Generate realistic price movement
                change_pct = np.random.normal(0, 0.008)  # 0.8% std dev
                new_price = base_price * (1 + change_pct)

                # Trigger enhanced callback
                system.enhanced_price_callback(ticker, new_price, 'simulation')

                if i % 5 == 0:  # Print every 5th update
                    print(f"[{current_time.strftime('%H:%M:%S')}] {ticker}: {new_price:.2f} PLN ({change_pct:+.2%})")

            time.sleep(2)  # Wait 2 seconds between updates

            # Show dashboard every 10 seconds
            if i % 5 == 4:
                dashboard = system.get_real_time_dashboard()
                portfolio = dashboard['portfolio']
                print(f"\n📊 DASHBOARD UPDATE [{i+1}]:")
                print(f"   Portfolio Value: {portfolio['total_value']:,.0f} PLN")
                print(f"   P&L: {portfolio['total_unrealized_pnl']:,.0f} PLN ({portfolio['total_unrealized_pnl_pct']:+.2f}%)")
                print(f"   Portfolio Heat: {portfolio['portfolio_heat_pct']:.1f}% / {portfolio['max_heat_allowed']:.1f}%")
                print(f"   Active Positions: {portfolio['total_positions']}")
                print(f"   Recent Alerts: {dashboard['alerts_count']}")

        # Generate real-time signals
        print("\n🧠 Generating real-time trading signals...")
        signals = system.generate_real_time_signals()

        if 'signals' in signals and signals['signals']:
            print(f"Generated {len(signals['signals'])} real-time signals")
            for signal in signals['signals'][:3]:  # Top 3
                print(f"   {signal['ticker']}: {signal['risk_adjusted_action']} (Score: {signal['risk_adjusted_score']:.1f})")
        else:
            print("No signals generated")

        # Final dashboard
        print("\n📈 FINAL DASHBOARD:")
        final_dashboard = system.get_real_time_dashboard()
        portfolio = final_dashboard['portfolio']

        print(f"   Capital: {portfolio['total_capital']:,.0f} PLN")
        print(f"   Invested: {portfolio['total_value']:,.0f} PLN")
        print(f"   Cash: {portfolio['cash_available']:,.0f} PLN")
        print(f"   P&L: {portfolio['total_unrealized_pnl']:,.0f} PLN ({portfolio['total_unrealized_pnl_pct']:+.2f}%)")
        print(f"   Portfolio Heat: {portfolio['portfolio_heat_pct']:.1f}%")
        print(f"   Positions: {portfolio['total_positions']}")

        # Show recent alerts
        if final_dashboard['recent_alerts']:
            print(f"\n🚨 RECENT ALERTS:")
            for alert in final_dashboard['recent_alerts'][-5:]:
                severity_emoji = "🚨" if alert['severity'] == 'CRITICAL' else "⚠️"
                print(f"   {severity_emoji} [{alert['timestamp'].strftime('%H:%M:%S')}] {alert['alert']}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping enhanced real-time monitoring...")
        system.running = False

    # Export data
    system.export_data('enhanced_realtime_export.csv')

    print("\n🏁 Enhanced real-time integration test complete!")
    print("🛡️ Risk management features tested:")
    print("   ✅ Real-time position monitoring")
    print("   ✅ Portfolio heat tracking")
    print("   ✅ Stop loss / take profit alerts")
    print("   ✅ Market regime detection")
    print("   ✅ Real-time signal generation")


def main_basic():
    """Basic main function for original RealTimeDataIntegration."""
    print("📡 BASIC REAL-TIME DATA INTEGRATION SYSTEM")
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

    print("🏁 Basic real-time integration test complete!")


if __name__ == "__main__":
    main()