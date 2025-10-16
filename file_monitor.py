#!/usr/bin/env python3
"""
File-Based Stock Monitoring System
Monitors stocks using local data files instead of API calls
Supports CSV, Excel, and GPW format files
"""

import pandas as pd
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import glob

# Configuration
MONITORING_INTERVAL = 15  # seconds between file checks
PRICE_CHANGE_THRESHOLD = 1.5  # percentage change for alerts
DATA_FILE = 'file_monitored_stocks.json'
LOG_FILE = 'file_monitoring_log.csv'

# File paths to monitor
STOCK_DATA_PATHS = [
    'data/current_prices.csv',        # Current prices file
    'data/gpw_quotes.csv',            # GPW quotes
    'data/stock_data.xlsx',           # Excel file
    'data/live_prices.json',          # JSON prices
    '*.csv',                          # Any CSV in current directory
]

class FileStockMonitor:
    """File-based stock monitoring system."""

    def __init__(self):
        self.monitored_stocks = []
        self.price_history = {}
        self.alerts_triggered = []
        self.monitoring_start_time = datetime.now()
        self.last_file_modification = {}

    def load_previous_analysis(self, csv_file: str = 'wig30_analysis_pe_threshold.csv'):
        """Load previously analyzed companies to monitor."""
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                self.monitored_stocks = df.to_dict('records')
                print(f"📊 Załadowano {len(self.monitored_stocks)} spółek do monitoringu z pliku")

                # Initialize price tracking
                for stock in self.monitored_stocks:
                    ticker = stock['ticker']
                    self.price_history[ticker] = []

                return True
            else:
                print(f"❌ Nie znaleziono pliku {csv_file}")
                return False
        except Exception as e:
            print(f"❌ Błąd ładowania danych: {str(e)}")
            return False

    def read_stock_data_from_files(self) -> Dict[str, float]:
        """Read current stock prices from multiple file sources."""
        prices = {}

        # Try different file sources
        file_sources = [
            self._read_csv_prices,
            self._read_excel_prices,
            self._read_json_prices,
            self._read_gpw_format,
        ]

        for read_func in file_sources:
            try:
                file_prices = read_func()
                if file_prices:
                    prices.update(file_prices)
                    print(f"📂 Wczytano {len(file_prices)} cen z pliku")
                    break
            except Exception as e:
                print(f"⚠️  Problem z odczytem pliku: {str(e)}")
                continue

        return prices

    def _read_csv_prices(self) -> Dict[str, float]:
        """Read prices from CSV files."""
        prices = {}

        # Try specific data files first
        data_files = [
            'data/current_prices.csv',
            'data/gpw_quotes.csv',
            'current_prices.csv',
            'wig30_prices.csv'
        ]

        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    # Try different column names for ticker and price
                    ticker_cols = ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']
                    price_cols = ['price', 'close', 'Price', 'Close', 'PRICE', 'CLOSE', 'kurs', 'Kurs']

                    ticker_col = None
                    price_col = None

                    for col in ticker_cols:
                        if col in df.columns:
                            ticker_col = col
                            break

                    for col in price_cols:
                        if col in df.columns:
                            price_col = col
                            break

                    if ticker_col and price_col:
                        for _, row in df.iterrows():
                            ticker = str(row[ticker_col]).strip()
                            price = row[price_col]
                            try:
                                price_float = float(price)
                                if price_float > 0:
                                    prices[ticker] = price_float
                            except (ValueError, TypeError):
                                continue

                        print(f"✅ Wczytano dane z CSV: {file_path}")
                        return prices

                except Exception as e:
                    print(f"⚠️  Błąd czytania CSV {file_path}: {str(e)}")
                    continue

        return prices

    def _read_excel_prices(self) -> Dict[str, float]:
        """Read prices from Excel files."""
        prices = {}

        excel_files = [
            'data/stock_data.xlsx',
            'data/gpw_data.xlsx',
            'stock_data.xlsx'
        ]

        for file_path in excel_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)

                    # Similar logic to CSV
                    ticker_cols = ['ticker', 'symbol', 'Ticker', 'Symbol']
                    price_cols = ['price', 'close', 'kurs', 'Kurs']

                    ticker_col = None
                    price_col = None

                    for col in ticker_cols:
                        if col in df.columns:
                            ticker_col = col
                            break

                    for col in price_cols:
                        if col in df.columns:
                            price_col = col
                            break

                    if ticker_col and price_col:
                        for _, row in df.iterrows():
                            ticker = str(row[ticker_col]).strip()
                            price = row[price_col]
                            try:
                                price_float = float(price)
                                if price_float > 0:
                                    prices[ticker] = price_float
                            except (ValueError, TypeError):
                                continue

                        print(f"✅ Wczytano dane z Excel: {file_path}")
                        return prices

                except Exception as e:
                    print(f"⚠️  Błąd czytania Excel {file_path}: {str(e)}")
                    continue

        return prices

    def _read_json_prices(self) -> Dict[str, float]:
        """Read prices from JSON files."""
        prices = {}

        json_files = [
            'data/live_prices.json',
            'data/stock_prices.json',
            'live_prices.json'
        ]

        for file_path in json_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle different JSON structures
                    if isinstance(data, dict):
                        for ticker, price_data in data.items():
                            if isinstance(price_data, dict) and 'price' in price_data:
                                prices[ticker] = float(price_data['price'])
                            elif isinstance(price_data, (int, float)):
                                prices[ticker] = float(price_data)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'ticker' in item and 'price' in item:
                                prices[item['ticker']] = float(item['price'])

                    if prices:
                        print(f"✅ Wczytano dane z JSON: {file_path}")
                        return prices

                except Exception as e:
                    print(f"⚠️  Błąd czytania JSON {file_path}: {str(e)}")
                    continue

        return prices

    def _read_gpw_format(self) -> Dict[str, float]:
        """Read prices from GPW format files."""
        prices = {}

        # Look for files with GPW naming pattern
        gpw_files = glob.glob('*gpw*.csv') + glob.glob('*GPW*.csv')

        for file_path in gpw_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Parse GPW format (usually comma or semicolon separated)
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.strip().replace(';', ',').split(',')
                        if len(parts) >= 2:
                            ticker = parts[0].strip().strip('"')
                            try:
                                price = float(parts[1].strip().strip('"'))
                                if price > 0:
                                    prices[ticker] = price
                            except (ValueError, TypeError):
                                continue

                if prices:
                    print(f"✅ Wczytano dane GPW: {file_path}")
                    return prices

            except Exception as e:
                print(f"⚠️  Błąd czytania GPW {file_path}: {str(e)}")
                continue

        return prices

    def create_sample_data_file(self):
        """Create a sample data file for testing."""
        sample_data = [
            {'ticker': 'TXT.WA', 'price': 51.00, 'change': 0.5, 'volume': 1000},
            {'ticker': 'XTB.WA', 'price': 67.44, 'change': -0.3, 'volume': 1500},
            {'ticker': 'PKN.WA', 'price': 87.76, 'change': 1.2, 'volume': 5000},
            {'ticker': 'PKO.WA', 'price': 73.72, 'change': -0.8, 'volume': 8000},
        ]

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Save as CSV
        df = pd.DataFrame(sample_data)
        csv_file = 'data/current_prices.csv'
        df.to_csv(csv_file, index=False)

        print(f"📁 Utworzono przykładowy plik danych: {csv_file}")
        return csv_file

    def update_prices_from_files(self):
        """Update prices by reading from files."""
        print(f"\n🔄 Aktualizacja cen z plików - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)

        # Read current prices from files
        current_prices = self.read_stock_data_from_files()

        if not current_prices:
            print("❌ Brak danych z plików. Tworzę przykładowy plik...")
            self.create_sample_data_file()
            current_prices = self.read_stock_data_from_files()

        for stock in self.monitored_stocks:
            ticker = stock['ticker']
            name = stock['name']

            current_price = current_prices.get(ticker)

            if current_price:
                # Add to price history
                self.price_history[ticker].append({
                    'price': current_price,
                    'timestamp': datetime.now()
                })

                # Keep only last 100 price points
                if len(self.price_history[ticker]) > 100:
                    self.price_history[ticker] = self.price_history[ticker][-100:]

                # Calculate price change if we have history
                price_change = 0
                price_change_pct = 0

                if len(self.price_history[ticker]) >= 2:
                    previous_price = self.price_history[ticker][-2]['price']
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100

                # Check for alerts
                if abs(price_change_pct) >= PRICE_CHANGE_THRESHOLD:
                    self.trigger_alert(ticker, name, current_price, price_change_pct)

                # Display current status
                change_emoji = "📈" if price_change_pct > 0 else "📉" if price_change_pct < 0 else "➡️"
                source_info = "[PLIK]"
                print(f"{change_emoji} {ticker:<8} | {current_price:>8.2f} PLN | {price_change_pct:>+6.2f}% | {source_info} | {name[:30]}")
            else:
                print(f"❌ {ticker:<8} | Brak danych w pliku | {name[:30]}")

    def trigger_alert(self, ticker: str, name: str, price: float, change_pct: float):
        """Trigger price alert for significant changes."""
        direction = "📈 WZROST" if change_pct > 0 else "📉 SPADEK"
        alert_msg = f"{direction} {ticker}: {price:.2f} PLN ({change_pct:+.2f}%) - {name}"

        # Check if this alert was recently triggered
        alert_key = f"{ticker}_{direction}"
        now = datetime.now()

        # Avoid duplicate alerts within 5 minutes
        recent_alerts = [a for a in self.alerts_triggered if
                        (now - a['timestamp']).seconds < 300 and a['key'] == alert_key]

        if not recent_alerts:
            self.alerts_triggered.append({
                'key': alert_key,
                'ticker': ticker,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'timestamp': now
            })

            print(f"\n🚨 ALERT! {alert_msg}")
            self.log_alert(ticker, name, price, change_pct)

    def log_alert(self, ticker: str, name: str, price: float, change_pct: float):
        """Log alert to CSV file."""
        try:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'alert_type': 'FILE_PRICE_CHANGE',
                'source': 'LOCAL_FILE'
            }

            # Create log file if it doesn't exist
            if not os.path.exists(LOG_FILE):
                df_log = pd.DataFrame([log_entry])
                df_log.to_csv(LOG_FILE, index=False)
            else:
                df_log = pd.read_csv(LOG_FILE)
                df_new = pd.DataFrame([log_entry])
                df_log = pd.concat([df_log, df_new], ignore_index=True)
                df_log.to_csv(LOG_FILE, index=False)

        except Exception as e:
            print(f"⚠️  Błąd logowania alertu: {str(e)}")

    def display_dashboard(self):
        """Display monitoring dashboard with current status."""
        os.system('cls' if os.name == 'nt' else 'clear')

        print("🏛️  FILE-BASED STOCK MONITORING DASHBOARD")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"🔄 Monitorowanych spółek: {len(self.monitored_stocks)}")
        print(f"⏱️  Czas działania: {datetime.now() - self.monitoring_start_time}")
        print(f"⚡ Częstotliwość aktualizacji: {MONITORING_INTERVAL}s")
        print(f"🚨 Próg alertu: ±{PRICE_CHANGE_THRESHOLD}%")
        print(f"📂 Źródło danych: PLIKI LOKALNE")
        print("=" * 80)

        # Display recent alerts
        recent_alerts = [a for a in self.alerts_triggered if
                        (datetime.now() - a['timestamp']).seconds < 3600]

        if recent_alerts:
            print(f"\n🚨 OSTATNIE ALERTY (ostatnia godzina):")
            for alert in recent_alerts[-5:]:  # Last 5 alerts
                emoji = "📈" if alert['change_pct'] > 0 else "📉"
                time_str = alert['timestamp'].strftime('%H:%M:%S')
                print(f"   {emoji} {time_str} {alert['ticker']}: {alert['price']:.2f} PLN ({alert['change_pct']:+.2f}%) [FILE]")

        print("\n📊 AKTUALNE CENY Z PLIKÓW I ZMIANY:")

    def display_summary_stats(self):
        """Display summary statistics for the monitoring session."""
        print(f"\n📈 PODSUMOWANIE SESJI MONITORINGU Z PLIKÓW:")
        print(f"   • Łącznie alertów: {len(self.alerts_triggered)}")
        print(f"   • Czas trwania: {datetime.now() - self.monitoring_start_time}")
        print(f"   • Źródło danych: PLIKI LOKALNE (bez API)")

        if self.price_history:
            print(f"   • Aktywnych spółek z danymi: {len([k for k, v in self.price_history.items() if v])}")

    def save_monitoring_data(self):
        """Save current monitoring data to JSON file."""
        try:
            data = {
                'session_start': self.monitoring_start_time.isoformat(),
                'last_update': datetime.now().isoformat(),
                'monitored_stocks': self.monitored_stocks,
                'alerts_count': len(self.alerts_triggered),
                'data_source': 'LOCAL_FILES',
                'price_history_summary': {
                    ticker: {
                        'latest_price': history[-1]['price'] if history else None,
                        'update_count': len(history)
                    }
                    for ticker, history in self.price_history.items()
                }
            }

            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️  Błąd zapisu danych: {str(e)}")

    def start_monitoring(self):
        """Start the file-based monitoring loop."""
        if not self.load_previous_analysis():
            print("❌ Nie można załadować danych do monitoringu")
            return

        print(f"\n🚀 Rozpoczynam monitoring z plików lokalnych...")
        print(f"📊 Monitorowanie {len(self.monitored_stocks)} spółek WIG30")
        print(f"⏱️  Interwał aktualizacji: {MONITORING_INTERVAL} sekund")
        print(f"🚨 Alerty przy zmianie ceny ≥ ±{PRICE_CHANGE_THRESHOLD}%")
        print(f"📂 Źródło: PLIKI CSV/Excel/JSON (bez API)")
        print(f"💾 Dane zapisywane w: {DATA_FILE}, {LOG_FILE}")
        print("\nNaciśnij Ctrl+C aby zatrzymać monitoring\n")

        try:
            while True:
                self.display_dashboard()
                self.update_prices_from_files()

                # Save monitoring data every 10 cycles
                if int((datetime.now() - self.monitoring_start_time).seconds / MONITORING_INTERVAL) % 10 == 0:
                    self.save_monitoring_data()

                time.sleep(MONITORING_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Zatrzymano monitoring plików")
            self.display_summary_stats()
            self.save_monitoring_data()
            print("💾 Dane zapisane. Do zobaczenia!")


def main():
    """Main function to start file-based stock monitoring."""
    monitor = FileStockMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()