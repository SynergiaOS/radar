#!/usr/bin/env python3
"""
Real-time Stock Monitoring System
Monitors WIG30 companies with real-time price tracking and alerts
"""

import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

# Configuration
MONITORING_INTERVAL = 30  # seconds between updates
PRICE_CHANGE_THRESHOLD = 2.0  # percentage change for alerts
DATA_FILE = 'monitored_stocks.json'
LOG_FILE = 'monitoring_log.csv'

class StockMonitor:
    """Real-time stock monitoring system for WIG30 companies."""

    def __init__(self):
        self.monitored_stocks = []
        self.price_history = {}
        self.alerts_triggered = []
        self.monitoring_start_time = datetime.now()

    def load_previous_analysis(self, csv_file: str = 'wig30_analysis_pe_threshold.csv'):
        """Load previously analyzed companies to monitor."""
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                self.monitored_stocks = df.to_dict('records')
                print(f"📊 Załadowano {len(self.monitored_stocks)} spółek do monitoringu")

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

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current stock price from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            # Try multiple methods for current price
            price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')

            if price and price > 0:
                return float(price)
            else:
                # Fallback to recent price data
                hist = stock.history(period='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])

        except Exception as e:
            print(f"⚠️  Błąd pobierania ceny {ticker}: {str(e)}")

        return None

    def update_prices(self):
        """Update current prices for all monitored stocks."""
        print(f"\n🔄 Aktualizacja cen - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)

        for stock in self.monitored_stocks:
            ticker = stock['ticker']
            name = stock['name']

            current_price = self.get_current_price(ticker)

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
                print(f"{change_emoji} {ticker:<8} | {current_price:>8.2f} PLN | {price_change_pct:>+6.2f}% | {name[:40]}")
            else:
                print(f"❌ {ticker:<8} | Brak danych ceny | {name[:40]}")

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
                'alert_type': 'PRICE_CHANGE'
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

        print("🏛️  REAL-TIME STOCK MONITORING DASHBOARD")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"🔄 Monitorowanych spółek: {len(self.monitored_stocks)}")
        print(f"⏱️  Czas działania: {datetime.now() - self.monitoring_start_time}")
        print(f"⚡ Częstotliwość aktualizacji: {MONITORING_INTERVAL}s")
        print(f"🚨 Próg alertu: ±{PRICE_CHANGE_THRESHOLD}%")
        print("=" * 80)

        # Display recent alerts
        recent_alerts = [a for a in self.alerts_triggered if
                        (datetime.now() - a['timestamp']).seconds < 3600]

        if recent_alerts:
            print(f"\n🚨 OSTATNIE ALERTY (ostatnia godzina):")
            for alert in recent_alerts[-5:]:  # Last 5 alerts
                emoji = "📈" if alert['change_pct'] > 0 else "📉"
                time_str = alert['timestamp'].strftime('%H:%M:%S')
                print(f"   {emoji} {time_str} {alert['ticker']}: {alert['price']:.2f} PLN ({alert['change_pct']:+.2f}%)")

        print("\n📊 AKTUALNE CENY I ZMIANY:")

    def display_summary_stats(self):
        """Display summary statistics for the monitoring session."""
        print(f"\n📈 PODSUMOWANIE SESJI MONITORINGU:")
        print(f"   • Łącznie alertów: {len(self.alerts_triggered)}")
        print(f"   • Czas trwania: {datetime.now() - self.monitoring_start_time}")

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
        """Start the real-time monitoring loop."""
        if not self.load_previous_analysis():
            print("❌ Nie można załadować danych do monitoringu")
            return

        print(f"\n🚀 Rozpoczynam monitoring w czasie rzeczywistym...")
        print(f"📊 Monitorowanie {len(self.monitored_stocks)} spółek z WIG30")
        print(f"⏱️  Interwał aktualizacji: {MONITORING_INTERVAL} sekund")
        print(f"🚨 Alerty przy zmianie ceny ≥ ±{PRICE_CHANGE_THRESHOLD}%")
        print(f"💾 Dane zapisywane w: {DATA_FILE}, {LOG_FILE}")
        print("\nNaciśnij Ctrl+C aby zatrzymać monitoring\n")

        try:
            while True:
                self.display_dashboard()
                self.update_prices()

                # Save monitoring data every 10 cycles
                if int((datetime.now() - self.monitoring_start_time).seconds / MONITORING_INTERVAL) % 10 == 0:
                    self.save_monitoring_data()

                time.sleep(MONITORING_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Zatrzymano monitoring")
            self.display_summary_stats()
            self.save_monitoring_data()
            print("💾 Dane zapisane. Do zobaczenia!")


def main():
    """Main function to start stock monitoring."""
    monitor = StockMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()