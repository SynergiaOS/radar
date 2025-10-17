#!/usr/bin/env python3
"""
HTML Dashboard Generator for Stock Monitoring
Creates real-time web dashboard for stock tracking
"""

import pandas as pd
import json
from datetime import datetime
import os

def generate_dashboard_html():
    """Generate HTML dashboard for stock monitoring."""

    # Load monitoring data
    data_file = 'monitored_stocks.json'
    log_file = 'monitoring_log.csv'

    stocks_data = []
    alerts_data = []

    # Load stock data
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            stocks_data = data.get('monitored_stocks', [])
            price_summary = data.get('price_history_summary', {})

    # Load recent alerts
    if os.path.exists(log_file):
        alerts_df = pd.read_csv(log_file)
        # Get last 24 hours of alerts
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        recent_alerts = alerts_df[alerts_df['timestamp'] > datetime.now() - pd.Timedelta(hours=24)]
        alerts_data = recent_alerts.to_dict('records')

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Stock Monitoring Dashboard - WIG30</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}

        .header h1 {{
            color: #2c3e50;
            font-size: 2.2em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
        }}

        .stat-card h3 {{
            color: #34495e;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .main-content {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 25px;
        }}

        .stocks-section, .alerts-section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}

        .section-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}

        .stock-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            transition: all 0.3s ease;
        }}

        .stock-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .ticker {{
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.2em;
        }}

        .price {{
            font-weight: bold;
            color: #27ae60;
            font-size: 1.1em;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}

        .metric {{
            font-size: 0.9em;
        }}

        .metric-label {{
            color: #7f8c8d;
        }}

        .metric-value {{
            font-weight: bold;
            color: #2c3e50;
        }}

        .alert {{
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }}

        .alert.price-up {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left-color: #28a745;
        }}

        .alert.price-down {{
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left-color: #dc3545;
        }}

        .alert-time {{
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
        }}

        .alert-content {{
            font-weight: bold;
        }}

        .no-data {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            padding: 40px;
        }}

        .footer {{
            text-align: center;
            margin-top: 30px;
            color: rgba(255,255,255,0.8);
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Stock Monitoring Dashboard</h1>
            <div class="subtitle">Real-time tracking of WIG30 companies • Last updated: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>🏢 Monitorowanych Spółek</h3>
                <div class="stat-value">{len(stocks_data)}</div>
            </div>
            <div class="stat-card">
                <h3>🚨 Alertów (24h)</h3>
                <div class="stat-value">{len(alerts_data)}</div>
            </div>
            <div class="stat-card">
                <h3>💰 Średnie ROE</h3>
                <div class="stat-value">{sum(s.get('roe', 0) for s in stocks_data)/len(stocks_data) if stocks_data else 0:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>📈 Średnie P/E</h3>
                <div class="stat-value">{sum(s.get('pe_ratio', 0) for s in stocks_data)/len(stocks_data) if stocks_data else 0:.1f}</div>
            </div>
        </div>

        <div class="main-content">
            <div class="stocks-section">
                <h2 class="section-title">📈 Monitorowane Spółki</h2>
                """

    # Add stock cards
    if stocks_data:
        for stock in stocks_data:
            ticker = stock['ticker']
            name = stock['name']
            roe = stock.get('roe', 0)
            pe_ratio = stock.get('pe_ratio', 0)
            net_income = stock.get('net_income', 0)

            # Get latest price from summary
            latest_price = price_summary.get(ticker, {}).get('latest_price', 'N/A')
            if latest_price != 'N/A':
                latest_price = f"{latest_price:.2f} PLN"

            html_content += f"""
                <div class="stock-card">
                    <div class="stock-header">
                        <span class="ticker">{ticker}</span>
                        <span class="price">{latest_price}</span>
                    </div>
                    <div style="font-size: 0.9em; color: #555; margin-bottom: 8px;">{name[:40]}</div>
                    <div class="metrics">
                        <div class="metric">
                            <span class="metric-label">ROE:</span>
                            <span class="metric-value">{roe:.1f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">P/E:</span>
                            <span class="metric-value">{pe_ratio:.1f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Zysk:</span>
                            <span class="metric-value">{net_income/1000000:.1f}M PLN</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Status:</span>
                            <span class="metric-value" style="color: #27ae60;">✅ Aktywna</span>
                        </div>
                    </div>
                </div>
                """
    else:
        html_content += '<div class="no-data">Brak danych o spółkach. Uruchom najpierw analizę WIG30.</div>'

    # Add alerts section
    html_content += f"""
            </div>

            <div class="alerts-section">
                <h2 class="section-title">🚨 Ostatnie Alerty</h2>
                """

    if alerts_data:
        for alert in alerts_data[-10:]:  # Last 10 alerts
            alert_time = pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')
            ticker = alert['ticker']
            price = alert['price']
            change_pct = alert['change_pct']
            alert_class = 'price-up' if change_pct > 0 else 'price-down'
            emoji = '📈' if change_pct > 0 else '📉'

            html_content += f"""
                <div class="alert {alert_class}">
                    <div class="alert-time">{alert_time}</div>
                    <div class="alert-content">
                        {emoji} {ticker}: {price:.2f} PLN ({change_pct:+.2f}%)
                    </div>
                </div>
                """
    else:
        html_content += '<div class="no-data">Brak alertów w ostatnich 24h.</div>'

    html_content += f"""
            </div>
        </div>

        <div class="footer">
            <p>📊 Dashboard auto-odświeżany • Dane z Yahoo Finance • Aktualizacja: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
            <p>⚠️ Disclaimer: Dane mają charakter informacyjny, nie stanowią porady inwestycyjnej</p>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function(){{
            location.reload();
        }}, 30000);

        // Add some interactivity
        document.querySelectorAll('.stock-card').forEach(card => {{
            card.addEventListener('click', function() {{
                this.style.backgroundColor = '#e3f2fd';
                setTimeout(() => {{
                    this.style.backgroundColor = '';
                }}, 500);
            }});
        }});
    </script>
</body>
</html>
    """

    # Save HTML file
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("✅ Dashboard HTML wygenerowany: dashboard.html")
    print("🌐 Otwórz w przeglądarce: file://" + os.path.abspath('dashboard.html'))

def main():
    """Generate dashboard and provide instructions."""
    print("📊 Generowanie dashboardu HTML...")

    generate_dashboard_html()

    print("\n🚀 Dashboard gotowy!")
    print("💡 Wskazówki:")
    print("   • Otwórz dashboard.html w przeglądarce")
    print("   • Dashboard odświeża się automatycznie co 30 sekund")
    print("   • Uruchom monitor.py aby aktywować śledzenie na żywo")
    print("   • Alerty pojawią się automatycznie przy dużych zmianach cen")

if __name__ == "__main__":
    main()