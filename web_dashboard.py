#!/usr/bin/env python3
"""
Simple Web Dashboard for WIG30/WIG20 Investment Strategy Bot
Integrates with the existing wig30_bot.py to provide web interface
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
from config import ACTIVE_INDEX, ROE_THRESHOLD, PE_THRESHOLD, ENABLE_DUAL_FILTER
from trading_chart_service import chart_service

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to store latest analysis
latest_analysis = None
last_update = None

def load_latest_analysis():
    """Load the latest analysis results from CSV files"""
    global latest_analysis, last_update

    try:
        # Load the complete analysis with all companies
        if ACTIVE_INDEX == 'WIG20':
            csv_file = 'wig20_analysis.csv'
        else:
            csv_file = 'wig30_analysis.csv'

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # Add investment decisions for all companies
            df['decision'] = df.apply(lambda row: make_investment_decision(row), axis=1)
            df['decision_color'] = df.apply(lambda row: get_decision_color(row), axis=1)

            latest_analysis = {
                'all_stocks': df.to_dict('records'),
                'count': len(df),
                'recommendations': df[df['decision'] == 'KUP'].to_dict('records'),
                'index': ACTIVE_INDEX,
                'thresholds': {
                    'roe': ROE_THRESHOLD,
                    'pe': PE_THRESHOLD,
                    'dual_filter': ENABLE_DUAL_FILTER
                },
                'timestamp': datetime.now().isoformat()
            }
            last_update = datetime.now()
            return True
        else:
            return False
    except Exception as e:
        print(f"Error loading analysis: {e}")
        return False

def make_investment_decision(row):
    """Make investment decision based on ROE and P/E criteria"""
    if not row['profitable']:
        return 'SPRZEDAJ'

    roe = row.get('roe', 0)
    pe_ratio = row.get('pe_ratio', 999)

    # Handle missing data
    if pd.isna(roe) or pd.isna(pe_ratio):
        return 'TRZYMAJ'  # Insufficient data

    # Strong BUY - meets both criteria (ROE ≥ 10% AND P/E ≤ 15)
    if roe >= ROE_THRESHOLD and pe_ratio <= PE_THRESHOLD:
        return 'KUP'

    # HOLD - moderate criteria or close to thresholds
    elif (roe >= ROE_THRESHOLD * 0.8 or pe_ratio <= PE_THRESHOLD * 1.2) and roe > 5:
        return 'TRZYMAJ'

    # SELL - doesn't meet criteria
    else:
        return 'SPRZEDAJ'

def get_decision_color(row):
    """Get color for investment decision"""
    decision = row['decision']
    if decision == 'KUP':
        return 'text-green-400'
    elif decision == 'TRZYMAJ':
        return 'text-yellow-400'
    else:
        return 'text-red-400'

def run_analysis():
    """Run the WIG30/WIG20 analysis bot"""
    try:
        # Run the bot directly (already in virtual environment)
        result = subprocess.run(
            ['python', 'wig30_bot.py'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )

        if result.returncode == 0:
            # Load the new analysis results
            if load_latest_analysis():
                return True, "Analysis completed successfully"
            else:
                return False, "Analysis completed but no results found"
        else:
            return False, f"Analysis failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Analysis timed out"
    except Exception as e:
        return False, f"Error running analysis: {str(e)}"

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analysis')
def get_analysis():
    """API endpoint to get latest analysis results"""
    if latest_analysis is None:
        if not load_latest_analysis():
            return jsonify({'error': 'No analysis data available'}), 404

    return jsonify(latest_analysis)

@app.route('/api/all_stocks')
def get_all_stocks():
    """API endpoint to get all stocks with investment decisions"""
    if latest_analysis is None:
        if not load_latest_analysis():
            return jsonify({'error': 'No analysis data available'}), 404

    return jsonify({
        'stocks': latest_analysis['all_stocks'],
        'count': latest_analysis['count'],
        'timestamp': latest_analysis['timestamp']
    })

@app.route('/api/run_analysis', methods=['POST'])
def run_new_analysis():
    """API endpoint to run new analysis"""
    success, message = run_analysis()

    if success:
        return jsonify({
            'success': True,
            'message': message,
            'data': latest_analysis
        })
    else:
        return jsonify({
            'success': False,
            'message': message
        }), 500

@app.route('/api/status')
def get_status():
    """API endpoint to get system status"""
    return jsonify({
        'last_update': last_update.isoformat() if last_update else None,
        'active_index': ACTIVE_INDEX,
        'recommendations_count': len(latest_analysis['recommendations']) if latest_analysis else 0,
        'thresholds': {
            'roe_threshold': ROE_THRESHOLD,
            'pe_threshold': PE_THRESHOLD,
            'dual_filter_enabled': ENABLE_DUAL_FILTER
        }
    })

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'active_index': ACTIVE_INDEX,
            'roe_threshold': ROE_THRESHOLD,
            'pe_threshold': PE_THRESHOLD,
            'dual_filter': ENABLE_DUAL_FILTER
        })
    else:
        # Note: This would require modifying config.py dynamically
        # For now, just return current config
        return jsonify({'message': 'Config update not implemented yet'})

@app.route('/api/chart/<ticker>')
def get_chart_data(ticker):
    """Get chart data for a specific ticker"""
    try:
        period = request.args.get('period', '1y')
        indicators = request.args.getlist('indicators') or ['SMA_20', 'RSI_14']

        data = chart_service.get_stock_data(ticker, period)
        if data:
            formatted_data = chart_service.format_chart_data(data, indicators)
            return jsonify(formatted_data)
        else:
            return jsonify({'error': f'No data available for {ticker}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare')
def compare_stocks():
    """Compare multiple stocks"""
    try:
        tickers = request.args.getlist('tickers')
        period = request.args.get('period', '1y')

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        data = chart_service.get_multiple_stocks_data(tickers, period)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/indicators/<ticker>')
def get_indicators(ticker):
    """Get technical indicators for a ticker"""
    try:
        data = chart_service.get_stock_data(ticker, '1y')
        if data and 'data' in data:
            df = pd.DataFrame(data['data'])
            latest = df.iloc[-1]

            return jsonify({
                'ticker': ticker,
                'price': float(latest['Close']),
                'rsi': float(latest.get('RSI_14', 0)) if pd.notna(latest.get('RSI_14')) else None,
                'sma_20': float(latest.get('SMA_20', 0)) if pd.notna(latest.get('SMA_20')) else None,
                'sma_50': float(latest.get('SMA_50', 0)) if pd.notna(latest.get('SMA_50')) else None,
                'macd': float(latest.get('MACD', 0)) if pd.notna(latest.get('MACD')) else None,
                'volume': int(latest['Volume']) if pd.notna(latest.get('Volume')) else 0,
                'atr': float(latest.get('ATR_14', 0)) if pd.notna(latest.get('ATR_14')) else None
            })
        else:
            return jsonify({'error': f'No data available for {ticker}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to GPW Smart Analyzer'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to real-time updates for a specific ticker"""
    ticker = data.get('ticker')
    if ticker:
        room = f'ticker_{ticker}'
        join_room(room)
        emit('subscribed', {'ticker': ticker})
        print(f'Client {request.sid} subscribed to {ticker}')

        # Start real-time updates for this ticker
        if not hasattr(app, 'ticker_subscriptions'):
            app.ticker_subscriptions = {}

        app.ticker_subscriptions[ticker] = True

        # Start monitoring thread if not already running
        if not hasattr(app, 'monitoring_active') or not app.monitoring_active:
            start_price_monitoring()

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Unsubscribe from real-time updates for a specific ticker"""
    ticker = data.get('ticker')
    if ticker:
        room = f'ticker_{ticker}'
        leave_room(room)
        emit('unsubscribed', {'ticker': ticker})
        print(f'Client {request.sid} unsubscribed from {ticker}')

        if hasattr(app, 'ticker_subscriptions'):
            app.ticker_subscriptions[ticker] = False

def start_price_monitoring():
    """Start background price monitoring thread"""
    def monitor_prices():
        app.monitoring_active = True
        print("Price monitoring started")

        while app.monitoring_active:
            try:
                # Check if any ticker is being monitored
                if hasattr(app, 'ticker_subscriptions'):
                    for ticker, is_active in app.ticker_subscriptions.items():
                        if is_active:
                            # Fetch latest price data
                            data = chart_service.get_stock_data(ticker, '1d')
                            if data and data.get('info'):
                                latest_price = data['info']['current_price']
                                change = data['info']['change']
                                change_percent = data['info']['change_percent']

                                # Broadcast price update to subscribers
                                room = f'ticker_{ticker}'
                                socketio.emit('price_update', {
                                    'ticker': ticker,
                                    'price': latest_price,
                                    'change': change,
                                    'change_percent': change_percent,
                                    'timestamp': datetime.now().isoformat()
                                }, room=room)

                                print(f"Price update for {ticker}: {latest_price} ({change_percent:+.2f}%)")

                # Wait before next update (30 seconds)
                time.sleep(30)

            except Exception as e:
                print(f"Error in price monitoring: {e}")
                time.sleep(5)  # Wait before retrying

    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_prices, daemon=True)
    monitor_thread.start()

if __name__ == '__main__':
    # Load existing analysis on startup
    load_latest_analysis()

    print("🚀 WIG30/WIG20 Investment Dashboard")
    print(f"📊 Active Index: {ACTIVE_INDEX}")
    print(f"🎯 ROE Threshold: {ROE_THRESHOLD}%")
    print(f"💰 P/E Threshold: {PE_THRESHOLD}")
    print(f"🔧 Dual Filter: {'Enabled' if ENABLE_DUAL_FILTER else 'Disabled'}")
    print("🌐 Dashboard available at: http://localhost:5000")
    print("=" * 50)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)