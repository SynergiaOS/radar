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
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import numpy as np
from src.config.config import ACTIVE_INDEX, ROE_THRESHOLD, PE_THRESHOLD, ENABLE_DUAL_FILTER, WIG30_TICKERS, WIG20_TICKERS
from src.services.integrations.trading_chart_service import chart_service
import yfinance as yf
from src.backend.analysis.technical_analysis import analyze_technical_indicators, calculate_rsi, detect_trend, get_trend_label

app = Flask(__name__)

# Configure CORS for frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3001", "http://127.0.0.1:3001"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to store latest analysis
latest_analysis = None
last_update = None

def clean_nan_values(obj):
    """Recursively clean NaN values from data structures, replacing with None"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return clean_nan_values(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

def safe_jsonify(data):
    """Safe JSON response that handles NaN values"""
    cleaned_data = clean_nan_values(data)
    return jsonify(cleaned_data)

def load_latest_analysis():
    """Load the latest analysis results from CSV files"""
    global latest_analysis, last_update

    try:
        # Load the complete analysis with all companies
        if ACTIVE_INDEX == 'WIG20':
            csv_file = 'wig20_analysis.csv'
            current_index_tickers = set(WIG20_TICKERS)
        else:
            csv_file = 'wig30_analysis.csv'
            current_index_tickers = set(WIG30_TICKERS)

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # Add investment decisions for all companies
            df['decision'] = df.apply(lambda row: make_investment_decision(row), axis=1)
            df['decision_color'] = df.apply(lambda row: get_decision_color(row), axis=1)

            # Add index information for each stock
            df['index'] = df['ticker'].apply(lambda ticker:
                'WIG20' if ticker in WIG20_TICKERS else
                'WIG30' if ticker in WIG30_TICKERS else
                'OTHER'
            )

            all_stocks = df.to_dict('records')

            # Filter by current index
            filtered_stocks = [stock for stock in all_stocks if stock['ticker'] in current_index_tickers]

            latest_analysis = {
                'all_stocks': all_stocks,
                'filtered_stocks': filtered_stocks,
                'count': len(all_stocks),
                'filtered_count': len(filtered_stocks),
                'recommendations': df[df['decision'] == 'KUP'].to_dict('records'),
                'filtered_recommendations': [stock for stock in df[df['decision'] == 'KUP'].to_dict('records')
                                           if stock['ticker'] in current_index_tickers],
                'index': ACTIVE_INDEX,
                'available_indices': ['WIG30', 'WIG20'],
                'index_stock_counts': {
                    'WIG30': len([t for t in all_stocks if t['ticker'] in WIG30_TICKERS]),
                    'WIG20': len([t for t in all_stocks if t['ticker'] in WIG20_TICKERS])
                },
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
            ['python', 'src/backend/trading/wig30_bot.py'],
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

    # Get requested index from query parameter
    requested_index = request.args.get('index', ACTIVE_INDEX)

    # If a specific index is requested, filter the data
    if requested_index in ['WIG30', 'WIG20'] and requested_index != ACTIVE_INDEX:
        # Filter stocks for requested index
        index_tickers = WIG30_TICKERS if requested_index == 'WIG30' else WIG20_TICKERS
        filtered_stocks = [stock for stock in latest_analysis['all_stocks']
                          if stock['ticker'] in index_tickers]
        filtered_recommendations = [stock for stock in latest_analysis['recommendations']
                                  if stock['ticker'] in index_tickers]

        analysis_copy = latest_analysis.copy()
        analysis_copy.update({
            'filtered_stocks': filtered_stocks,
            'filtered_recommendations': filtered_recommendations,
            'filtered_count': len(filtered_stocks),
            'index': requested_index
        })
        return safe_jsonify(analysis_copy)

    return safe_jsonify(latest_analysis)

@app.route('/api/all_stocks')
def get_all_stocks():
    """API endpoint to get all stocks with investment decisions"""
    if latest_analysis is None:
        if not load_latest_analysis():
            return jsonify({'error': 'No analysis data available'}), 404

    return safe_jsonify({
        'stocks': latest_analysis['all_stocks'],
        'count': latest_analysis['count'],
        'timestamp': latest_analysis['timestamp']
    })

@app.route('/api/run_analysis', methods=['POST'])
def run_new_analysis():
    """API endpoint to run new analysis"""
    success, message = run_analysis()

    if success:
        return safe_jsonify({
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
    return safe_jsonify({
        'last_update': last_update.isoformat() if last_update else None,
        'active_index': ACTIVE_INDEX,
        'recommendations_count': len(latest_analysis['recommendations']) if latest_analysis else 0,
        'thresholds': {
            'roe_threshold': ROE_THRESHOLD,
            'pe_threshold': PE_THRESHOLD,
            'dual_filter_enabled': ENABLE_DUAL_FILTER
        }
    })

@app.route('/api/indices')
def get_indices():
    """Get available indices and their stock counts"""
    return safe_jsonify({
        'available_indices': ['WIG30', 'WIG20'],
        'active_index': ACTIVE_INDEX,
        'wig30_stocks': WIG30_TICKERS,
        'wig20_stocks': WIG20_TICKERS,
        'wig30_count': len(WIG30_TICKERS),
        'wig20_count': len(WIG20_TICKERS)
    })

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    if request.method == 'GET':
        return safe_jsonify({
            'active_index': ACTIVE_INDEX,
            'roe_threshold': ROE_THRESHOLD,
            'pe_threshold': PE_THRESHOLD,
            'dual_filter': ENABLE_DUAL_FILTER,
            'available_indices': ['WIG30', 'WIG20']
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
            return safe_jsonify(formatted_data)
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
        return safe_jsonify(data)
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

            return safe_jsonify({
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

def get_enriched_stock_data(ticker: str) -> dict:
    """
    Get enriched stock data with technical indicators and sector information.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with comprehensive stock data
    """
    try:
        # Get basic stock data
        data = chart_service.get_stock_data(ticker, '1d')
        if not data or not data.get('data'):
            return {'error': f'No data available for {ticker}'}

        # Convert to DataFrame
        df = pd.DataFrame(data['data'])

        # Get additional yfinance data for market cap and volume
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get technical indicators
        indicators = analyze_technical_indicators(df)
        trend_result = detect_trend(df)

        # Calculate RSI
        rsi = calculate_rsi(df['Close'], 14)

        # Get sector
        sector = get_sector_for_ticker(ticker)

        # Get latest data
        latest = df.iloc[-1]

        # Map trend to English
        trend_map = {
            'Wzrostowy': 'upward',
            'Spadkowy': 'downward',
            'Boczny': 'sideways',
            'Nieznany': 'unknown'
        }

        return {
            'ticker': ticker,
            'name': data['info'].get('name', ticker),
            'price': float(latest['Close']),
            'change': float(latest['Close'] - latest['Open']),
            'change_percent': float(((latest['Close'] - latest['Open']) / latest['Open']) * 100),
            'volume': int(latest['Volume']) if pd.notna(latest.get('Volume')) else 0,
            'market_cap': info.get('marketCap'),
            'sector': sector,
            'rsi': float(rsi) if pd.notna(rsi) else None,
            'macd': float(indicators.get('macd', {}).get('values', [0, 0, 0, 0])[-1]) if indicators.get('macd') else None,
            'macd_signal': float(indicators.get('macd_signal', {}).get('values', [0, 0, 0, 0])[-1]) if indicators.get('macd_signal') else None,
            'trend': trend_map.get(get_trend_label(df), 'unknown'),
            'trend_label': get_trend_label(df),
            'sma5': float(indicators.get('sma5', {}).get('values', [0, 0, 0, 0])[-1]) if indicators.get('sma5') else None,
            'sma10': float(indicators.get('sma10', {}).get('values', [0, 0, 0, 0])[-1]) if indicators.get('sma10') else None,
            'sma20': float(indicators.get('sma20', {}).get('values', [0, 0, 0, 0])[-1]) if indicators.get('sma20') else None,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error getting enriched data for {ticker}: {e}")
        return {
            'ticker': ticker,
            'name': ticker,
            'price': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'market_cap': None,
            'sector': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'trend': 'unknown',
            'trend_label': 'Nieznany',
            'sma5': None,
            'sma10': None,
            'sma20': None,
            'timestamp': datetime.now().isoformat(),
            'error': True
        }

def get_sector_for_ticker(ticker: str) -> str:
    """
    Get sector name for a given ticker.

    Delegates to risk_management module for consistency.
    """
    from src.core.risk_management import get_sector_for_ticker as risk_get_sector
    return risk_get_sector(ticker)

# Watchlist management endpoints
@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def manage_watchlist():
    """Manage watchlist - GET, POST, DELETE"""
    import json

    WATCHLIST_FILE = 'watchlist.json'

    def load_watchlist():
        """Load watchlist from file"""
        try:
            if os.path.exists(WATCHLIST_FILE):
                with open(WATCHLIST_FILE, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading watchlist: {e}")
            return []

    def save_watchlist(watchlist):
        """Save watchlist to file"""
        try:
            with open(WATCHLIST_FILE, 'w') as f:
                json.dump(watchlist, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving watchlist: {e}")
            return False

    if request.method == 'GET':
        """Get current watchlist with stock data"""
        watchlist = load_watchlist()
        watchlist_data = []

        for ticker in watchlist:
            try:
                data = chart_service.get_stock_data(ticker, '1d')
                if data and data.get('info'):
                    watchlist_data.append({
                        'ticker': ticker,
                        'name': data['info'].get('name', ticker),
                        'price': data['info'].get('current_price', 0),
                        'change': data['info'].get('change', 0),
                        'change_percent': data['info'].get('change_percent', 0),
                        'volume': data['info'].get('volume', 0)
                    })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                # Add ticker with placeholder data if fetch fails
                watchlist_data.append({
                    'ticker': ticker,
                    'name': ticker,
                    'price': 0,
                    'change': 0,
                    'change_percent': 0,
                    'volume': 0,
                    'error': True
                })

        return jsonify({
            'watchlist': watchlist_data,
            'count': len(watchlist_data)
        })

    elif request.method == 'POST':
        """Add ticker to watchlist"""
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        watchlist = load_watchlist()

        if ticker not in watchlist:
            watchlist.append(ticker)
            if save_watchlist(watchlist):
                return jsonify({
                    'success': True,
                    'message': f'{ticker} added to watchlist',
                    'watchlist': watchlist
                })
            else:
                return jsonify({'error': 'Failed to save watchlist'}), 500
        else:
            return jsonify({'message': f'{ticker} already in watchlist'})

    elif request.method == 'DELETE':
        """Remove ticker from watchlist"""
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()

        if not ticker:
            return jsonify({'error': 'Ticker is required'}), 400

        watchlist = load_watchlist()

        if ticker in watchlist:
            watchlist.remove(ticker)
            if save_watchlist(watchlist):
                return jsonify({
                    'success': True,
                    'message': f'{ticker} removed from watchlist',
                    'watchlist': watchlist
                })
            else:
                return jsonify({'error': 'Failed to save watchlist'}), 500
        else:
            return jsonify({'error': f'{ticker} not in watchlist'}), 404

@app.route('/api/watchlist/search')
def search_watchlist_stocks():
    """Search for stocks to add to watchlist"""
    query = request.args.get('q', '').strip().upper()

    if not query or len(query) < 2:
        return jsonify({'error': 'Query must be at least 2 characters'}), 400

    try:
        # Search in both WIG30 and WIG20 tickers
        all_tickers = WIG30_TICKERS + WIG20_TICKERS
        matches = []

        for ticker in all_tickers:
            if query in ticker:
                matches.append(ticker)

        # Get basic data for matches
        results = []
        for ticker in matches[:10]:  # Limit to 10 results
            try:
                data = chart_service.get_stock_data(ticker, '1d')
                if data and data.get('info'):
                    results.append({
                        'ticker': ticker,
                        'name': data['info'].get('name', ticker),
                        'price': data['info'].get('current_price', 0),
                        'change': data['info'].get('change', 0),
                        'change_percent': data['info'].get('change_percent', 0)
                    })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'name': ticker,
                    'price': 0,
                    'change': 0,
                    'change_percent': 0,
                    'error': True
                })

        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Professional Charts API endpoints
@app.route('/api/charts/<ticker>/<chart_type>')
def get_professional_chart(ticker, chart_type):
    """Get professional chart configuration for a stock"""
    from src.utils.advanced_charts import generate_candlestick_chart, generate_rsi_chart, generate_macd_chart, generate_volume_chart
    import json
    import os

    try:
        # Get stock data
        data = chart_service.get_stock_data(ticker, '1y')
        if not data or not data.get('data'):
            return jsonify({'error': f'No data available for {ticker}'}), 404

        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        df.index = pd.to_datetime(df.index)

        # Generate professional chart based on type
        chart_config = None

        if chart_type == 'candlestick':
            chart_config = generate_candlestick_chart(
                ticker, data.get('info', {}).get('name', ticker), df
            )
        elif chart_type == 'rsi':
            chart_config = generate_rsi_chart(ticker, df)
        elif chart_type == 'macd':
            chart_config = generate_macd_chart(ticker, df)
        elif chart_type == 'volume':
            chart_config = generate_volume_chart(ticker, df)
        else:
            return jsonify({'error': f'Invalid chart type: {chart_type}'}), 400

        if chart_config:
            return safe_jsonify(chart_config)
        else:
            return jsonify({'error': f'Failed to generate {chart_type} chart'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/<ticker>')
def get_all_charts(ticker):
    """Get all professional charts for a stock"""
    try:
        # Get stock data
        data = chart_service.get_stock_data(ticker, '1y')
        if not data or not data.get('data'):
            return jsonify({'error': f'No data available for {ticker}'}), 404

        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        df.index = pd.to_datetime(df.index)

        # Get technical indicators
        chart_data = chart_service.get_chart_data(ticker, '1y', ['SMA_20', 'RSI_14', 'MACD'])

        # Generate all professional charts
        from src.utils.advanced_charts import generate_comprehensive_chart_package
        charts = generate_comprehensive_chart_package(
            ticker,
            data.get('info', {}).get('name', ticker),
            df,
            chart_data.get('indicators', {}).get('SMA_20', {}).get('values', [None, None, None, None])[-1] if chart_data.get('indicators', {}).get('SMA_20') else None,
            chart_data.get('indicators', {}).get('SMA_10', {}).get('values', [None, None, None, None])[-1] if chart_data.get('indicators', {}).get('SMA_10') else None,
            chart_data.get('indicators', {}).get('SMA_5', {}).get('values', [None, None, None, None])[-1] if chart_data.get('indicators', {}).get('SMA_5') else None,
            chart_data.get('trend', 'Boczny')
        )

        return safe_jsonify({
            'ticker': ticker,
            'company_name': data.get('info', {}).get('name', ticker),
            'charts': charts,
            'chart_types': ['candlestick', 'rsi', 'macd', 'volume']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# System configuration for radar-wig.pl
@app.route('/api/system/config')
def get_system_config():
    """Get system configuration for frontend"""
    return safe_jsonify({
        'site_name': 'GPW Smart Analyzer',
        'domain': 'radar-wig.pl',
        'version': '2.0.0',
        'features': {
            'professional_charts': True,
            'watchlist': True,
            'realtime_data': True,
            'technical_indicators': ['SMA', 'RSI', 'MACD', 'Bollinger Bands', 'ADX'],
            'supported_indices': ['WIG30', 'WIG20']
        },
        'theme': {
            'mode': 'dark',
            'primary_color': '#1e1e1e',
            'style': 'tradingview'
        }
    })

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

        # Track previous prices for conditional updates
        previous_prices = {}

        while app.monitoring_active:
            try:
                # Check if any ticker is being monitored
                if hasattr(app, 'ticker_subscriptions'):
                    batch_updates = []
                    active_tickers = [t for t, active in app.ticker_subscriptions.items() if active]

                    for ticker in active_tickers:
                        # Get enriched stock data
                        enriched_data = get_enriched_stock_data(ticker)

                        if 'error' not in enriched_data:
                            # Check if this is the first update or if change is significant (>0.5%)
                            current_price = enriched_data['price']
                            change_percent = enriched_data['change_percent']

                            should_update = (
                                ticker not in previous_prices or  # First update
                                abs(change_percent) > 0.5 or  # Significant change
                                enriched_data.get('trend') != 'sideways'  # Trend change
                            )

                            if should_update:
                                batch_updates.append(enriched_data)
                                previous_prices[ticker] = current_price
                                print(f"Batch update for {ticker}: {current_price:.2f} ({change_percent:+.2f}%)")

                        # Small delay between tickers to avoid rate limiting
                        time.sleep(0.1)

                    # Emit batch update if we have any
                    if batch_updates:
                        socketio.emit('price_update', {
                            'updates': batch_updates,
                            'timestamp': datetime.now().isoformat()
                        })  # Broadcast to all (no room parameter)
                        print(f"Emitted batch update with {len(batch_updates)} stocks")

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

    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)