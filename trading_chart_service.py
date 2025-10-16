#!/usr/bin/env python3
"""
Advanced Chart Service for TradingView-style charts with technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from flask import jsonify
import ta

class TradingChartService:
    def __init__(self):
        self.indicators = {
            'SMA': self.calculate_sma,
            'EMA': self.calculate_ema,
            'RSI': self.calculate_rsi,
            'MACD': self.calculate_macd,
            'BB': self.calculate_bollinger_bands,
            'VWAP': self.calculate_vwap,
            'ATR': self.calculate_atr,
            'Volume': self.get_volume_data,
            'ADX': self.calculate_adx_indicator,
            'DI': self.calculate_directional_indicators,
            'Market_Regime': self.calculate_market_regime
        }

    def get_stock_data(self, ticker, period='1y', interval='1d'):
        """Get historical stock data with technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                return None

            # Calculate technical indicators
            enhanced_data = self.calculate_all_indicators(data)

            return {
                'ticker': ticker,
                'data': enhanced_data.to_dict('records'),
                'info': {
                    'name': stock.info.get('longName', ticker),
                    'currency': stock.info.get('currency', 'PLN'),
                    'current_price': float(stock.info.get('currentPrice', data['Close'].iloc[-1])),
                    'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
                    'change_percent': float((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100)
                }
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        df = data.copy()

        try:
            # Basic Moving Averages
            df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            df['SMA_200'] = self.calculate_sma(df['Close'], 200)

            # Exponential Moving Averages
            df['EMA_12'] = self.calculate_ema(df['Close'], 12)
            df['EMA_26'] = self.calculate_ema(df['Close'], 26)

            # RSI
            df['RSI_14'] = self.calculate_rsi(df['Close'], 14)

            # MACD
            macd_data = self.calculate_macd(df['Close'])
            df['MACD'] = macd_data['MACD']
            df['MACD_Signal'] = macd_data['Signal']
            df['MACD_Histogram'] = macd_data['Histogram']

            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(df['Close'], 20, 2)
            df['BB_Upper'] = bb_data['Upper']
            df['BB_Middle'] = bb_data['Middle']
            df['BB_Lower'] = bb_data['Lower']

            # Volume Profile (simplified VWAP) - with error handling
            try:
                vwap_data = self.calculate_vwap(df)
                df['VWAP'] = vwap_data
            except Exception as e:
                print(f"VWAP calculation failed: {e}")
                df['VWAP'] = df['Close']  # Fallback to close price

            # ATR
            df['ATR_14'] = self.calculate_atr(df, 14)

            # ADX and Directional Indicators
            adx_data = self.calculate_adx_indicator(df)
            df['ADX_14'] = adx_data['ADX']
            df['DI_Plus'] = adx_data['DI_Plus']
            df['DI_Minus'] = adx_data['DI_Minus']

            # Market Regime Classification
            regime_data = self.calculate_market_regime(df)
            df['Market_Regime'] = regime_data['regime']
            df['Regime_Strength'] = regime_data['strength']

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Add basic indicators if calculation fails
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['RSI_14'] = None
            df['ADX_14'] = None
            df['DI_Plus'] = None
            df['DI_Minus'] = None
            df['Market_Regime'] = 'UNKNOWN'
            df['Regime_Strength'] = 0

        return df

    def calculate_sma(self, prices, period):
        """Simple Moving Average"""
        return ta.trend.sma_indicator(prices, window=period)

    def calculate_ema(self, prices, period):
        """Exponential Moving Average"""
        return ta.trend.ema_indicator(prices, window=period)

    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index"""
        return ta.momentum.rsi(prices, window=period)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        macd = ta.trend.MACD(prices, window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            'MACD': macd.macd(),
            'Signal': macd.macd_signal(),
            'Histogram': macd.macd_diff()
        }

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands"""
        bb = ta.volatility.BollingerBands(prices, window=period, window_dev=std_dev)
        return {
            'Upper': bb.bollinger_hband(),
            'Middle': bb.bollinger_mavg(),
            'Lower': bb.bollinger_lband()
        }

    def calculate_vwap(self, data):
        """Volume Weighted Average Price"""
        try:
            return ta.volume.volume_weighted_average_price(
                data['High'], data['Low'], data['Close'], data['Volume']
            )
        except Exception as e:
            print(f"VWAP calculation error: {e}")
            # Fallback: simple price-based VWAP approximation
            return data['Close'].rolling(window=20, min_periods=1).mean()

    def calculate_atr(self, data, period=14):
        """Average True Range"""
        return ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=period)

    def get_volume_data(self, data):
        """Volume data"""
        return data['Volume']

    def format_chart_data(self, data, indicators=['SMA_20', 'RSI_14', 'MACD']):
        """Format data for Chart.js consumption"""
        if not data or 'data' not in data:
            return None

        df = pd.DataFrame(data['data'])
        df.index = pd.to_datetime(df.index)

        # Prepare candlestick data
        candlestick_data = []
        for idx, row in df.iterrows():
            candlestick_data.append({
                'x': int(idx.timestamp() * 1000),  # Convert to milliseconds for Chart.js
                'o': float(row['Open']),
                'h': float(row['High']),
                'l': float(row['Low']),
                'c': float(row['Close']),
                'v': float(row['Volume'])
            })

        # Prepare indicator data including ADX and DI series
        indicator_data = {}
        for indicator in indicators:
            if indicator in df.columns:
                indicator_data[indicator] = [
                    {
                        'x': int(idx.timestamp() * 1000),
                        'y': float(row[indicator]) if pd.notna(row[indicator]) else None
                    }
                    for idx, row in df.iterrows()
                ]

        # Handle special case for DI indicators - if 'DI' is requested, include both DI_Plus and DI_Minus
        if 'DI' in indicators:
            if 'DI_Plus' in df.columns:
                indicator_data['DI_Plus'] = [
                    {
                        'x': int(idx.timestamp() * 1000),
                        'y': float(row['DI_Plus']) if pd.notna(row['DI_Plus']) else None
                    }
                    for idx, row in df.iterrows()
                ]
            if 'DI_Minus' in df.columns:
                indicator_data['DI_Minus'] = [
                    {
                        'x': int(idx.timestamp() * 1000),
                        'y': float(row['DI_Minus']) if pd.notna(row['DI_Minus']) else None
                    }
                    for idx, row in df.iterrows()
                ]

        # Handle Market_Regime with numeric encoding for charting
        if 'Market_Regime' in indicators and 'Market_Regime' in df.columns:
            # Convert regime strings to numeric values for charting
            regime_mapping = {
                'CONSOLIDATION': 1,
                'WEAK_TREND': 2,
                'STRONG_TREND': 3,
                'EMERGING_STRONG_TREND': 4,
                'VERY_STRONG_TREND': 5,
                'UNKNOWN': 0,
                'INSUFFICIENT_DATA': 0
            }
            indicator_data['Market_Regime'] = [
                {
                    'x': int(idx.timestamp() * 1000),
                    'y': regime_mapping.get(row['Market_Regime'], 0)
                }
                for idx, row in df.iterrows()
            ]

        return {
            'info': data['info'],
            'candlestick': candlestick_data,
            'indicators': indicator_data,
            'volume': [
                {
                    'x': int(idx.timestamp() * 1000),
                    'y': float(row['Volume'])
                }
                for idx, row in df.iterrows()
            ]
        }

    def get_multiple_stocks_data(self, tickers, period='1y'):
        """Get data for multiple stocks for comparison"""
        results = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period)
            if data:
                # Get only closing prices for comparison
                df = pd.DataFrame(data['data'])
                results[ticker] = {
                    'name': data['info']['name'],
                    'prices': [
                        {
                            'x': idx.timestamp() * 1000,
                            'y': float(row['Close'])
                        }
                        for idx, row in df.iterrows()
                    ]
                }
        return results

    def calculate_adx_indicator(self, df, period=14):
        """Calculate ADX (Average Directional Index) using ta library."""
        try:
            if len(df) < period:
                return {
                    'ADX': None,
                    'DI_Plus': None,
                    'DI_Minus': None
                }

            adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period)
            return {
                'ADX': adx_indicator.adx(),
                'DI_Plus': adx_indicator.adx_pos(),
                'DI_Minus': adx_indicator.adx_neg()
            }
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return {
                'ADX': None,
                'DI_Plus': None,
                'DI_Minus': None
            }

    def calculate_directional_indicators(self, df, period=14):
        """Calculate Directional Indicators (+DI, -DI)."""
        return self.calculate_adx_indicator(df, period)

    def calculate_market_regime(self, df):
        """Calculate market regime based on ADX values."""
        try:
            adx_data = self.calculate_adx_indicator(df)
            adx_value = adx_data['ADX']

            if adx_value is None or pd.isna(adx_value):
                return {
                    'regime': 'INSUFFICIENT_DATA',
                    'strength': 0
                }

            # Classify market regime based on ADX thresholds
            if adx_value < 20:
                regime = 'CONSOLIDATION'
                strength = adx_value / 20  # 0-1 scale
            elif adx_value < 25:
                regime = 'WEAK_TREND'
                strength = (adx_value - 20) / 5  # 0-1 scale
            elif adx_value < 40:
                regime = 'STRONG_TREND'
                strength = (adx_value - 25) / 15  # 0-1 scale
            elif adx_value < 60:
                regime = 'EMERGING_STRONG_TREND'
                strength = (adx_value - 40) / 20  # 0-1 scale
            else:
                regime = 'VERY_STRONG_TREND'
                strength = min(1.0, (adx_value - 60) / 40 + 1)  # 1-2 scale but capped

            return {
                'regime': regime,
                'strength': strength
            }

        except Exception as e:
            print(f"Error calculating market regime: {e}")
            return {
                'regime': 'UNKNOWN',
                'strength': 0
            }


# Global chart service instance
chart_service = TradingChartService()