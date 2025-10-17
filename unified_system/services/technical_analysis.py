# -*- coding: utf-8 -*-
"""
Technical Analysis Service - Zunifikowana analiza techniczna
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class TechnicalAnalysisService:
    """Service for technical analysis calculations"""

    def __init__(self):
        pass

    def calculate_moving_averages(self, prices: pd.Series, periods: List[int]) -> Dict[str, float]:
        """Calculate moving averages for given periods"""
        ma_data = {}
        for period in periods:
            if len(prices) >= period:
                ma_data[f'ma_{period}'] = prices.rolling(period).mean().iloc[-1]
        return ma_data

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return {}

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {}

        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        return {
            'upper': (sma + std * std_dev).iloc[-1],
            'middle': sma.iloc[-1],
            'lower': (sma - std * std_dev).iloc[-1],
            'width': ((sma + std * std_dev).iloc[-1] - (sma - std * std_dev).iloc[-1]) / sma.iloc[-1] * 100
        }

    def analyze_trend(self, prices: pd.Series, ma_data: Dict) -> Dict:
        """Analyze price trend using moving averages"""
        current_price = prices.iloc[-1]

        # Get available MAs
        ma5 = ma_data.get('ma_5')
        ma10 = ma_data.get('ma_10')
        ma20 = ma_data.get('ma_20')

        if not all([ma5, ma10, ma20]):
            return {'direction': 'unknown', 'strength': 0}

        # Trend determination
        if current_price > ma5 > ma10 > ma20:
            direction = 'uptrend'
            strength = 0.8
        elif current_price < ma5 < ma10 < ma20:
            direction = 'downtrend'
            strength = 0.8
        else:
            direction = 'sideways'
            strength = 0.5

        # Calculate trend strength based on MA spread
        ma_spread = (ma5 - ma20) / ma20 * 100
        if abs(ma_spread) > 2:
            strength = min(strength + 0.2, 1.0)

        return {
            'direction': direction,
            'strength': strength,
            'current_vs_ma20': (current_price - ma20) / ma20 * 100,
            'ma_spread': ma_spread
        }

    def analyze_volume(self, volumes: pd.Series, ma_period: int = 20) -> Dict:
        """Analyze volume patterns"""
        if volumes.empty:
            return {}

        recent_volume = volumes.iloc[-5:].mean()
        avg_volume = volumes.rolling(ma_period).mean().iloc[-1] if len(volumes) >= ma_period else volumes.mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # Volume trend classification
        if volume_ratio > 1.5:
            trend = 'high'
        elif volume_ratio > 1.2:
            trend = 'increasing'
        elif volume_ratio < 0.8:
            trend = 'decreasing'
        else:
            trend = 'normal'

        return {
            'recent_avg': recent_volume,
            'overall_avg': avg_volume,
            'ratio': volume_ratio,
            'trend': trend,
            'volume_spike': volume_ratio > 2.0
        }

    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < period:
            return 0

        returns = prices.pct_change().dropna()
        volatility = returns.rolling(period).std().iloc[-1] * np.sqrt(252)  # Annualized

        return volatility if not pd.isna(volatility) else 0

    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0

        momentum = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1] * 100
        return momentum if not pd.isna(momentum) else 0

    def detect_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict:
        """Detect potential support and resistance levels"""
        if len(prices) < window * 2:
            return {}

        recent_prices = prices.tail(window)

        # Find local maxima and minima
        highs = []
        lows = []

        for i in range(1, len(recent_prices) - 1):
            if recent_prices.iloc[i] > recent_prices.iloc[i-1] and recent_prices.iloc[i] > recent_prices.iloc[i+1]:
                highs.append(recent_prices.iloc[i])
            elif recent_prices.iloc[i] < recent_prices.iloc[i-1] and recent_prices.iloc[i] < recent_prices.iloc[i+1]:
                lows.append(recent_prices.iloc[i])

        current_price = prices.iloc[-1]

        # Find nearest resistance and support
        resistance_levels = sorted([h for h in highs if h > current_price])
        support_levels = sorted([l for l in lows if l < current_price], reverse=True)

        return {
            'resistance': resistance_levels[:3] if resistance_levels else [],
            'support': support_levels[:3] if support_levels else [],
            'nearest_resistance': resistance_levels[0] if resistance_levels else None,
            'nearest_support': support_levels[0] if support_levels else None,
            'current_position': 'between_levels' if resistance_levels and support_levels else 'unclear'
        }

    def complete_technical_analysis(self, prices: pd.Series, volumes: pd.Series = None) -> Dict:
        """Perform complete technical analysis"""
        if prices.empty:
            return {}

        # Configuration
        config = {
            'ma_periods': [5, 10, 20, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2
        }

        analysis = {}

        # Moving averages
        analysis['moving_averages'] = self.calculate_moving_averages(prices, config['ma_periods'])

        # RSI
        analysis['rsi'] = self.calculate_rsi(prices, config['rsi_period'])

        # MACD
        analysis['macd'] = self.calculate_macd(
            prices,
            config['macd_fast'],
            config['macd_slow'],
            config['macd_signal']
        )

        # Bollinger Bands
        analysis['bollinger_bands'] = self.calculate_bollinger_bands(
            prices,
            config['bollinger_period'],
            config['bollinger_std']
        )

        # Trend analysis
        analysis['trend'] = self.analyze_trend(prices, analysis['moving_averages'])

        # Volume analysis
        if volumes is not None and not volumes.empty:
            analysis['volume'] = self.analyze_volume(volumes)

        # Volatility
        analysis['volatility'] = self.calculate_volatility(prices)

        # Momentum
        analysis['momentum'] = self.calculate_momentum(prices)

        # Support/Resistance
        analysis['support_resistance'] = self.detect_support_resistance(prices)

        # Trading signals
        analysis['signals'] = self._generate_trading_signals(analysis)

        return analysis

    def _generate_trading_signals(self, analysis: Dict) -> Dict:
        """Generate trading signals based on technical analysis"""
        signals = {
            'overall': 'neutral',
            'strength': 0,
            'reasons': []
        }

        # Trend-based signal
        trend = analysis.get('trend', {})
        if trend.get('direction') == 'uptrend':
            signals['strength'] += 0.3
            signals['reasons'].append('Uptrend confirmed')
        elif trend.get('direction') == 'downtrend':
            signals['strength'] -= 0.3
            signals['reasons'].append('Downtrend confirmed')

        # RSI signal
        rsi = analysis.get('rsi')
        if rsi:
            if rsi < 30:
                signals['strength'] += 0.2
                signals['reasons'].append('RSI oversold')
            elif rsi > 70:
                signals['strength'] -= 0.2
                signals['reasons'].append('RSI overbought')

        # MACD signal
        macd = analysis.get('macd', {})
        if macd.get('histogram', 0) > 0:
            signals['strength'] += 0.15
            signals['reasons'].append('MACD bullish')
        elif macd.get('histogram', 0) < 0:
            signals['strength'] -= 0.15
            signals['reasons'].append('MACD bearish')

        # Volume confirmation
        volume = analysis.get('volume', {})
        if volume.get('trend') == 'high' and signals['strength'] > 0:
            signals['strength'] += 0.1
            signals['reasons'].append('High volume confirmation')

        # Determine overall signal
        if signals['strength'] > 0.3:
            signals['overall'] = 'bullish'
        elif signals['strength'] < -0.3:
            signals['overall'] = 'bearish'
        else:
            signals['overall'] = 'neutral'

        return signals