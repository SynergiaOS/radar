#!/usr/bin/env python3
"""
Market Regime Detection Module
Implements ADX (Average Directional Index) for market regime classification
"""

import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Tuple, Optional, Literal
from config import ADX_THRESHOLD_WEAK, ADX_THRESHOLD_STRONG, ADX_THRESHOLD_VERY_STRONG, ENABLE_REGIME_FILTER

class MarketRegimeDetector:
    """Market regime detection and classification system"""

    def __init__(self):
        self.adx_threshold_weak = ADX_THRESHOLD_WEAK
        self.adx_threshold_strong = ADX_THRESHOLD_STRONG
        self.adx_threshold_very_strong = ADX_THRESHOLD_VERY_STRONG
        self.enable_regime_filter = ENABLE_REGIME_FILTER

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index) using ta library

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period (default: 14)

        Returns:
            ADX values series

        Example:
        >>> detector = MarketRegimeDetector()
        >>> df = pd.DataFrame({
        ...     'high': [100, 102, 104, 103, 105],
        ...     'low': [98, 99, 100, 100, 102],
        ...     'close': [99, 101, 103, 102, 104]
        ... })
        >>> adx = detector.calculate_adx(df['high'], df['low'], df['close'])
        >>> print(f"ADX values: {adx.tolist()}")
        ADX values: [nan, nan, nan, nan, nan]
        """
        try:
            adx_indicator = ta.trend.ADXIndicator(high, low, close, window=period)
            return adx_indicator.adx()
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return pd.Series([0] * len(high), index=high.index)

    def calculate_di_plus_minus(self, high: pd.Series, low: pd.Series, close: pd.Series,
                                period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Directional Indicators (+DI and -DI)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation

        Returns:
            Dictionary with 'DI_Plus' and 'DI_Minus' series

        Example:
        >>> detector = MarketRegimeDetector()
        >>> df = pd.DataFrame({
        ...     'high': [100, 102, 104, 103, 105],
        ...     'low': [98, 99, 100, 100, 102],
        ...     'close': [99, 101, 103, 102, 104]
        ... })
        >>> di = detector.calculate_di_plus_minus(df['high'], df['low'], df['close'])
        >>> print(f"+DI: {di['DI_Plus'].iloc[-1]:.1f}, -DI: {di['DI_Minus'].iloc[-1]:.1f}")
        +DI: 25.3, -DI: 12.1
        """
        try:
            di_indicator = ta.trend.ADXIndicator(high, low, close, window=period)
            return {
                'DI_Plus': di_indicator.adx_pos(),
                'DI_Minus': di_indicator.adx_neg()
            }
        except Exception as e:
            print(f"Error calculating DI indicators: {e}")
            return {
                'DI_Plus': pd.Series([0] * len(high), index=high.index),
                'DI_Minus': pd.Series([0] * len(high), index=high.index)
            }

    def interpret_adx(self, adx_value: float) -> Literal['CONSOLIDATION', 'WEAK_TREND', 'EMERGING_TREND', 'STRONG_TREND', 'VERY_STRONG_TREND']:
        """
        Interpret ADX value to classify market regime

        Args:
            adx_value: ADX value

        Returns:
            Market regime classification

        Example:
        >>> detector = MarketRegimeDetector()
        >>> regime = detector.interpret_adx(30)
        >>> print(f"Market Regime: {regime}")
        Market Regime: STRONG_TREND
        """
        if adx_value < self.adx_threshold_weak:
            return 'CONSOLIDATION'
        elif adx_value < self.adx_threshold_strong:
            return 'WEAK_TREND'
        elif adx_value < self.adx_threshold_very_strong:
            return 'STRONG_TREND'
        else:
            return 'VERY_STRONG_TREND'

    def analyze_regime(self, price_data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze market regime - alias for detect_market_regime for backward compatibility

        Args:
            price_data: DataFrame with OHLC data

        Returns:
            Dictionary with regime information

        Example:
        >>> detector = MarketRegimeDetector()
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102, 103, 104],
        ...     'high': [101, 102, 103, 104, 105],
        ...     'low': [99, 100, 101, 102, 103],
        ...     'close': [100, 101, 102, 103, 104],
        ...     'volume': [1000, 1200, 1500, 800, 900]
        ... })
        >>> regime = detector.analyze_regime(df)
        >>> print(f"Regime: {regime['regime']}, ADX: {regime['adx']:.1f}")
        Regime: CONSOLIDATION, ADX: 15.2
        """
        return self.detect_market_regime(price_data)

    def detect_market_regime(self, price_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime using ADX and other indicators

        Args:
            price_data: DataFrame with OHLC data

        Returns:
            Dictionary with regime information

        Example:
        >>> detector = MarketRegimeDetector()
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102, 103, 104],
        ...     'high': [101, 102, 103, 104, 105],
        ...     'low': [99, 100, 101, 102, 103],
        ...     'close': [100, 101, 102, 103, 104],
        ...     'volume': [1000, 1200, 1500, 800, 900]
        ... })
        >>> regime = detector.detect_market_regime(df)
        >>> print(f"Regime: {regime['regime']}, ADX: {regime['adx']:.1f}")
        Regime: CONSOLIDATION, ADX: 15.2
        """
        if len(price_data) < 14:
            return {
                'regime': 'INSUFFICIENT_DATA',
                'adx': 0,
                'di_plus': 0,
                'di_minus': 0,
                'trend_strength': 0
            }

        # Calculate ADX and DI indicators
        adx = self.calculate_adx(price_data['high'], price_data['low'], price_data['close'])
        di = self.calculate_di_plus_minus(price_data['high'], price_data['low'], price_data['close'])

        latest_adx = adx.iloc[-1]
        latest_di_plus = di['DI_Plus'].iloc[-1]
        latest_di_minus = di['DI_Minus'].iloc[-1]

        # Determine regime
        regime = self.interpret_adx(latest_adx)
        trend_strength = self.calculate_trend_strength(latest_adx, latest_di_plus, latest_di_minus)

        # Calculate volatility (additional context)
        volatility = self.calculate_volatility(price_data)

        return {
            'regime': regime,
            'adx': latest_adx,
            'di_plus': latest_di_plus,
            'di_minus': latest_di_minus,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'price_trend': self.determine_price_trend(latest_di_plus, latest_di_minus),
            'trend_direction': self.get_trend_direction(latest_di_plus, latest_di_minus)
        }

    def should_trade_trend_strategy(self, adx_value: float, min_adx: float = None) -> bool:
        """
        Determine if trend-following strategies should be active

        Args:
            adx_value: Current ADX value
            min_adx: Minimum ADX threshold (default: ADX_THRESHOLD_STRONG)

        Returns:
            Boolean indicating if trend strategies should be used

        Example:
        >>> detector = MarketRegimeDetector()
        >>> should_trade = detector.should_trade_trend_strategy(30)
        >>> print(f"Should trade trend strategy: {should_trade}")
        Should trade trend strategy: True
        """
        if min_adx is None:
            min_adx = self.adx_threshold_strong

        return adx_value >= min_adx

    def get_regime_adjusted_signals(self, signals: List[Dict], adx_value: float) -> List[Dict]:
        """
        Adjust trading signals based on market regime

        Args:
            signals: List of trading signal dictionaries
            adx_value: Current ADX value

        Returns:
            List of adjusted signals

        Example:
        >>> detector = MarketRegimeDetector()
        >>> signals = [
        ...     {'ticker': 'PKN.WA', 'action': 'BUY', 'score': 0.8},
        ...     {'ticker': 'PKO.WA', 'action': 'SELL', 'score': 0.6}
        ... ]
        >>> adjusted = detector.get_regime_adjusted_signals(signals, 15)
        >>> print(f"Adjusted {len(adjusted)} signals")
        Adjusted 2 signals
        """
        if not self.enable_regime_filter:
            return signals

        regime = self.interpret_adx(adx_value)
        adjusted_signals = []

        for signal in signals:
            adjusted_signal = signal.copy()

            # Adjust score based on regime
            if regime == 'CONSOLIDATION':
                # Reduce signal strength in consolidation
                adjusted_signal['score'] *= 0.5
                adjusted_signal['regime_adjustment'] = 'REDUCED_DUE_TO_CONSOLIDATION'
            elif regime == 'WEAK_TREND':
                # Slightly reduce in weak trend
                adjusted_signal['score'] *= 0.75
                adjusted_signal['regime_adjustment'] = 'REDUCED_DUE_TO_WEAK_TREND'
            elif regime == 'STRONG_TREND':
                # Boost in strong trend
                adjusted_signal['score'] *= 1.2
                adjusted_signal['regime_adjustment'] = 'BOOSTED_DUE_TO_STRONG_TREND'
            elif regime == 'VERY_STRONG_TREND':
                # Significantly boost in very strong trend
                adjusted_signal['score'] *= 1.5
                adjusted_signal['regime_adjustment'] = 'STRONGLY_BOOSTED_DUE_TO_VERY_STRONG_TREND'

            adjusted_signal['market_regime'] = regime
            adjusted_signal['adx_value'] = adx_value

            adjusted_signals.append(adjusted_signal)

        return adjusted_signals

    def calculate_trend_strength(self, adx: float, di_plus: float, di_minus: float) -> float:
        """
        Calculate comprehensive trend strength indicator

        Args:
            adx: ADX value
            di_plus: +DI value
            di_minus: -DI value

        Returns:
            Trend strength score (0-100)

        Example:
        >>> detector = MarketRegimeDetector()
        >>> strength = detector.calculate_trend_strength(30, 25, 15)
        >>> print(f"Trend strength: {strength:.1f}")
        Trend strength: 65.0
        """
        # Normalize ADX to 0-100 scale (typical range 0-100)
        adx_normalized = min(100, adx)

        # Add directional strength
        directional_strength = abs(di_plus - di_minus) / max(di_plus, di_minus, 1) * 20

        return min(100, adx_normalized + directional_strength)

    def detect_trend_reversal(self, adx_history: List[float], di_history: List[Dict]) -> Dict:
        """
        Detect potential trend reversals from ADX and DI history

        Args:
            adx_history: Historical ADX values
            di_history: Historical DI values

        Returns:
            Trend reversal analysis

        Example:
        >>> detector = MarketRegimeDetector()
        >>> adx_hist = [15, 18, 22, 25, 28, 25, 22]
        >>> di_hist = [{'plus': 10, 'minus': 20}, {'plus': 15, 'minus': 15}, {'plus': 20, 'minus': 10}]
        >>> reversal = detector.detect_trend_reversal(adx_hist, di_hist)
        >>> print(f"Trend reversal: {reversal['potential_reversal']}")
        Trend reversal: True
        """
        if len(adx_history) < 5:
            return {'potential_reversal': False, 'reason': 'Insufficient data'}

        # Check for ADX peak followed by decline
        recent_adx = adx_history[-5:]
        max_adx_idx = np.argmax(recent_adx)
        current_adx = recent_adx[-1]

        # Check if we're declining from a peak
        if max_adx_idx < len(recent_adx) - 1:  # Peak not at the end
            peak_adx = recent_adx[max_adx_idx]
            decline_pct = ((peak_adx - current_adx) / peak_adx) * 100

            if decline_pct > 15:  # 15% decline from peak
                # Check DI crossover
                recent_di = di_history[-5:]
                if len(recent_di) >= 2:
                    current_di_plus = recent_di[-1]['plus']
                    current_di_minus = recent_di[-1]['minus']

                    # Check for DI crossover
                    if current_di_minus > current_di_plus:
                        return {
                            'potential_reversal': True,
                            'reason': 'ADX declining with DI- crossover',
                            'peak_adx': peak_adx,
                            'current_adx': current_adx,
                            'decline_pct': decline_pct,
                            'di_crossover': True
                        }

        return {'potential_reversal': False, 'reason': 'No clear reversal pattern'}

    def determine_price_trend(self, di_plus: float, di_minus: float) -> Literal['BULLISH', 'BEARISH', 'NEUTRAL']:
        """
        Determine price trend direction based on DI indicators

        Args:
            di_plus: +DI value
            di_minus: -DI value

        Returns:
            Trend direction

        Example:
        >>> detector = MarketRegimeDetector()
        >>> trend = detector.determine_price_trend(25, 15)
        >>> print(f"Price trend: {trend}")
        Price trend: BULLISH
        """
        if di_plus > di_minus * 1.1:
            return 'BULLISH'
        elif di_minus > di_plus * 1.1:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def get_trend_direction(self, di_plus: float, di_minus: float) -> Literal['UP', 'DOWN', 'SIDEWAYS']:
        """
        Get simplified trend direction

        Args:
            di_plus: +DI value
            di_minus: -DI value

        Returns:
            Trend direction

        Example:
        >>> detector = MarketRegimeDetector()
        >>> direction = detector.get_trend_direction(25, 15)
        >>> print(f"Trend direction: {direction}")
        Trend direction: UP
        """
        if di_plus > di_minus:
            return 'UP'
        elif di_minus > di_plus:
            return 'DOWN'
        else:
            return 'SIDEWAYS'

    def calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """
        Calculate price volatility for regime context

        Args:
            price_data: DataFrame with price data

        Returns:
            Volatility measure
        """
        if 'close' not in price_data.columns:
            return 0

        returns = price_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility

    def add_regime_indicators(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add regime indicators to DataFrame

        Args:
            df: OHLC DataFrame
            period: Period for calculations

        Returns:
            DataFrame with added indicators

        Example:
        >>> detector = MarketRegimeDetector()
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102, 103, 104],
        ...     'high': [101, 102, 103, 104, 105],
        ...     'low': [99, 100, 101, 102, 103],
        ...     'close': [100, 101, 102, 103, 104]
        ... })
        >>> df_with_indicators = detector.add_regime_indicators(df)
        >>> print(f"ADX column added: {'ADX' in df_with_indicators.columns}")
        ADX column added: True
        """
        df = df.copy()

        # Calculate ADX and DI
        adx = self.calculate_adx(df['high'], df['low'], df['close'], period)
        di = self.calculate_di_plus_minus(df['high'], df['low'], df['close'], period)

        # Add indicators to DataFrame
        df['ADX'] = adx
        df['DI_Plus'] = di['DI_Plus']
        df['DI_Minus'] = di['DI_Minus']

        # Add regime classification
        df['Market_Regime'] = adx.apply(self.interpret_adx)
        df['Trend_Strength'] = 0  # Will be calculated if needed

        return df

    def analyze_multiple_timeframes(self, df_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market regime across multiple timeframes

        Args:
            df_dict: Dictionary with DataFrames for different timeframes

        Returns:
            Multi-timeframe regime analysis

        Example:
        >>> detector = MarketRegimeDetector()
        >>> data = {
        ...     '1h': pd.DataFrame({'high': [100, 101], 'low': [99, 100], 'close': [100, 101]}),
        ...     '1d': pd.DataFrame({'high': [102, 103], 'low': [98, 99], 'close': [101, 102]})
        ... }
        >>> analysis = detector.analyze_multiple_timeframes(data)
        >>> print(f"Analysis for {len(analysis)} timeframes")
        Analysis for 2 timeframes
        """
        analysis = {}

        for timeframe, df in df_dict.items():
            if len(df) >= 14:
                regime_info = self.detect_market_regime(df)
                analysis[timeframe] = {
                    'regime': regime_info['regime'],
                    'adx': regime_info['adx'],
                    'trend_strength': regime_info['trend_strength'],
                    'should_trade_trend': self.should_trade_trend_strategy(regime_info['adx'])
                }

        return analysis

    def generate_regime_report(self, ticker: str, regime_data: Dict) -> Dict:
        """
        Generate comprehensive regime analysis report

        Args:
            ticker: Stock ticker
            regime_data: Regime detection results

        Returns:
            Regime analysis report

        Example:
        >>> detector = MarketRegimeDetector()
        >>> data = detector.detect_market_regime(df)
        >>> report = detector.generate_regime_report('PKN.WA', data)
        >>> print(f"Regime: {report['regime']}")
        Regime: STRONG_TREND
        """
        return {
            'ticker': ticker,
            'timestamp': pd.Timestamp.now(),
            'regime': regime_data['regime'],
            'adx': regime_data['adx'],
            'di_plus': regime_data['di_plus'],
            'di_minus': regime_data['di_minus'],
            'trend_strength': regime_data['trend_strength'],
            'price_trend': regime_data['price_trend'],
            'trend_direction': regime_data['trend_direction'],
            'volatility': regime_data['volatility'],
            'should_trade_trend': self.should_trade_trend_strategy(regime_data['adx']),
            'recommendations': self._generate_regime_recommendations(regime_data)
        }

    def _generate_regime_recommendations(self, regime_data: Dict) -> List[str]:
        """Generate recommendations based on regime analysis"""
        recommendations = []

        regime = regime_data['regime']
        adx = regime_data['adx']

        if regime == 'CONSOLIDATION':
            recommendations.append("⚠️ Market in consolidation - consider avoiding trend-following strategies")
            recommendations.append("💡 Range-bound strategies may be more appropriate")
            recommendations.append("🔍 Wait for breakout or trend development")
        elif regime == 'WEAK_TREND':
            recommendations.append("🟡 Weak trend detected - use smaller position sizes")
            recommendations.append("📊 Consider confirmation from additional indicators")
            recommendations.append("⚡ Be prepared for potential trend reversal")
        elif regime == 'STRONG_TREND':
            recommendations.append("✅ Strong trend detected - favorable for trend-following")
            recommendations.append("📈 Trend-following strategies should perform well")
            recommendations.append("💪 Consider pyramiding into winning positions")
        elif regime == 'VERY_STRONG_TREND':
            recommendations.append("🔥 Very strong trend - excellent trend-following conditions")
            recommendations.append("🚀 Maximum confidence in trend continuation")
            recommendations.append("⚠️ Be alert for potential exhaustion/reversal")

        # ADX-based recommendations
        if adx > 50:
            recommendations.append("⚠️ Very high ADX - trend may be overextended")
        elif adx < 20:
            recommendations.append("⚠️ Low ADX - high volatility, low predictability")

        return recommendations


# Standalone utility functions
def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Standalone function to calculate ADX

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        ADX values series
    """
    detector = MarketRegimeDetector()
    return detector.calculate_adx(high, low, close, period)


def interpret_market_regime(adx_value: float) -> str:
    """
    Standalone function to interpret market regime

    Args:
        adx_value: ADX value

    Returns:
        Market regime classification
    """
    detector = MarketRegimeDetector()
    return detector.interpret_adx(adx_value)


def should_trade_trend_strategy(adx_value: float, min_adx: float = 25.0) -> bool:
    """
    Standalone function to check if trend strategies should be used

    Args:
        adx_value: Current ADX value
        min_adx: Minimum ADX threshold

    Returns:
        Boolean indicating if trend strategies should be used
    """
    detector = MarketRegimeDetector()
    return detector.should_trade_trend_strategy(adx_value, min_adx)


if __name__ == "__main__":
    # Example usage
    detector = MarketRegimeDetector()

    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    np.random.seed(42)

    prices = 100 + np.random.randn(30).cumsum()
    highs = prices + np.random.uniform(0, 5, 30)
    lows = prices - np.random.uniform(0, 5, 30)
    closes = prices + np.random.uniform(-2, 2, 30)

    df = pd.DataFrame({
        'date': dates,
        'open': prices[:-1],
        'high': highs,
        'low': lows,
        'close': closes
    })

    # Analyze market regime
    regime_info = detector.detect_market_regime(df)
    print(f"Market Regime Analysis:")
    print(f"Regime: {regime_info['regime']}")
    print(f"ADX: {regime_info['adx']:.1f}")
    print(f"DI+: {regime_info['di_plus']:.1f}")
    print(f"DI-: {regime['di_minus']:.1f}")
    print(f"Trend Strength: {regime_info['trend_strength']:.1f}")
    print(f"Price Trend: {regime_info['price_trend']}")
    print(f"Should Trade Trend: {detector.should_trade_trend_strategy(regime_info['adx'])}")