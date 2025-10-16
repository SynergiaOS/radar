#!/usr/bin/env python3
"""
Advanced Trading Signals System with Risk Management
Enhanced with Kelly Criterion, market regime detection, and comprehensive risk analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Import advanced risk management modules
from risk_management import RiskManager
from market_regime import MarketRegimeDetector
from ml_trading_system import AdvancedMLTradingSystem

class TradingSignalGenerator:
    """Advanced trading signals generator with comprehensive risk management."""

    def __init__(self):
        self.signals = []
        self.price_history = {}
        self.fundamentals = {}
        self.risk_levels = {
            'CONSERVATIVE': {'min_roe': 12, 'max_pe': 12, 'min_score': 75, 'max_risk_per_trade': 0.01},
            'MODERATE': {'min_roe': 10, 'max_pe': 15, 'min_score': 65, 'max_risk_per_trade': 0.02},
            'AGGRESSIVE': {'min_roe': 8, 'max_pe': 20, 'min_score': 55, 'max_risk_per_trade': 0.03}
        }

        # Enhanced risk management components
        self.risk_manager = RiskManager()
        self.market_regime_detector = MarketRegimeDetector()
        self.ml_system = AdvancedMLTradingSystem()
        self.portfolio_positions = []
        self.market_regimes = {}

        # Performance tracking
        self.signal_history = []
        self.execution_results = []

    def load_data(self, fundamentals_file='wig30_analysis_pe_threshold.csv', prices_file='data/current_prices.csv'):
        """Load fundamental and price data."""
        try:
            # Load fundamentals
            if fundamentals_file and os.path.exists(fundamentals_file):
                df_fund = pd.read_csv(fundamentals_file)
                for _, row in df_fund.iterrows():
                    self.fundamentals[row['ticker']] = {
                        'name': row['name'],
                        'roe': row.get('roe', 0),
                        'pe_ratio': row.get('pe_ratio', 0),
                        'net_income': row.get('net_income', 0),
                        'profitable': row.get('profitable', False),
                        'current_price': row.get('current_price', 0)
                    }

            # Generate synthetic price history for testing
            self.generate_synthetic_price_history()

            print(f"✅ Data loaded: {len(self.fundamentals)} fundamentals, {len(self.price_history)} price histories")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def generate_synthetic_price_history(self, days=30):
        """Generate synthetic price history for testing."""
        import random

        for ticker, fund_data in self.fundamentals.items():
            base_price = fund_data.get('current_price', 100)
            random.seed(hash(ticker) % 1000)  # Consistent random seed per ticker

            prices = []
            for day in range(days):
                # Generate realistic price movement
                change = random.gauss(0.001, 0.02)  # 0.1% mean, 2% std dev
                if day == 0:
                    price = base_price
                else:
                    price = max(prices[-1] * (1 + change), 1)  # Ensure positive prices

                prices.append({
                    'price': price,
                    'timestamp': datetime.now() - timedelta(days=days-day),
                    'volume': int(random.gauss(1000000, 200000)),
                    'change_pct': change * 100
                })

            self.price_history[ticker] = prices

    def calculate_technical_indicators(self, ticker: str) -> Dict:
        """Calculate technical indicators for a stock."""
        if ticker not in self.price_history or len(self.price_history[ticker]) < 1:
            return {}

        prices = [p['price'] for p in self.price_history[ticker]]
        current_price = prices[-1]
        volumes = [p.get('volume', 0) for p in self.price_history[ticker]]
        changes = [p.get('change_pct', 0) for p in self.price_history[ticker]]

        # Moving averages
        sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma_5

        # RSI (Relative Strength Index)
        rsi = self.calculate_rsi(prices)

        # Price momentum
        momentum_3d = (current_price - prices[-4]) / prices[-4] * 100 if len(prices) >= 4 else 0
        momentum_7d = (current_price - prices[-8]) / prices[-8] * 100 if len(prices) >= 8 else 0

        # Volume analysis
        avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1] if volumes else 1
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1

        # Volatility
        volatility = np.std(changes[-5:]) if len(changes) >= 5 else 0

        # Support/Resistance levels
        support = min(prices[-10:]) if len(prices) >= 10 else min(prices)
        resistance = max(prices[-10:]) if len(prices) >= 10 else max(prices)

        return {
            'current_price': current_price,
            'sma_5': sma_5,
            'sma_10': sma_10,
            'rsi': rsi,
            'momentum_3d': momentum_3d,
            'momentum_7d': momentum_7d,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'support': support,
            'resistance': resistance,
            'price_above_sma5': current_price > sma_5,
            'price_above_sma10': current_price > sma_10,
            'sma_trend': 'UP' if sma_5 > sma_10 else 'DOWN'
        }

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50  # Neutral

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_fundamental_score(self, ticker: str) -> Dict:
        """Calculate fundamental analysis score."""
        if ticker not in self.fundamentals:
            return {'score': 0, 'details': 'No fundamental data'}

        fund = self.fundamentals[ticker]
        score = 0
        details = []

        # ROE scoring (40% weight)
        roe = fund.get('roe', 0)
        if roe >= 20:
            score += 40
            details.append(f"Excellent ROE: {roe:.1f}%")
        elif roe >= 15:
            score += 30
            details.append(f"Good ROE: {roe:.1f}%")
        elif roe >= 10:
            score += 20
            details.append(f"Decent ROE: {roe:.1f}%")
        else:
            details.append(f"Low ROE: {roe:.1f}%")

        # P/E scoring (30% weight)
        pe = fund.get('pe_ratio', 0)
        if pe > 0:
            if pe <= 8:
                score += 30
                details.append(f"Very attractive P/E: {pe:.1f}")
            elif pe <= 12:
                score += 25
                details.append(f"Good P/E: {pe:.1f}")
            elif pe <= 18:
                score += 15
                details.append(f"Fair P/E: {pe:.1f}")
            else:
                details.append(f"High P/E: {pe:.1f}")
        else:
            details.append("No P/E data")

        # Profitability (30% weight)
        if fund.get('profitable', False):
            score += 30
            details.append("Profitable company")
        else:
            details.append("Not profitable")

        return {'score': score, 'details': '; '.join(details)}

    def generate_signal(self, ticker: str, risk_profile: str = 'MODERATE') -> Dict:
        """Generate comprehensive trading signal for a stock."""
        if ticker not in self.fundamentals or ticker not in self.price_history:
            return None

        # Get technical and fundamental data
        technical = self.calculate_technical_indicators(ticker)
        fundamental = self.calculate_fundamental_score(ticker)

        current_price = technical.get('current_price', 0)
        fund = self.fundamentals[ticker]

        # Initialize signal
        signal = {
            'ticker': ticker,
            'name': fund['name'],
            'current_price': current_price,
            'risk_profile': risk_profile,
            'timestamp': datetime.now().isoformat(),
            'technical_score': 0,
            'fundamental_score': fundamental['score'],
            'overall_score': 0,
            'action': 'HOLD',
            'confidence': 'LOW',
            'reasoning': []
        }

        # Technical Analysis Scoring (50% weight)
        tech_score = 0
        tech_reasons = []

        # RSI signals
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            tech_score += 20
            tech_reasons.append(f"Oversold (RSI: {rsi:.1f})")
        elif rsi > 70:
            tech_score -= 10
            tech_reasons.append(f"Overbought (RSI: {rsi:.1f})")

        # Moving average signals
        if technical.get('price_above_sma5') and technical.get('price_above_sma10'):
            tech_score += 15
            tech_reasons.append("Above moving averages")
        elif not technical.get('price_above_sma5'):
            tech_score -= 10
            tech_reasons.append("Below SMA5")

        # Momentum signals
        momentum = technical.get('momentum_3d', 0)
        if momentum > 2:
            tech_score += 15
            tech_reasons.append(f"Strong momentum (+{momentum:.1f}%)")
        elif momentum < -2:
            tech_score -= 15
            tech_reasons.append(f"Negative momentum ({momentum:.1f}%)")

        # Volume confirmation
        vol_ratio = technical.get('volume_ratio', 1)
        if vol_ratio > 1.5:
            tech_score += 10
            tech_reasons.append("High volume")
        elif vol_ratio < 0.5:
            tech_score -= 5
            tech_reasons.append("Low volume")

        signal['technical_score'] = max(0, min(100, tech_score + 50))  # Normalize to 0-100
        signal['technical_analysis'] = '; '.join(tech_reasons)

        # Calculate overall score
        signal['overall_score'] = (signal['technical_score'] * 0.5 + fundamental['score'] * 0.5)

        # Determine action based on risk profile
        risk_config = self.risk_levels[risk_profile]
        min_score = risk_config['min_score']

        if signal['overall_score'] >= min_score + 15:
            signal['action'] = 'STRONG BUY'
            signal['confidence'] = 'HIGH'
        elif signal['overall_score'] >= min_score + 5:
            signal['action'] = 'BUY'
            signal['confidence'] = 'MEDIUM'
        elif signal['overall_score'] >= min_score - 5:
            signal['action'] = 'HOLD'
            signal['confidence'] = 'LOW'
        else:
            signal['action'] = 'SELL'
            signal['confidence'] = 'HIGH'

        # Add support/resistance levels
        signal['support'] = technical.get('support', current_price * 0.95)
        signal['resistance'] = technical.get('resistance', current_price * 1.05)

        # Add reasoning
        signal['reasoning'].append(f"Overall Score: {signal['overall_score']:.1f}")
        signal['reasoning'].append(f"Technical: {signal['technical_score']:.1f}")
        signal['reasoning'].append(f"Fundamental: {fundamental['score']:.1f}")
        signal['reasoning'].append(f"Risk Profile: {risk_profile}")
        signal['reasoning'].extend(tech_reasons)
        signal['reasoning'].extend(fundamental['details'].split('; '))

        return signal

    def generate_portfolio_signals(self, risk_profile: str = 'MODERATE') -> List[Dict]:
        """Generate signals for all monitored stocks."""
        signals = []

        for ticker in self.fundamentals.keys():
            signal = self.generate_signal(ticker, risk_profile)
            if signal:
                signals.append(signal)

        # Sort by overall score
        signals.sort(key=lambda x: x['overall_score'], reverse=True)
        return signals

    def generate_enhanced_signals(self, capital: float = 100000, risk_profile: str = 'MODERATE') -> Dict:
        """Generate trading signals enhanced with comprehensive risk management."""
        print("🛡️ Generating risk-enhanced trading signals...")

        # Load ML system data
        self.ml_system.load_historical_data()
        self.ml_system.train_ml_models()

        # Get base signals
        base_signals = self.generate_portfolio_signals(risk_profile)

        # Get ML/RL signals
        ml_signals = self.ml_system.generate_risk_aware_signals(capital)

        # Enhance each signal with risk management
        enhanced_signals = []
        market_regimes = {}

        for signal in base_signals:
            ticker = signal['ticker']

            # Get ML/RL prediction for this ticker
            ml_signal = None
            for ml_s in ml_signals['signals']:
                if ml_s['ticker'] == ticker:
                    ml_signal = ml_s
                    break

            # Calculate market regime if we have price history
            if ticker in self.price_history and len(self.price_history[ticker]) >= 14:
                prices = [p['price'] for p in self.price_history[ticker]]
                df = pd.DataFrame({'close': prices})
                regime_info = self.market_regime_detector.analyze_regime(df)
                market_regimes[ticker] = regime_info
            else:
                market_regimes[ticker] = {'regime': 'UNKNOWN', 'strength': 0}

            # Calculate ATR for stop loss
            atr = self._calculate_atr(ticker)

            # Calculate position sizing using Kelly Criterion
            current_price = signal['current_price']
            stop_loss = self.risk_manager.calculate_optimal_stop_loss(
                current_price, atr, method='atr'
            )
            take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 risk/reward

            # Get risk parameters for profile
            risk_config = self.risk_levels[risk_profile]
            risk_per_trade = risk_config['max_risk_per_trade']

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                capital, risk_per_trade, current_price, stop_loss
            )

            # Calculate Kelly fraction
            if ml_signal:
                kelly_fraction = self.risk_manager.calculate_kelly_with_confidence(
                    ml_signal.get('ml_confidence', 0.5),
                    ml_signal.get('rl_confidence', 0.5),
                    ml_signal.get('ml_accuracy', 0.5)
                )
            else:
                kelly_fraction = self.risk_manager.calculate_kelly_criterion(0.6, 100, 50)

            # Validate risk-reward
            risk_reward_valid = self.risk_manager.validate_trade_risk_reward(
                current_price, stop_loss, take_profit
            )

            # Check correlation risk
            correlation_risk = self.risk_manager.check_correlation_risk(
                ticker, self.portfolio_positions
            )

            # Combine scores (Technical + Fundamental + ML/RL)
            base_score = signal['overall_score']
            ml_score = ml_signal['risk_adjusted_score'] if ml_signal else base_score * 0.8

            # Market regime adjustment
            regime = market_regimes[ticker]['regime']
            regime_adjustment = self._get_regime_multiplier(regime, signal['action'])

            # Risk-adjusted score
            risk_adjusted_score = (base_score * 0.4 + ml_score * 0.6) * regime_adjustment

            # Final risk-adjusted action
            final_action = self._determine_final_action(
                signal['action'], risk_adjusted_score, risk_reward_valid,
                correlation_risk['is_high_correlation'], regime, risk_profile
            )

            # Generate risk recommendations
            risk_recommendations = self._generate_risk_recommendations(
                final_action, kelly_fraction, correlation_risk, regime, risk_reward_valid
            )

            # Create base enhanced signal
            enhanced_signal = {
                **signal,
                'risk_adjusted_score': risk_adjusted_score,
                'risk_adjusted_action': final_action,
                'market_regime': market_regimes[ticker]['regime'],
                'regime_strength': market_regimes[ticker]['strength'],
                'kelly_fraction': kelly_fraction,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': self.risk_manager.calculate_risk_reward_ratio(
                    current_price, stop_loss, take_profit
                ),
                'atr_value': atr,
                'correlation_risk': correlation_risk,
                'ml_prediction': ml_signal.get('ml_prediction', 'N/A') if ml_signal else 'N/A',
                'ml_confidence': ml_signal.get('ml_confidence', 0) if ml_signal else 0,
                'rl_action': ml_signal.get('rl_action', 'N/A') if ml_signal else 'N/A',
                'risk_per_trade_pct': risk_per_trade * 100,
                'max_position_size': self.risk_manager.calculate_max_position_size(capital),
                'risk_recommendations': risk_recommendations,
                'portfolio_heat_check': self._check_portfolio_heat(capital)
            }

            # Apply ADX enhancement for advanced trend analysis
            enhanced_signal = self._enhance_with_adx_signals(ticker, enhanced_signal)

            enhanced_signals.append(enhanced_signal)

        # Sort by risk-adjusted score
        enhanced_signals.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)

        return {
            'signals': enhanced_signals,
            'market_regimes': market_regimes,
            'risk_profile': risk_profile,
            'capital': capital,
            'portfolio_summary': self._get_portfolio_summary(capital),
            'risk_management_enabled': True,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_atr(self, ticker: str, period: int = 14) -> float:
        """Calculate Average True Range for a ticker."""
        if ticker not in self.price_history or len(self.price_history[ticker]) < period:
            return 0.0

        prices = self.price_history[ticker]
        true_ranges = []

        for i in range(1, len(prices)):
            high = prices[i].get('high', prices[i]['price'])  # Use price if high not available
            low = prices[i].get('low', prices[i]['price'])    # Use price if low not available
            prev_close = prices[i-1]['price']

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _get_regime_multiplier(self, regime: str, action: str) -> float:
        """Get multiplier based on market regime and action with enhanced ADX filtering."""
        # Enhanced regime multipliers with trend direction consideration
        multipliers = {
            # Very Strong Trend (ADX > 60) - Strong trend following
            ('VERY_STRONG_TREND', 'STRONG BUY'): 1.4,
            ('VERY_STRONG_TREND', 'BUY'): 1.3,
            ('VERY_STRONG_TREND', 'SELL'): 0.5,  # Strongly avoid counter-trend

            # Strong Trend (ADX 40-60) - Good trend following
            ('STRONG_TREND', 'STRONG BUY'): 1.3,
            ('STRONG_TREND', 'BUY'): 1.2,
            ('STRONG_TREND', 'SELL'): 0.6,  # Avoid counter-trend

            # Emerging Strong Trend (ADX 25-40) - Acceptable trend following
            ('EMERGING_STRONG_TREND', 'STRONG BUY'): 1.1,
            ('EMERGING_STRONG_TREND', 'BUY'): 1.0,
            ('EMERGING_STRONG_TREND', 'SELL'): 0.8,

            # Weak Trend (ADX 20-25) - Reduced confidence
            ('WEAK_TREND', 'BUY'): 0.9,
            ('WEAK_TREND', 'SELL'): 0.9,
            ('WEAK_TREND', 'STRONG BUY'): 0.8,  # Downgrade strong signals

            # Consolidation (ADX < 20) - Range-bound, avoid directional bets
            ('CONSOLIDATION', 'BUY'): 0.6,
            ('CONSOLIDATION', 'SELL'): 0.6,
            ('CONSOLIDATION', 'STRONG BUY'): 0.5,  # Significant downgrade
            ('CONSOLIDATION', 'HOLD'): 1.2,  # Favor holding in consolidation
        }

        return multipliers.get((regime, action), 1.0)

    def _enhance_with_adx_signals(self, ticker: str, base_signal: Dict) -> Dict:
        """Enhance signal with ADX-based trend strength and direction analysis."""
        if ticker not in self.price_history or len(self.price_history[ticker]) < 14:
            return base_signal

        try:
            prices = [p['price'] for p in self.price_history[ticker]]
            highs = [p.get('high', p['price']) for p in self.price_history[ticker]]
            lows = [p.get('low', p['price']) for p in self.price_history[ticker]]

            if len(prices) < 14:
                return base_signal

            # Create DataFrame for analysis
            df = pd.DataFrame({
                'close': prices,
                'high': highs,
                'low': lows
            })

            # Get regime analysis
            regime_info = self.market_regime_detector.analyze_regime(df)
            regime = regime_info.get('regime', 'UNKNOWN')
            regime_strength = regime_info.get('strength', 0)

            # Calculate ADX components if available
            adx_value = regime_info.get('adx_value', 0)
            di_plus = regime_info.get('di_plus', 0)
            di_minus = regime_info.get('di_minus', 0)

            # Enhanced signal strength based on ADX
            adx_adjustment = 1.0
            if adx_value > 0:
                if adx_value > 60:  # Very strong trend
                    adx_adjustment = 1.3
                elif adx_value > 40:  # Strong trend
                    adx_adjustment = 1.2
                elif adx_value > 25:  # Emerging strong trend
                    adx_adjustment = 1.1
                elif adx_value < 20:  # Consolidation
                    adx_adjustment = 0.7

            # Trend direction confirmation (if DI data available)
            trend_direction_boost = 1.0
            if di_plus > di_minus and di_plus > 25:  # Bullish trend
                if base_signal['action'] in ['BUY', 'STRONG BUY']:
                    trend_direction_boost = 1.1
            elif di_minus > di_plus and di_minus > 25:  # Bearish trend
                if base_signal['action'] in ['SELL']:
                    trend_direction_boost = 1.1

            # Apply adjustments
            enhanced_signal = base_signal.copy()
            enhanced_signal['market_regime'] = regime
            enhanced_signal['regime_strength'] = regime_strength
            enhanced_signal['adx_value'] = adx_value
            enhanced_signal['di_plus'] = di_plus
            enhanced_signal['di_minus'] = di_minus
            enhanced_signal['adx_adjustment'] = adx_adjustment
            enhanced_signal['trend_direction_boost'] = trend_direction_boost

            # Adjust scores
            base_score = enhanced_signal.get('overall_score', 0)
            adjusted_score = base_score * adx_adjustment * trend_direction_boost
            enhanced_signal['adx_enhanced_score'] = adjusted_score

            # Enhanced position sizing based on ADX
            if 'kelly_fraction' in enhanced_signal:
                # Reduce Kelly in weak trends/consolidation
                if regime in ['CONSOLIDATION', 'WEAK_TREND']:
                    enhanced_signal['kelly_fraction'] *= 0.8
                # Increase Kelly in strong trends
                elif regime in ['VERY_STRONG_TREND', 'STRONG_TREND']:
                    enhanced_signal['kelly_fraction'] *= 1.1

            # Enhanced stop loss based on ADX
            if 'stop_loss' in enhanced_signal:
                current_price = enhanced_signal['current_price']
                atr = self._calculate_atr(ticker)

                # Tighter stops in strong trends, wider stops in consolidation
                if regime in ['VERY_STRONG_TREND', 'STRONG_TREND']:
                    enhanced_signal['adx_adjusted_stop_loss'] = current_price - (atr * 1.5)  # Tighter
                elif regime in ['CONSOLIDATION']:
                    enhanced_signal['adx_adjusted_stop_loss'] = current_price - (atr * 2.5)  # Wider
                else:
                    enhanced_signal['adx_adjusted_stop_loss'] = enhanced_signal['stop_loss']

            return enhanced_signal

        except Exception as e:
            print(f"⚠️ Error enhancing {ticker} with ADX signals: {e}")
            return base_signal

    def _determine_final_action(self, base_action: str, score: float, risk_reward_valid: bool,
                              high_correlation: bool, regime: str, risk_profile: str) -> str:
        """Determine final action considering all risk factors."""
        # Skip if risk-reward is invalid
        if not risk_reward_valid:
            if base_action in ['STRONG BUY', 'BUY']:
                return 'HOLD'

        # Skip if high correlation with existing positions
        if high_correlation and base_action in ['STRONG BUY', 'BUY']:
            return 'HOLD'

        # Conservative profiles need stronger signals
        min_scores = {'CONSERVATIVE': 70, 'MODERATE': 60, 'AGGRESSIVE': 50}

        if score < min_scores.get(risk_profile, 60):
            return 'HOLD'

        # Downgrade in unfavorable regimes
        if regime == 'CONSOLIDATION' and base_action == 'STRONG BUY':
            return 'BUY'

        return base_action

    def _generate_risk_recommendations(self, action: str, kelly_fraction: float,
                                     correlation_risk: Dict, regime: str, risk_reward_valid: bool) -> List[str]:
        """Generate comprehensive risk recommendations."""
        recommendations = []

        # Kelly fraction recommendations
        if kelly_fraction > 0.25:
            recommendations.append(f"⚠️ High Kelly fraction ({kelly_fraction:.2%}) - consider reducing position size")
        elif kelly_fraction < 0.05:
            recommendations.append(f"⚠️ Very low Kelly fraction ({kelly_fraction:.2%}) - consider avoiding this trade")

        # Correlation risk
        if correlation_risk['is_high_correlation']:
            recommendations.append(f"⚠️ High correlation with existing positions: {correlation_risk['correlated_positions']}")

        # Market regime recommendations
        if regime == 'CONSOLIDATION':
            recommendations.append("📊 Market in consolidation - reduce position sizes")
        elif regime in ['STRONG_TREND', 'VERY_STRONG_TREND']:
            recommendations.append(f"📈 Strong {regime.replace('_', ' ').lower()} detected - favorable conditions")
        elif regime == 'UNKNOWN':
            recommendations.append("❓ Insufficient data for regime analysis")

        # Risk-reward recommendations
        if not risk_reward_valid:
            recommendations.append("⚠️ Risk/reward ratio below minimum threshold - consider avoiding")

        # Action-specific recommendations
        if action == 'HOLD':
            recommendations.append("🔄 Risk management suggests holding - signal strength insufficient")

        return recommendations

    def _check_portfolio_heat(self, capital: float) -> Dict:
        """Check current portfolio heat."""
        if not self.portfolio_positions:
            return {
                'total_risk': 0,
                'portfolio_heat_pct': 0,
                'is_overheated': False,
                'remaining_capacity': self.risk_manager.max_portfolio_heat
            }

        return self.risk_manager.calculate_portfolio_heat(self.portfolio_positions, capital)

    def _get_portfolio_summary(self, capital: float) -> Dict:
        """Get current portfolio summary."""
        if not self.portfolio_positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'unrealized_pnl': 0,
                'total_risk': 0,
                'portfolio_heat': 0,
                'cash_available': capital
            }

        total_value = sum(pos.get('position_value', 0) for pos in self.portfolio_positions)
        total_risk = sum(pos.get('position_risk', 0) for pos in self.portfolio_positions)
        portfolio_heat = (total_risk / capital) * 100

        return {
            'total_positions': len(self.portfolio_positions),
            'total_value': total_value,
            'unrealized_pnl': sum(pos.get('unrealized_pnl', 0) for pos in self.portfolio_positions),
            'total_risk': total_risk,
            'portfolio_heat': portfolio_heat,
            'cash_available': capital - total_value,
            'risk_overheated': portfolio_heat > self.risk_manager.max_portfolio_heat
        }

    def display_enhanced_signals(self, enhanced_results: Dict):
        """Display enhanced trading signals with risk management."""
        signals = enhanced_results['signals']
        portfolio_summary = enhanced_results['portfolio_summary']
        risk_profile = enhanced_results['risk_profile']

        print(f"\n🛡️ RISK-ENHANCED TRADING SIGNALS - {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        print("=" * 100)
        print(f"📊 Risk Profile: {risk_profile} | Capital: {enhanced_results['capital']:,.0f} PLN | Total Signals: {len(signals)}")
        print(f"🔥 Portfolio Heat: {portfolio_summary['portfolio_heat']:.1f}% | Max Heat: {self.risk_manager.max_portfolio_heat:.1f}%")
        print("=" * 100)

        # Group by risk-adjusted action
        buy_signals = [s for s in signals if s['risk_adjusted_action'] in ['STRONG BUY', 'BUY']]
        hold_signals = [s for s in signals if s['risk_adjusted_action'] == 'HOLD']
        sell_signals = [s for s in signals if s['risk_adjusted_action'] == 'SELL']

        if buy_signals:
            print(f"\n🟢 RISK-ADJUSTED BUY SIGNALS ({len(buy_signals)}):")
            print("-" * 110)
            for signal in buy_signals[:5]:  # Top 5
                action_emoji = "🚀" if signal['risk_adjusted_action'] == 'STRONG BUY' else "📈"
                adx_score = signal.get('adx_enhanced_score', signal['risk_adjusted_score'])
                adx_value = signal.get('adx_value', 0)
                print(f"{action_emoji} {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {adx_score:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                      f"ADX: {adx_value:>4.0f} | RR: {signal['risk_reward_ratio']:.1f} | "
                      f"{signal['name'][:25]}")

        if hold_signals:
            print(f"\n🟡 RISK-ADJUSTED HOLD SIGNALS ({len(hold_signals)}):")
            print("-" * 110)
            for signal in hold_signals[:3]:  # Top 3
                adx_score = signal.get('adx_enhanced_score', signal['risk_adjusted_score'])
                adx_value = signal.get('adx_value', 0)
                print(f"⏸️  {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {adx_score:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                      f"ADX: {adx_value:>4.0f} | RR: {signal['risk_reward_ratio']:.1f} | "
                      f"{signal['name'][:25]}")

        if sell_signals:
            print(f"\n🔴 RISK-ADJUSTED SELL SIGNALS ({len(sell_signals)}):")
            print("-" * 110)
            for signal in sell_signals[:3]:  # Top 3
                adx_score = signal.get('adx_enhanced_score', signal['risk_adjusted_score'])
                adx_value = signal.get('adx_value', 0)
                print(f"📉 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {adx_score:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                      f"ADX: {adx_value:>4.0f} | RR: {signal['risk_reward_ratio']:.1f} | "
                      f"{signal['name'][:25]}")

        # Detailed analysis for top signal
        if buy_signals:
            top_signal = buy_signals[0]
            print(f"\n🔍 TOP RISK-ADJUSTED SIGNAL ANALYSIS:")
            print("=" * 80)
            print(f"📊 {top_signal['ticker']} - {top_signal['name']}")
            print(f"💰 Current Price: {top_signal['current_price']:.2f} PLN")
            print(f"🎯 Risk-Adjusted Action: {top_signal['risk_adjusted_action']}")
            print(f"📈 Risk-Adjusted Score: {top_signal['risk_adjusted_score']:.1f}")
            print(f"🛡️ Kelly Fraction: {top_signal['kelly_fraction']:.2%}")
            print(f"📊 Risk/Reward Ratio: {top_signal['risk_reward_ratio']:.2f}")
            print(f"🎯 Stop Loss: {top_signal['stop_loss']:.2f} PLN")
            print(f"🎯 Take Profit: {top_signal['take_profit']:.2f} PLN")
            print(f"📊 Market Regime: {top_signal['market_regime']}")
            print(f"🤖 ML Prediction: {top_signal['ml_prediction']}")
            print(f"🧠 RL Action: {top_signal['rl_action']}")
            print(f"📈 Position Size: {top_signal['position_size']['shares']:.0f} shares")
            print(f"💰 Position Value: {top_signal['position_size']['position_value']:,.0f} PLN")

            # Risk recommendations
            if top_signal['risk_recommendations']:
                print(f"\n⚠️  RISK RECOMMENDATIONS:")
                for rec in top_signal['risk_recommendations']:
                    print(f"   {rec}")

    def save_enhanced_signals(self, enhanced_results: Dict, filename: str = 'enhanced_trading_signals.json'):
        """Save enhanced signals to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 Enhanced signals saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Error saving enhanced signals: {str(e)}")

    def create_risk_aware_portfolio(self, signals: List[Dict], total_capital: float, risk_profile: str) -> Dict:
        """Create risk-aware portfolio allocation based on enhanced signals."""
        buy_signals = [s for s in signals if s['risk_adjusted_action'] in ['STRONG BUY', 'BUY']]

        if not buy_signals:
            return {"error": "No risk-qualified buy signals available"}

        # Calculate weights based on risk-adjusted scores and Kelly fractions
        total_score = sum(s['risk_adjusted_score'] * s['kelly_fraction'] for s in buy_signals)

        portfolio = {
            "total_capital": total_capital,
            "risk_profile": risk_profile,
            "allocation": [],
            "expected_annual_return": 0,
            "max_portfolio_risk": 0,
            "sharpe_ratio": 0,
            "risk_management_enabled": True
        }

        remaining_capital = total_capital
        used_capital = 0

        for signal in buy_signals[:8]:  # Max 8 positions for diversification
            if signal['current_price'] <= 0:
                continue

            # Weight based on risk-adjusted score and Kelly fraction
            weight = (signal['risk_adjusted_score'] * signal['kelly_fraction']) / total_score
            allocation_amount = min(remaining_capital * weight, signal['max_position_size'])

            # Calculate number of shares
            shares = int(allocation_amount / signal['current_price'])
            actual_cost = shares * signal['current_price']

            if actual_cost > 0 and actual_cost <= remaining_capital:
                portfolio["allocation"].append({
                    "ticker": signal['ticker'],
                    "name": signal['name'],
                    "shares": shares,
                    "price_per_share": signal['current_price'],
                    "total_cost": actual_cost,
                    "weight_percentage": (actual_cost / total_capital) * 100,
                    "stop_loss": signal['stop_loss'],
                    "take_profit": signal['take_profit'],
                    "kelly_fraction": signal['kelly_fraction'],
                    "risk_reward_ratio": signal['risk_reward_ratio'],
                    "market_regime": signal['market_regime'],
                    "ml_confidence": signal.get('ml_confidence', 0),
                    "position_risk": shares * (signal['current_price'] - signal['stop_loss'])
                })

                used_capital += actual_cost
                remaining_capital -= actual_cost

        # Calculate portfolio metrics
        portfolio["remaining_cash"] = remaining_capital
        portfolio["total_invested"] = used_capital
        portfolio["max_portfolio_risk"] = sum(pos['position_risk'] for pos in portfolio['allocation']) / total_capital

        # Estimate expected return and Sharpe ratio
        if portfolio['allocation']:
            avg_return = sum(pos['kelly_fraction'] * 0.15 for pos in portfolio['allocation']) / len(portfolio['allocation'])  # 15% base return assumption
            portfolio["expected_annual_return"] = avg_return
            portfolio["sharpe_ratio"] = avg_return / (portfolio["max_portfolio_risk"] * 100) if portfolio["max_portfolio_risk"] > 0 else 0

        return portfolio


def main():
    """Main function to generate enhanced trading signals with risk management."""
    print("🛡️ ENHANCED TRADING SIGNALS GENERATOR WITH RISK MANAGEMENT")
    print("=" * 70)

    generator = TradingSignalGenerator()

    # Load data
    if not generator.load_data():
        print("❌ Cannot load data. Please run analysis first.")
        return

    # Generate enhanced signals with risk management
    capital = 100000  # 100k PLN starting capital
    risk_profiles = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE']

    for profile in risk_profiles:
        print(f"\n🎯 GENERATING RISK-ENHANCED SIGNALS FOR {profile} PROFILE:")
        print(f"💰 Starting Capital: {capital:,.0f} PLN")

        # Generate enhanced signals
        enhanced_results = generator.generate_enhanced_signals(capital, profile)

        # Display enhanced signals
        generator.display_enhanced_signals(enhanced_results)

        # Save enhanced signals
        generator.save_enhanced_signals(enhanced_results, f'enhanced_trading_signals_{profile.lower()}.json')

        # Create risk-aware portfolio allocation
        portfolio = generator.create_risk_aware_portfolio(enhanced_results['signals'], capital, profile)
        if 'allocation' in portfolio:
            print(f"\n💼 RISK-AWARE PORTFOLIO ALLOCATION ({profile}):")
            print(f"   Total Capital: {portfolio['total_capital']:,.0f} PLN")
            print(f"   Invested: {portfolio['total_invested']:,.0f} PLN")
            print(f"   Cash: {portfolio['remaining_cash']:,.0f} PLN")
            print(f"   Positions: {len(portfolio['allocation'])}")
            print(f"   Max Portfolio Risk: {portfolio['max_portfolio_risk']:.1%}")
            print(f"   Expected Annual Return: {portfolio['expected_annual_return']:.1%}")
            print(f"   Risk-Adjusted Return: {portfolio['sharpe_ratio']:.2f}")

            for pos in portfolio['allocation'][:5]:
                print(f"   • {pos['ticker']}: {pos['shares']} shares @ {pos['price_per_share']:.2f} PLN "
                      f"({pos['weight_percentage']:.1f}%) | Kelly: {pos['kelly_fraction']:.2%} | RR: {pos['risk_reward_ratio']:.1f}")

        # Compare with traditional signals
        print(f"\n📊 TRADITIONAL VS RISK-ENHANCED COMPARISON ({profile}):")
        traditional_signals = generator.generate_portfolio_signals(profile)

        traditional_buy = len([s for s in traditional_signals if s['action'] in ['STRONG BUY', 'BUY']])
        enhanced_buy = len([s for s in enhanced_results['signals'] if s['risk_adjusted_action'] in ['STRONG BUY', 'BUY']])

        print(f"   Traditional Buy Signals: {traditional_buy}")
        print(f"   Risk-Enhanced Buy Signals: {enhanced_buy}")
        print(f"   Signal Reduction: {((traditional_buy - enhanced_buy) / traditional_buy * 100):.1f}%")
        print(f"   Risk Management Filter: {'Active' if enhanced_buy < traditional_buy else 'Passive'}")

        # Display risk metrics summary
        portfolio_summary = enhanced_results['portfolio_summary']
        print(f"\n📈 RISK METRICS SUMMARY ({profile}):")
        print(f"   Current Portfolio Heat: {portfolio_summary['portfolio_heat']:.1f}%")
        print(f"   Max Allowed Heat: {generator.risk_manager.max_portfolio_heat:.1f}%")
        print(f"   Available Capital: {portfolio_summary['cash_available']:,.0f} PLN")
        print(f"   Risk Management Status: {'✅ Optimal' if not portfolio_summary['risk_overheated'] else '⚠️ Overheated'}")

    print(f"\n🏁 Enhanced trading signals generation complete!")
    print(f"📊 Check enhanced signal files: enhanced_trading_signals_*.json")
    print(f"🛡️ Risk management features:")
    print(f"   ✅ Kelly Criterion position sizing")
    print(f"   ✅ Market regime detection and filtering")
    print(f"   ✅ Risk-reward ratio validation")
    print(f"   ✅ Correlation risk assessment")
    print(f"   ✅ Portfolio heat management")
    print(f"   ✅ ML/RL confidence integration")


if __name__ == "__main__":
    main()