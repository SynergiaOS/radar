#!/usr/bin/env python3
"""
Advanced Trading Signals System
Provides buy/sell recommendations based on technical and fundamental analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class TradingSignalGenerator:
    """Advanced trading signals generator for WIG30 stocks."""

    def __init__(self):
        self.signals = []
        self.price_history = {}
        self.fundamentals = {}
        self.risk_levels = {
            'CONSERVATIVE': {'min_roe': 12, 'max_pe': 12, 'min_score': 75},
            'MODERATE': {'min_roe': 10, 'max_pe': 15, 'min_score': 65},
            'AGGRESSIVE': {'min_roe': 8, 'max_pe': 20, 'min_score': 55}
        }

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
                        'profitable': row.get('profitable', False)
                    }

            # Load current prices
            if prices_file and os.path.exists(prices_file):
                df_prices = pd.read_csv(prices_file)
                for _, row in df_prices.iterrows():
                    ticker = row['ticker']
                    if ticker not in self.price_history:
                        self.price_history[ticker] = []

                    price = row.get('price', 0)
                    if price > 0:  # Only add valid prices
                        self.price_history[ticker].append({
                            'price': float(price),
                            'timestamp': datetime.now(),
                            'volume': row.get('volume', 0),
                            'change_pct': row.get('change', 0)
                        })

                        # Keep only last 50 price points
                        if len(self.price_history[ticker]) > 50:
                            self.price_history[ticker] = self.price_history[ticker][-50:]

            print(f"✅ Załadowano dane: {len(self.fundamentals)} fundamentals, {len(self.price_history)} cen")
            return True

        except Exception as e:
            print(f"❌ Błąd ładowania danych: {str(e)}")
            return False

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

    def display_signals(self, signals: List[Dict]):
        """Display trading signals in professional format."""
        print(f"\n🎯 TRADING SIGNALS - {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        print("=" * 80)
        print(f"📊 Profile: {signals[0]['risk_profile'] if signals else 'N/A'} | Total Signals: {len(signals)}")
        print("=" * 80)

        # Group by action
        buy_signals = [s for s in signals if s['action'] in ['STRONG BUY', 'BUY']]
        hold_signals = [s for s in signals if s['action'] == 'HOLD']
        sell_signals = [s for s in signals if s['action'] == 'SELL']

        if buy_signals:
            print(f"\n🟢 BUY SIGNALS ({len(buy_signals)}):")
            print("-" * 80)
            for signal in buy_signals[:5]:  # Top 5
                action_emoji = "🚀" if signal['action'] == 'STRONG BUY' else "📈"
                print(f"{action_emoji} {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {signal['overall_score']:>6.1f} | {signal['confidence']:<6} | "
                      f"{signal['name'][:30]}")

        if hold_signals:
            print(f"\n🟡 HOLD SIGNALS ({len(hold_signals)}):")
            print("-" * 80)
            for signal in hold_signals[:3]:  # Top 3
                print(f"⏸️  {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {signal['overall_score']:>6.1f} | {signal['confidence']:<6} | "
                      f"{signal['name'][:30]}")

        if sell_signals:
            print(f"\n🔴 SELL SIGNALS ({len(sell_signals)}):")
            print("-" * 80)
            for signal in sell_signals[:3]:  # Top 3
                print(f"📉 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                      f"Score: {signal['overall_score']:>6.1f} | {signal['confidence']:<6} | "
                      f"{signal['name'][:30]}")

        # Detailed analysis for top signals
        if buy_signals:
            print(f"\n🔍 DETAILED ANALYSIS - TOP 3 BUY SIGNALS:")
            print("=" * 80)
            for signal in buy_signals[:3]:
                print(f"\n📊 {signal['ticker']} - {signal['name']}")
                print(f"   💰 Current Price: {signal['current_price']:.2f} PLN")
                print(f"   🎯 Action: {signal['action']} (Confidence: {signal['confidence']})")
                print(f"   📈 Overall Score: {signal['overall_score']:.1f}/100")
                print(f"   🔧 Technical Score: {signal['technical_score']:.1f}/100")
                print(f"   💼 Fundamental Score: {signal['fundamental_score']:.1f}/100")
                print(f"   🛡️  Support: {signal['support']:.2f} PLN")
                print(f"   🚀 Resistance: {signal['resistance']:.2f} PLN")
                print(f"   💡 Key Reasons: {', '.join(signal['reasoning'][:3])}")

    def save_signals(self, signals: List[Dict], filename: str = 'trading_signals.json'):
        """Save signals to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(signals, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Signals saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Error saving signals: {str(e)}")

    def create_portfolio_allocation(self, signals: List[Dict], total_capital: float = 10000) -> Dict:
        """Create optimal portfolio allocation based on signals."""
        buy_signals = [s for s in signals if s['action'] in ['STRONG BUY', 'BUY']]

        if not buy_signals:
            return {"error": "No buy signals available"}

        # Calculate weights based on scores
        total_score = sum(s['overall_score'] for s in buy_signals)

        portfolio = {
            "total_capital": total_capital,
            "risk_profile": buy_signals[0]['risk_profile'],
            "allocation": [],
            "expected_return": 0,
            "risk_level": "MODERATE"
        }

        remaining_capital = total_capital

        for signal in buy_signals[:8]:  # Max 8 positions
            if signal['current_price'] <= 0:
                continue  # Skip invalid prices

            weight = signal['overall_score'] / total_score
            allocation_amount = remaining_capital * weight

            # Calculate number of shares
            shares = int(allocation_amount / signal['current_price'])
            actual_cost = shares * signal['current_price']

            portfolio["allocation"].append({
                "ticker": signal['ticker'],
                "name": signal['name'],
                "shares": shares,
                "price_per_share": signal['current_price'],
                "total_cost": actual_cost,
                "weight_percentage": (actual_cost / total_capital) * 100,
                "stop_loss": signal['support'] * 0.95,
                "take_profit": signal['resistance'] * 1.10
            })

            remaining_capital -= actual_cost

        portfolio["remaining_cash"] = remaining_capital
        portfolio["total_invested"] = total_capital - remaining_capital

        return portfolio


def main():
    """Main function to generate trading signals."""
    print("🎯 TRADING SIGNALS GENERATOR")
    print("=" * 50)

    generator = TradingSignalGenerator()

    # Load data
    if not generator.load_data():
        print("❌ Cannot load data. Please run analysis first.")
        return

    # Generate signals for different risk profiles
    risk_profiles = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE']

    for profile in risk_profiles:
        print(f"\n🎯 GENERATING SIGNALS FOR {profile} PROFILE:")
        signals = generator.generate_portfolio_signals(profile)
        generator.display_signals(signals)
        generator.save_signals(signals, f'trading_signals_{profile.lower()}.json')

        # Create portfolio allocation
        portfolio = generator.create_portfolio_allocation(signals)
        if 'allocation' in portfolio:
            print(f"\n💼 SUGGESTED PORTFOLIO ALLOCATION ({profile}):")
            print(f"   Total Capital: {portfolio['total_capital']:,.0f} PLN")
            print(f"   Invested: {portfolio['total_invested']:,.0f} PLN")
            print(f"   Cash: {portfolio['remaining_cash']:,.0f} PLN")
            print(f"   Positions: {len(portfolio['allocation'])}")

            for pos in portfolio['allocation'][:5]:
                print(f"   • {pos['ticker']}: {pos['shares']} shares @ {pos['price_per_share']:.2f} PLN "
                      f"({pos['weight_percentage']:.1f}%)")

    print(f"\n🏁 Trading signals generation complete!")
    print(f"📊 Check signal files: trading_signals_*.json")


if __name__ == "__main__":
    main()