#!/usr/bin/env python3
"""
Advanced ML/RL Trading System
Machine Learning and Reinforcement Learning for stock trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLTradingSystem:
    """Advanced trading system with ML/RL capabilities."""

    def __init__(self):
        self.price_data = {}
        self.fundamental_data = {}
        self.technical_indicators = {}
        self.ml_models = {}
        self.scalers = {}
        self.prediction_history = []
        self.rl_agent = None

    def load_historical_data(self, prices_file='data/current_prices.csv',
                           fundamentals_file='wig30_analysis_pe_threshold.csv'):
        """Load historical data for ML training."""
        try:
            # Load fundamental data
            if os.path.exists(fundamentals_file):
                df_fund = pd.read_csv(fundamentals_file)
                for _, row in df_fund.iterrows():
                    self.fundamental_data[row['ticker']] = {
                        'name': row['name'],
                        'roe': row.get('roe', 0),
                        'pe_ratio': row.get('pe_ratio', 0),
                        'net_income': row.get('net_income', 0),
                        'profitable': row.get('profitable', False),
                        'current_price': row.get('current_price', 0)
                    }

            # Generate synthetic historical data for ML training
            self.generate_synthetic_historical_data()

            print(f"✅ ML Data loaded: {len(self.fundamental_data)} stocks with historical data")
            return True

        except Exception as e:
            print(f"❌ Error loading ML data: {str(e)}")
            return False

    def generate_synthetic_historical_data(self, days=252):
        """Generate synthetic historical price data for ML training."""
        print("🔄 Generating synthetic historical data for ML training...")

        for ticker, fund_data in self.fundamental_data.items():
            base_price = fund_data.get('current_price', 100)

            # Generate realistic price movements
            np.random.seed(hash(ticker) % 1000)  # Consistent random seed per ticker
            daily_returns = np.random.normal(0.001, 0.02, days)  # 0.1% mean, 2% std dev

            # Add trend based on fundamentals
            if fund_data.get('roe', 0) > 15:  # Good ROE -> slight uptrend
                daily_returns += 0.0005
            elif fund_data.get('roe', 0) < 5:  # Poor ROE -> slight downtrend
                daily_returns -= 0.0005

            prices = [base_price]
            for ret in daily_returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1))  # Ensure positive prices

            # Create OHLC data
            historical_data = []
            for i in range(1, len(prices)):
                high = prices[i] * (1 + abs(np.random.normal(0, 0.01)))
                low = prices[i] * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
                close = prices[i]
                volume = int(np.random.normal(1000000, 200000))

                historical_data.append({
                    'date': datetime.now() - timedelta(days=days-i),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })

            self.price_data[ticker] = historical_data

    def calculate_advanced_indicators(self, ticker: str) -> Dict:
        """Calculate advanced technical indicators for ML."""
        if ticker not in self.price_data:
            return {}

        prices_df = pd.DataFrame(self.price_data[ticker])

        # Price-based indicators
        prices_df['sma_5'] = prices_df['close'].rolling(5).mean()
        prices_df['sma_10'] = prices_df['close'].rolling(10).mean()
        prices_df['sma_20'] = prices_df['close'].rolling(20).mean()
        prices_df['ema_12'] = prices_df['close'].ewm(span=12).mean()
        prices_df['ema_26'] = prices_df['close'].ewm(span=26).mean()

        # MACD
        prices_df['macd'] = prices_df['ema_12'] - prices_df['ema_26']
        prices_df['macd_signal'] = prices_df['macd'].ewm(span=9).mean()
        prices_df['macd_histogram'] = prices_df['macd'] - prices_df['macd_signal']

        # RSI
        delta = prices_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        prices_df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        prices_df['bb_middle'] = prices_df['close'].rolling(20).mean()
        bb_std = prices_df['close'].rolling(20).std()
        prices_df['bb_upper'] = prices_df['bb_middle'] + (bb_std * 2)
        prices_df['bb_lower'] = prices_df['bb_middle'] - (bb_std * 2)

        # Stochastic Oscillator
        low_min = prices_df['low'].rolling(14).min()
        high_max = prices_df['high'].rolling(14).max()
        prices_df['stoch_k'] = 100 * (prices_df['close'] - low_min) / (high_max - low_min)
        prices_df['stoch_d'] = prices_df['stoch_k'].rolling(3).mean()

        # Volume indicators
        prices_df['volume_sma'] = prices_df['volume'].rolling(10).mean()
        prices_df['volume_ratio'] = prices_df['volume'] / prices_df['volume_sma']

        # Price momentum
        prices_df['momentum_5'] = prices_df['close'].pct_change(5)
        prices_df['momentum_10'] = prices_df['close'].pct_change(10)

        # Volatility
        prices_df['volatility'] = prices_df['close'].pct_change().rolling(10).std()

        # Store latest indicators
        if len(prices_df) > 0:
            latest = prices_df.iloc[-1].to_dict()
            self.technical_indicators[ticker] = latest
            return latest

        return {}

    def prepare_ml_features(self, ticker: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML training."""
        if ticker not in self.price_data or len(self.price_data[ticker]) < 30:
            return np.array([]), np.array([])

        # Get technical indicators
        self.calculate_advanced_indicators(ticker)
        df = pd.DataFrame(self.price_data[ticker])

        # Calculate all indicators
        self.calculate_advanced_indicators(ticker)
        df_full = pd.DataFrame(self.price_data[ticker])

        # Recalculate indicators for full dataset
        df_full['sma_5'] = df_full['close'].rolling(5).mean()
        df_full['sma_10'] = df_full['close'].rolling(10).mean()
        df_full['sma_20'] = df_full['close'].rolling(20).mean()
        df_full['rsi'] = 100 - (100 / (1 + (df_full['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                                                     (-df_full['close'].diff().where(lambda x: x < 0, 0)).rolling(14).mean())))
        df_full['volume_ratio'] = df_full['volume'] / df_full['volume'].rolling(10).mean()
        df_full['volatility'] = df_full['close'].pct_change().rolling(10).std()

        # Create target variable (next day's price direction)
        df_full['target'] = (df_full['close'].shift(-1) > df_full['close']).astype(int)
        df_full['target_return'] = df_full['close'].pct_change().shift(-1)

        # Select features
        feature_columns = ['sma_5', 'sma_10', 'sma_20', 'rsi', 'volume_ratio', 'volatility']

        # Add price-based features
        df_full['price_to_sma5'] = df_full['close'] / df_full['sma_5']
        df_full['price_to_sma20'] = df_full['close'] / df_full['sma_20']
        df_full['volume_price_ratio'] = df_full['volume'] / df_full['close']

        feature_columns.extend(['price_to_sma5', 'price_to_sma20', 'volume_price_ratio'])

        # Remove NaN values
        df_clean = df_full.dropna()

        if len(df_clean) < 50:
            return np.array([]), np.array([])

        X = df_clean[feature_columns].values
        y_classification = df_clean['target'].values
        y_regression = df_clean['target_return'].values

        # Add fundamental features
        if ticker in self.fundamental_data:
            fund = self.fundamental_data[ticker]
            fundamental_features = np.array([
                fund.get('roe', 0) / 100,  # Normalize
                fund.get('pe_ratio', 0) / 20,  # Normalize
                1 if fund.get('profitable', False) else 0
            ])

            # Repeat fundamental features for each row
            fund_features = np.tile(fundamental_features, (len(X), 1))
            X = np.hstack([X, fund_features])

        return X, y_classification

    def train_ml_models(self):
        """Train ML models for each stock."""
        print("🤖 Training ML models for each stock...")

        for ticker in self.fundamental_data.keys():
            X, y = self.prepare_ml_features(ticker)

            if len(X) < 50:
                print(f"⚠️  Insufficient data for {ticker}")
                continue

            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train classification model
                rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_classifier.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = rf_classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                # Store model and scaler
                self.ml_models[ticker] = {
                    'classifier': rf_classifier,
                    'scaler': scaler,
                    'accuracy': accuracy,
                    'feature_count': X.shape[1]
                }

                print(f"✅ {ticker}: ML model trained (accuracy: {accuracy:.3f})")

            except Exception as e:
                print(f"❌ Error training model for {ticker}: {str(e)}")

    def predict_next_day(self, ticker: str) -> Dict:
        """Predict next day's price movement using ML."""
        if ticker not in self.ml_models:
            return {}

        try:
            X, _ = self.prepare_ml_features(ticker)
            if len(X) == 0:
                return {}

            # Use latest features
            latest_features = X[-1:].reshape(1, -1)
            model_data = self.ml_models[ticker]

            # Scale features
            scaled_features = model_data['scaler'].transform(latest_features)

            # Predict
            prediction_proba = model_data['classifier'].predict_proba(scaled_features)[0]
            prediction = model_data['classifier'].predict(scaled_features)[0]

            current_price = self.fundamental_data[ticker].get('current_price', 0)
            indicators = self.technical_indicators.get(ticker, {})

            result = {
                'ticker': ticker,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': prediction_proba[1] if prediction == 1 else prediction_proba[0],
                'accuracy': model_data['accuracy'],
                'current_price': current_price,
                'rsi': indicators.get('rsi', 50),
                'macd': indicators.get('macd', 0),
                'volume_ratio': indicators.get('volume_ratio', 1),
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            print(f"❌ Error predicting for {ticker}: {str(e)}")
            return {}

    def generate_ml_signals(self) -> List[Dict]:
        """Generate trading signals using ML predictions."""
        print("🧠 Generating ML-based trading signals...")

        signals = []

        for ticker in self.fundamental_data.keys():
            prediction = self.predict_next_day(ticker)
            if prediction:
                # Calculate signal strength
                signal_strength = 0

                # ML prediction (40% weight)
                if prediction['prediction'] == 'UP':
                    signal_strength += prediction['confidence'] * 40
                else:
                    signal_strength -= (1 - prediction['confidence']) * 40

                # Technical indicators (30% weight)
                rsi = prediction.get('rsi', 50)
                if rsi < 30:  # Oversold
                    signal_strength += 30
                elif rsi > 70:  # Overbought
                    signal_strength -= 30

                # Volume confirmation (15% weight)
                vol_ratio = prediction.get('volume_ratio', 1)
                if vol_ratio > 1.5:  # High volume
                    signal_strength += 15
                elif vol_ratio < 0.5:  # Low volume
                    signal_strength -= 10

                # Model accuracy (15% weight)
                accuracy = prediction.get('accuracy', 0.5)
                if prediction['prediction'] == 'UP':
                    signal_strength += accuracy * 15
                else:
                    signal_strength -= (1 - accuracy) * 15

                # Determine action
                if signal_strength > 60:
                    action = 'STRONG BUY'
                elif signal_strength > 30:
                    action = 'BUY'
                elif signal_strength > -30:
                    action = 'HOLD'
                else:
                    action = 'SELL'

                signals.append({
                    'ticker': ticker,
                    'name': self.fundamental_data[ticker]['name'],
                    'current_price': prediction['current_price'],
                    'action': action,
                    'signal_strength': signal_strength,
                    'ml_prediction': prediction['prediction'],
                    'ml_confidence': prediction['confidence'],
                    'ml_accuracy': prediction['accuracy'],
                    'rsi': rsi,
                    'volume_ratio': vol_ratio,
                    'timestamp': datetime.now().isoformat()
                })

        # Sort by signal strength
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        return signals

    def create_reinforcement_learner(self):
        """Create a simple Q-learning agent for trading."""
        print("🧠 Initializing Reinforcement Learning agent...")

        class SimpleQLearner:
            def __init__(self, actions=['BUY', 'SELL', 'HOLD']):
                self.actions = actions
                self.q_table = {}
                self.learning_rate = 0.1
                self.discount_factor = 0.95
                self.epsilon = 0.1

            def get_state(self, price_change, rsi, volume_ratio):
                """Discretize continuous state values."""
                price_state = 'UP' if price_change > 0.01 else 'DOWN' if price_change < -0.01 else 'NEUTRAL'
                rsi_state = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
                vol_state = 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.7 else 'NORMAL'

                return f"{price_state}_{rsi_state}_{vol_state}"

            def choose_action(self, state):
                """Choose action using epsilon-greedy policy."""
                if np.random.random() < self.epsilon:
                    return np.random.choice(self.actions)

                if state not in self.q_table:
                    self.q_table[state] = {action: 0 for action in self.actions}

                return max(self.q_table[state], key=self.q_table[state].get)

            def update_q_value(self, state, action, reward, next_state):
                """Update Q-value using Q-learning formula."""
                if state not in self.q_table:
                    self.q_table[state] = {action: 0 for action in self.actions}
                if next_state not in self.q_table:
                    self.q_table[next_state] = {action: 0 for action in self.actions}

                current_q = self.q_table[state][action]
                max_next_q = max(self.q_table[next_state].values())
                new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

                self.q_table[state][action] = new_q

        self.rl_agent = SimpleQLearner()
        print("✅ RL agent initialized")

    def train_rl_agent(self, episodes=1000):
        """Train the RL agent using historical data."""
        if not self.rl_agent:
            self.create_reinforcement_learner()

        print(f"🎯 Training RL agent for {episodes} episodes...")

        for episode in range(episodes):
            total_reward = 0

            for ticker in list(self.fundamental_data.keys())[:3]:  # Train on subset for speed
                if ticker not in self.price_data:
                    continue

                prices = [p['close'] for p in self.price_data[ticker]]
                indicators = self.technical_indicators.get(ticker, {})

                for i in range(1, len(prices)-1):
                    price_change = (prices[i] - prices[i-1]) / prices[i-1]
                    rsi = indicators.get('rsi', 50)
                    volume_ratio = indicators.get('volume_ratio', 1)

                    state = self.rl_agent.get_state(price_change, rsi, volume_ratio)
                    action = self.rl_agent.choose_action(state)

                    # Calculate reward
                    next_price_change = (prices[i+1] - prices[i]) / prices[i]

                    if action == 'BUY' and next_price_change > 0:
                        reward = next_price_change * 100  # Profit
                    elif action == 'SELL' and next_price_change < 0:
                        reward = abs(next_price_change) * 100  # Avoided loss
                    elif action == 'HOLD':
                        reward = -0.1  # Small cost for holding
                    else:
                        reward = -next_price_change * 100  # Loss

                    next_state = self.rl_agent.get_state(next_price_change, rsi, volume_ratio)
                    self.rl_agent.update_q_value(state, action, reward, next_state)

                    total_reward += reward

            if episode % 100 == 0:
                print(f"Episode {episode}: Total reward: {total_reward:.2f}")

        print("✅ RL agent training completed")

    def get_rl_recommendation(self, ticker: str) -> Dict:
        """Get trading recommendation from RL agent."""
        if not self.rl_agent or ticker not in self.technical_indicators:
            return {}

        indicators = self.technical_indicators[ticker]

        # Use recent price change
        if ticker in self.price_data and len(self.price_data[ticker]) > 1:
            prices = [p['close'] for p in self.price_data[ticker]]
            price_change = (prices[-1] - prices[-2]) / prices[-2]
        else:
            price_change = 0

        state = self.rl_agent.get_state(
            price_change,
            indicators.get('rsi', 50),
            indicators.get('volume_ratio', 1)
        )

        action = self.rl_agent.choose_action(state)
        q_values = self.rl_agent.q_table.get(state, {})

        return {
            'ticker': ticker,
            'rl_action': action,
            'rl_confidence': max(q_values.values()) if q_values else 0,
            'state': state,
            'q_values': q_values
        }

    def generate_comprehensive_signals(self) -> Dict:
        """Generate comprehensive signals combining ML and RL."""
        print("🚀 Generating comprehensive ML/RL trading signals...")

        # Train models if not already trained
        if not self.ml_models:
            self.train_ml_models()

        # Get ML signals
        ml_signals = self.generate_ml_signals()

        # Get RL recommendations
        rl_recommendations = {}
        if self.rl_agent:
            for ticker in self.fundamental_data.keys():
                rl_rec = self.get_rl_recommendation(ticker)
                if rl_rec:
                    rl_recommendations[ticker] = rl_rec

        # Combine signals
        comprehensive_signals = []
        for ml_signal in ml_signals:
            ticker = ml_signal['ticker']
            rl_rec = rl_recommendations.get(ticker, {})

            # Calculate combined score
            ml_score = ml_signal['signal_strength']
            rl_score = 0

            if rl_rec:
                if rl_rec['rl_action'] == 'BUY':
                    rl_score = rl_rec['rl_confidence'] * 20
                elif rl_rec['rl_action'] == 'SELL':
                    rl_score = -rl_rec['rl_confidence'] * 20

            combined_score = ml_score + rl_score

            # Determine final action
            if combined_score > 70:
                final_action = 'STRONG BUY'
            elif combined_score > 40:
                final_action = 'BUY'
            elif combined_score > -40:
                final_action = 'HOLD'
            else:
                final_action = 'SELL'

            comprehensive_signals.append({
                **ml_signal,
                'rl_action': rl_rec.get('rl_action', 'N/A'),
                'rl_confidence': rl_rec.get('rl_confidence', 0),
                'combined_score': combined_score,
                'final_action': final_action
            })

        # Sort by combined score
        comprehensive_signals.sort(key=lambda x: x['combined_score'], reverse=True)

        return {
            'signals': comprehensive_signals,
            'ml_model_count': len(self.ml_models),
            'rl_trained': self.rl_agent is not None,
            'timestamp': datetime.now().isoformat()
        }

    def save_models(self, directory='ml_models'):
        """Save trained models for later use."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save ML models metadata
        models_metadata = {}
        for ticker, model_data in self.ml_models.items():
            models_metadata[ticker] = {
                'accuracy': model_data['accuracy'],
                'feature_count': model_data['feature_count'],
                'timestamp': datetime.now().isoformat()
            }

        with open(f'{directory}/models_metadata.json', 'w') as f:
            json.dump(models_metadata, f, indent=2)

        # Note: In a real implementation, you'd save the actual model objects
        print(f"💾 ML models metadata saved to {directory}/models_metadata.json")


def main():
    """Main function to run the advanced ML/RL trading system."""
    print("🤖 ADVANCED ML/RL TRADING SYSTEM")
    print("=" * 60)

    system = AdvancedMLTradingSystem()

    # Load and prepare data
    if not system.load_historical_data():
        print("❌ Cannot load data. Exiting.")
        return

    # Train ML models
    system.train_ml_models()

    # Train RL agent
    system.train_rl_agent(episodes=500)  # Reduced for faster execution

    # Generate comprehensive signals
    results = system.generate_comprehensive_signals()

    # Display results
    print(f"\n🎯 COMPREHENSIVE ML/RL TRADING SIGNALS")
    print("=" * 80)
    print(f"🤖 ML Models: {results['ml_model_count']}")
    print(f"🧠 RL Agent: {'Trained' if results['rl_trained'] else 'Not trained'}")
    print("=" * 80)

    # Display top signals
    signals = results['signals'][:10]  # Top 10

    buy_signals = [s for s in signals if s['final_action'] in ['STRONG BUY', 'BUY']]
    hold_signals = [s for s in signals if s['final_action'] == 'HOLD']
    sell_signals = [s for s in signals if s['final_action'] == 'SELL']

    if buy_signals:
        print(f"\n🟢 ML/RL BUY SIGNALS ({len(buy_signals)}):")
        print("-" * 80)
        for signal in buy_signals[:5]:
            print(f"🚀 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['combined_score']:>6.1f} | ML: {signal['ml_prediction']:<4} "
                  f"| RL: {signal['rl_action']:<4} | {signal['name'][:30]}")

    if hold_signals:
        print(f"\n🟡 ML/RL HOLD SIGNALS ({len(hold_signals)}):")
        print("-" * 80)
        for signal in hold_signals[:3]:
            print(f"⏸️  {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['combined_score']:>6.1f} | ML: {signal['ml_prediction']:<4} "
                  f"| RL: {signal['rl_action']:<4} | {signal['name'][:30]}")

    if sell_signals:
        print(f"\n🔴 ML/RL SELL SIGNALS ({len(sell_signals)}):")
        print("-" * 80)
        for signal in sell_signals[:3]:
            print(f"📉 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['combined_score']:>6.1f} | ML: {signal['ml_prediction']:<4} "
                  f"| RL: {signal['rl_action']:<4} | {signal['name'][:30]}")

    # Detailed analysis for top signal
    if signals:
        top_signal = signals[0]
        print(f"\n🔍 TOP ML/RL SIGNAL ANALYSIS:")
        print(f"   📊 {top_signal['ticker']} - {top_signal['name']}")
        print(f"   💰 Current Price: {top_signal['current_price']:.2f} PLN")
        print(f"   🎯 Final Action: {top_signal['final_action']}")
        print(f"   📈 Combined Score: {top_signal['combined_score']:.1f}")
        print(f"   🤖 ML Prediction: {top_signal['ml_prediction']} (confidence: {top_signal['ml_confidence']:.3f})")
        print(f"   🧠 RL Action: {top_signal['rl_action']} (confidence: {top_signal['rl_confidence']:.3f})")
        print(f"   📊 ML Accuracy: {top_signal['ml_accuracy']:.3f}")
        print(f"   📈 RSI: {top_signal['rsi']:.1f}")

    # Save results
    with open('ml_rl_trading_signals.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    system.save_models()

    print(f"\n💾 Results saved to: ml_rl_trading_signals.json")
    print(f"🏁 Advanced ML/RL analysis complete!")


if __name__ == "__main__":
    main()