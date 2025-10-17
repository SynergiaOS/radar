#!/usr/bin/env python3
"""
Advanced ML/RL Trading System
Machine Learning and Reinforcement Learning for stock trading decisions
Enhanced with comprehensive risk management features
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

# Import risk management and market regime modules
from risk_management import RiskManager
from market_regime import MarketRegimeDetector

class AdvancedMLTradingSystem:
    """Advanced trading system with ML/RL capabilities enhanced with risk management."""

    def __init__(self):
        self.price_data = {}
        self.fundamental_data = {}
        self.technical_indicators = {}
        self.ml_models = {}
        self.scalers = {}
        self.prediction_history = []
        self.rl_agent = None

        # Enhanced features with risk management
        self.risk_manager = RiskManager()
        self.market_regime_detector = MarketRegimeDetector()
        self.portfolio_positions = []
        self.trading_history = []
        self.market_regimes = {}

        # Performance tracking
        self.win_rate_history = {}
        self.avg_win_loss = {}
        self.model_confidence_scores = {}

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

    def generate_risk_aware_signals(self, capital: float = 100000) -> Dict:
        """Generate trading signals enhanced with risk management and regime filtering."""
        print("🛡️ Generating advanced risk-aware ML/RL trading signals with regime filtering...")

        # Get comprehensive signals
        base_signals = self.generate_comprehensive_signals()

        # Update market regimes for all tickers using enhanced ADX analysis
        print("📊 Analyzing market regimes across all tickers...")
        for ticker in self.fundamental_data.keys():
            if ticker in self.price_data:
                prices = [p['close'] for p in self.price_data[ticker]]
                highs = [p['high'] for p in self.price_data[ticker]]
                lows = [p['low'] for p in self.price_data[ticker]]

                if len(prices) >= 14:
                    df = pd.DataFrame({
                        'close': prices,
                        'high': highs,
                        'low': lows
                    })

                    # Use enhanced market regime detection with ADX
                    regime_info = self.market_regime_detector.analyze_regime(df)

                    # Apply regime filtering - only consider tickers with acceptable regimes
                    if self._is_acceptable_regime(regime_info.get('regime', 'UNKNOWN')):
                        self.market_regimes[ticker] = regime_info
                    else:
                        # Still store but mark as filtered
                        self.market_regimes[ticker] = {
                            **regime_info,
                            'filtered': True,
                            'filter_reason': f"Unfavorable regime: {regime_info.get('regime', 'UNKNOWN')}"
                        }

        # Apply regime filtering to base signals
        filtered_signals = self._get_regime_filtered_signals(base_signals['signals'])
        risk_enhanced_signals = []

        print(f"🔍 Regime filtering: {len(base_signals['signals'])} → {len(filtered_signals)} signals")

        for signal in filtered_signals:
            # Skip regime filtered signals (but we could still track them)
            if signal.get('regime_filtered', False):
                risk_enhanced_signals.append(signal)
                continue

            # Enhance signal with comprehensive risk metrics
            enhanced_signal = self._enhance_with_risk_metrics(signal, capital)

            # Calculate position details with ATR-based stops
            current_price = enhanced_signal['current_price']
            atr_value = enhanced_signal['atr_value']

            stop_loss = self.risk_manager.calculate_optimal_stop_loss(
                current_price, atr_value, method='atr'
            )

            take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 risk/reward

            # Validate risk-reward
            risk_reward_valid = self.risk_manager.validate_trade_risk_reward(
                current_price, stop_loss, take_profit
            )

            # Check correlation risk with existing positions
            correlation_risk = self.risk_manager.check_correlation_risk(
                signal['ticker'], self.portfolio_positions
            )

            # Adjust signal based on market regime (already partially done in enhancement)
            regime_adjustment = self._get_regime_adjustment(
                enhanced_signal['regime'], enhanced_signal['final_action']
            )
            adjusted_score = signal['combined_score'] * regime_adjustment

            # Enhanced final action with all risk considerations
            final_action = self._determine_risk_adjusted_action(
                enhanced_signal['final_action'],
                adjusted_score,
                risk_reward_valid,
                correlation_risk['is_high_correlation'],
                enhanced_signal['regime']
            )

            # Update enhanced signal with final risk calculations
            enhanced_signal.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': self.risk_manager.calculate_risk_reward_ratio(
                    current_price, stop_loss, take_profit
                ),
                'correlation_risk': correlation_risk,
                'risk_adjusted_score': adjusted_score,
                'risk_adjusted_action': final_action,
                'max_position_size': self.risk_manager.calculate_max_position_size(capital),
                'portfolio_heat_check': self._check_portfolio_heat(capital),
                'risk_recommendations': self._generate_risk_recommendations(
                    final_action,
                    enhanced_signal['kelly_fraction'],
                    correlation_risk,
                    enhanced_signal['regime']
                ),
                'risk_reward_valid': risk_reward_valid,
                'regime_adjustment_factor': regime_adjustment
            })

            risk_enhanced_signals.append(enhanced_signal)

        # Sort by risk-adjusted score
        risk_enhanced_signals.sort(key=lambda x: x.get('risk_adjusted_score', x.get('combined_score', 0)), reverse=True)

        # Calculate summary statistics
        filtered_count = sum(1 for s in risk_enhanced_signals if s.get('regime_filtered', False))
        accepted_count = len(risk_enhanced_signals) - filtered_count

        return {
            'signals': risk_enhanced_signals,
            'ml_model_count': len(self.ml_models),
            'rl_trained': self.rl_agent is not None,
            'risk_management_enabled': True,
            'regime_filtering_enabled': True,
            'portfolio_summary': self._get_portfolio_summary(capital),
            'filtering_summary': {
                'total_signals': len(base_signals['signals']),
                'filtered_by_regime': filtered_count,
                'accepted_signals': accepted_count,
                'acceptance_rate': accepted_count / len(base_signals['signals']) * 100 if base_signals['signals'] else 0
            },
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_atr(self, ticker: str, period: int = 14) -> float:
        """Calculate Average True Range for a ticker."""
        if ticker not in self.price_data:
            return 0.0

        prices = self.price_data[ticker]
        if len(prices) < period:
            return 0.0

        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]['high']
            low = prices[i]['low']
            prev_close = prices[i-1]['close']

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _get_regime_adjustment(self, regime: str, action: str) -> float:
        """Get regime-based adjustment factor with enhanced filtering."""
        # Enhanced regime adjustments with ADX-based filtering
        adjustments = {
            # Very Strong Trend (ADX > 60)
            ('VERY_STRONG_TREND', 'STRONG BUY'): 1.3,
            ('VERY_STRONG_TREND', 'BUY'): 1.2,
            ('VERY_STRONG_TREND', 'SELL'): 0.6,  # Avoid counter-trend trading

            # Strong Trend (ADX 40-60)
            ('STRONG_TREND', 'STRONG BUY'): 1.2,
            ('STRONG_TREND', 'BUY'): 1.1,
            ('STRONG_TREND', 'SELL'): 0.7,  # Avoid counter-trend trading

            # Emerging Strong Trend (ADX 25-40)
            ('EMERGING_STRONG_TREND', 'STRONG BUY'): 1.1,
            ('EMERGING_STRONG_TREND', 'BUY'): 1.0,
            ('EMERGING_STRONG_TREND', 'SELL'): 0.8,

            # Weak Trend (ADX 20-25)
            ('WEAK_TREND', 'BUY'): 0.9,
            ('WEAK_TREND', 'SELL'): 0.9,
            ('WEAK_TREND', 'STRONG BUY'): 0.8,  # Downgrade strong signals

            # Consolidation (ADX < 20)
            ('CONSOLIDATION', 'BUY'): 0.6,
            ('CONSOLIDATION', 'SELL'): 0.6,
            ('CONSOLIDATION', 'STRONG BUY'): 0.5,  # Significant downgrade
            ('CONSOLIDATION', 'HOLD'): 1.2,  # Favor holding in consolidation
        }

        return adjustments.get((regime, action), 1.0)

    def _determine_risk_adjusted_action(self, base_action: str, score: float,
                                      risk_reward_valid: bool, high_correlation: bool,
                                      regime: str) -> str:
        """Determine final action considering all risk factors."""
        # Reduce position size if risk-reward is poor
        if not risk_reward_valid:
            if base_action in ['STRONG BUY', 'BUY']:
                return 'HOLD'
            elif base_action == 'STRONG BUY':
                return 'BUY'

        # Avoid correlated positions
        if high_correlation and base_action in ['STRONG BUY', 'BUY']:
            return 'HOLD'

        # Avoid trades in unfavorable market regimes
        if regime == 'CONSOLIDATION' and base_action in ['STRONG BUY', 'BUY']:
            if score < 60:  # Only strong signals in consolidation
                return 'HOLD'
            else:
                return 'BUY'  # Downgrade from STRONG BUY

        return base_action

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

    def _generate_risk_recommendations(self, action: str, kelly_fraction: float,
                                     correlation_risk: Dict, regime: str) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        if kelly_fraction > 0.25:
            recommendations.append(f"⚠️ High Kelly fraction ({kelly_fraction:.2%}) - consider reducing position size")
        elif kelly_fraction < 0.05:
            recommendations.append(f"⚠️ Very low Kelly fraction ({kelly_fraction:.2%}) - consider avoiding this trade")

        if correlation_risk['is_high_correlation']:
            recommendations.append(f"⚠️ High correlation with existing positions: {correlation_risk['correlated_positions']}")

        if regime == 'CONSOLIDATION':
            recommendations.append("📊 Market in consolidation - reduce position sizes")
        elif regime in ['STRONG_TREND', 'VERY_STRONG_TREND']:
            recommendations.append(f"📈 Strong {regime.replace('_', ' ').lower()} detected - favorable for trend following")

        if action == 'HOLD':
            recommendations.append("🔄 Risk management suggests holding - signal strength insufficient")

        return recommendations

    def _is_acceptable_regime(self, regime: str) -> bool:
        """Check if a market regime is acceptable for trading."""
        # Define acceptable regimes for different strategies
        acceptable_regimes = {
            'STRONG_BUY': ['VERY_STRONG_TREND', 'STRONG_TREND', 'EMERGING_STRONG_TREND'],
            'BUY': ['STRONG_TREND', 'EMERGING_STRONG_TREND', 'WEAK_TREND'],
            'SELL': ['VERY_STRONG_TREND', 'STRONG_TREND', 'EMERGING_STRONG_TREND'],  # Only sell in strong downtrends
            'HOLD': ['CONSOLIDATION', 'WEAK_TREND']
        }

        # For general filtering, allow strong and emerging trends
        generally_acceptable = ['VERY_STRONG_TREND', 'STRONG_TREND', 'EMERGING_STRONG_TREND']

        return regime in generally_acceptable

    def _get_regime_filtered_signals(self, signals: List[Dict]) -> List[Dict]:
        """Apply regime filtering to trading signals."""
        filtered_signals = []

        for signal in signals:
            ticker = signal['ticker']
            regime_info = self.market_regimes.get(ticker, {})

            # Skip if signal is filtered due to regime
            if regime_info.get('filtered', False):
                signal['regime_filtered'] = True
                signal['regime_filter_reason'] = regime_info.get('filter_reason', 'Unknown')
                continue

            # Apply regime-based signal adjustment
            regime = regime_info.get('regime', 'UNKNOWN')
            base_action = signal['final_action']

            # Filter out weak signals in unfavorable regimes
            if regime == 'CONSOLIDATION' and base_action in ['STRONG BUY', 'BUY']:
                if signal['combined_score'] < 70:  # Only very strong signals
                    signal['regime_filtered'] = True
                    signal['regime_filter_reason'] = f"Weak signal in {regime}"
                    continue
                else:
                    signal['final_action'] = 'BUY'  # Downgrade from STRONG BUY

            # Avoid counter-trend trading in strong trends
            if regime in ['VERY_STRONG_TREND', 'STRONG_TREND']:
                # This would need trend direction, simplified for now
                pass

            filtered_signals.append(signal)

        return filtered_signals

    def _enhance_with_risk_metrics(self, signal: Dict, capital: float) -> Dict:
        """Enhance signal with comprehensive risk metrics."""
        ticker = signal['ticker']

        # Calculate additional risk metrics
        current_price = signal['current_price']
        atr_value = self._calculate_atr(ticker)

        # Enhanced position sizing with Kelly Criterion
        historical_win_rate = self.ml_models[ticker]['accuracy'] if ticker in self.ml_models else 0.5
        ml_confidence = signal['ml_confidence']
        rl_confidence = signal.get('rl_confidence', 0)

        # Calculate Kelly fraction with confidence adjustment
        kelly_fraction = self.risk_manager.calculate_kelly_with_confidence(
            ml_confidence, rl_confidence, historical_win_rate
        )

        # Apply regime-based Kelly adjustment
        regime = self.market_regimes.get(ticker, {}).get('regime', 'UNKNOWN')
        kelly_fraction *= self._get_regime_adjustment(regime, signal['final_action'])

        # Calculate position size with risk limits
        position_size = self.risk_manager.calculate_position_size(
            capital,
            min(0.02, kelly_fraction),  # Use Kelly fraction but cap at 2%
            current_price,
            self.risk_manager.calculate_optimal_stop_loss(current_price, atr_value)
        )

        # Add enhanced risk metrics
        signal.update({
            'kelly_fraction': kelly_fraction,
            'position_size': position_size,
            'atr_value': atr_value,
            'regime_adjusted_kelly': kelly_fraction,
            'risk_score': self._calculate_risk_score(signal, kelly_fraction, regime),
            'regime': regime,
            'regime_strength': self.market_regimes.get(ticker, {}).get('strength', 0)
        })

        return signal

    def _calculate_risk_score(self, signal: Dict, kelly_fraction: float, regime: str) -> float:
        """Calculate comprehensive risk score (0-100, lower is better)."""
        risk_score = 50  # Base risk score

        # Kelly fraction risk (0-20 points)
        if kelly_fraction > 0.25:
            risk_score += 20
        elif kelly_fraction > 0.15:
            risk_score += 10
        elif kelly_fraction < 0.05:
            risk_score += 15  # Very low Kelly suggests poor opportunity

        # Regime risk (0-15 points)
        regime_risk_map = {
            'VERY_STRONG_TREND': 0,
            'STRONG_TREND': 2,
            'EMERGING_STRONG_TREND': 5,
            'WEAK_TREND': 10,
            'CONSOLIDATION': 15,
            'UNKNOWN': 20
        }
        risk_score += regime_risk_map.get(regime, 15)

        # Model accuracy risk (0-10 points)
        accuracy = signal.get('ml_accuracy', 0.5)
        if accuracy < 0.6:
            risk_score += 10
        elif accuracy < 0.7:
            risk_score += 5

        # Signal strength risk (0-5 points)
        combined_score = signal.get('combined_score', 0)
        if combined_score < 30:
            risk_score += 5

        return min(100, risk_score)

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

    def simulate_trade_execution(self, signal: Dict, capital: float) -> Dict:
        """Simulate trade execution with risk management."""
        try:
            ticker = signal['ticker']
            action = signal['risk_adjusted_action']

            if action not in ['BUY', 'STRONG BUY']:
                return {'status': 'skipped', 'reason': f'Action {action} not suitable for buying'}

            # Get position size
            position_info = signal['position_size']
            shares = position_info['shares']

            if shares <= 0:
                return {'status': 'skipped', 'reason': 'Position size too small'}

            # Simulate execution
            entry_price = signal['current_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']

            # Create position record
            position = {
                'ticker': ticker,
                'shares': shares,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'current_price': entry_price,
                'position_value': shares * entry_price,
                'position_risk': shares * (entry_price - stop_loss),
                'unrealized_pnl': 0,
                'entry_time': datetime.now(),
                'regime_at_entry': signal.get('market_regime', 'UNKNOWN'),
                'kelly_fraction': signal.get('kelly_fraction', 0),
                'ml_confidence': signal.get('ml_confidence', 0),
                'rl_confidence': signal.get('rl_confidence', 0)
            }

            # Add to portfolio
            self.portfolio_positions.append(position)

            # Update cash
            used_capital = position['position_value']
            remaining_capital = capital - used_capital

            return {
                'status': 'executed',
                'position': position,
                'used_capital': used_capital,
                'remaining_capital': remaining_capital,
                'execution_time': datetime.now().isoformat()
            }

        except Exception as e:
            return {'status': 'error', 'reason': str(e)}


def main():
    """Main function to run the advanced ML/RL trading system with risk management."""
    print("🤖 ADVANCED ML/RL TRADING SYSTEM WITH RISK MANAGEMENT")
    print("=" * 70)

    system = AdvancedMLTradingSystem()

    # Load and prepare data
    if not system.load_historical_data():
        print("❌ Cannot load data. Exiting.")
        return

    # Train ML models
    system.train_ml_models()

    # Train RL agent
    system.train_rl_agent(episodes=500)  # Reduced for faster execution

    # Generate risk-aware signals
    capital = 100000  # 100k PLN starting capital
    risk_aware_results = system.generate_risk_aware_signals(capital)

    # Display results
    print(f"\n🎯 RISK-AWARE ML/RL TRADING SIGNALS")
    print("=" * 90)
    print(f"🤖 ML Models: {risk_aware_results['ml_model_count']}")
    print(f"🧠 RL Agent: {'Trained' if risk_aware_results['rl_trained'] else 'Not trained'}")
    print(f"🛡️ Risk Management: {'Enabled' if risk_aware_results['risk_management_enabled'] else 'Disabled'}")
    print(f"💰 Starting Capital: {capital:,.0f} PLN")
    print("=" * 90)

    # Display portfolio summary
    portfolio_summary = risk_aware_results['portfolio_summary']
    print(f"\n📊 PORTFOLIO SUMMARY:")
    print(f"   💰 Available Capital: {portfolio_summary['cash_available']:,.0f} PLN")
    print(f"   📈 Portfolio Heat: {portfolio_summary['portfolio_heat']:.1f}%")
    print(f"   🔥 Max Heat Allowed: {system.risk_manager.max_portfolio_heat:.1f}%")
    print(f"   📊 Open Positions: {portfolio_summary['total_positions']}")

    # Display top risk-aware signals
    signals = risk_aware_results['signals'][:10]  # Top 10

    buy_signals = [s for s in signals if s['risk_adjusted_action'] in ['STRONG BUY', 'BUY']]
    hold_signals = [s for s in signals if s['risk_adjusted_action'] == 'HOLD']
    sell_signals = [s for s in signals if s['risk_adjusted_action'] == 'SELL']

    if buy_signals:
        print(f"\n🟢 RISK-AWARE BUY SIGNALS ({len(buy_signals)}):")
        print("-" * 100)
        for signal in buy_signals[:5]:
            print(f"🚀 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['risk_adjusted_score']:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                  f"RR: {signal['risk_reward_ratio']:.1f} | Regime: {signal['market_regime'][:4]} | "
                  f"{signal['name'][:25]}")

    if hold_signals:
        print(f"\n🟡 RISK-AWARE HOLD SIGNALS ({len(hold_signals)}):")
        print("-" * 100)
        for signal in hold_signals[:3]:
            print(f"⏸️  {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['risk_adjusted_score']:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                  f"RR: {signal['risk_reward_ratio']:.1f} | Regime: {signal['market_regime'][:4]} | "
                  f"{signal['name'][:25]}")

    if sell_signals:
        print(f"\n🔴 RISK-AWARE SELL SIGNALS ({len(sell_signals)}):")
        print("-" * 100)
        for signal in sell_signals[:3]:
            print(f"📉 {signal['ticker']:<8} | {signal['current_price']:>8.2f} PLN | "
                  f"Score: {signal['risk_adjusted_score']:>6.1f} | Kelly: {signal['kelly_fraction']:.2%} | "
                  f"RR: {signal['risk_reward_ratio']:.1f} | Regime: {signal['market_regime'][:4]} | "
                  f"{signal['name'][:25]}")

    # Detailed analysis for top signal with trade execution simulation
    if signals:
        top_signal = signals[0]
        print(f"\n🔍 TOP RISK-AWARE SIGNAL ANALYSIS:")
        print(f"   📊 {top_signal['ticker']} - {top_signal['name']}")
        print(f"   💰 Current Price: {top_signal['current_price']:.2f} PLN")
        print(f"   🎯 Risk-Adjusted Action: {top_signal['risk_adjusted_action']}")
        print(f"   📈 Risk-Adjusted Score: {top_signal['risk_adjusted_score']:.1f}")
        print(f"   🛡️ Kelly Fraction: {top_signal['kelly_fraction']:.2%}")
        print(f"   📊 Risk/Reward Ratio: {top_signal['risk_reward_ratio']:.2f}")
        print(f"   🎯 Stop Loss: {top_signal['stop_loss']:.2f} PLN")
        print(f"   🎯 Take Profit: {top_signal['take_profit']:.2f} PLN")
        print(f"   📊 Market Regime: {top_signal['market_regime']}")
        print(f"   🤖 ML Confidence: {top_signal['ml_confidence']:.3f}")
        print(f"   🧠 RL Confidence: {top_signal['rl_confidence']:.3f}")
        print(f"   📈 Position Size: {top_signal['position_size']['shares']:.0f} shares")
        print(f"   💰 Position Value: {top_signal['position_size']['position_value']:,.0f} PLN")

        # Risk recommendations
        if top_signal['risk_recommendations']:
            print(f"\n   ⚠️  RISK RECOMMENDATIONS:")
            for rec in top_signal['risk_recommendations']:
                print(f"      {rec}")

        # Simulate trade execution for top signal
        print(f"\n🔄 SIMULATING TRADE EXECUTION:")
        execution_result = system.simulate_trade_execution(top_signal, capital)

        if execution_result['status'] == 'executed':
            position = execution_result['position']
            print(f"   ✅ Trade Executed Successfully")
            print(f"   📊 Position: {position['shares']:.0f} shares of {position['ticker']}")
            print(f"   💰 Used Capital: {execution_result['used_capital']:,.0f} PLN")
            print(f"   💵 Remaining Capital: {execution_result['remaining_capital']:,.0f} PLN")
            print(f"   🛡️ Stop Loss: {position['stop_loss']:.2f} PLN")
            print(f"   🎯 Take Profit: {position['take_profit']:.2f} PLN")
        else:
            print(f"   ❌ Trade Skipped: {execution_result['reason']}")

    # Save risk-aware results
    with open('risk_aware_trading_signals.json', 'w') as f:
        json.dump(risk_aware_results, f, indent=2, default=str)

    # Save models
    system.save_models()

    print(f"\n💾 Risk-aware results saved to: risk_aware_trading_signals.json")
    print(f"🏁 Advanced ML/RL with Risk Management analysis complete!")


if __name__ == "__main__":
    main()