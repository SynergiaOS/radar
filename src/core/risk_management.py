#!/usr/bin/env python3
"""
Risk Management Module
Implements Kelly Criterion position sizing, trailing stop-loss, and comprehensive risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config import MAX_POSITION_SIZE_PCT, MAX_PORTFOLIO_HEAT, MIN_RISK_REWARD_RATIO
from trading_chart_service import chart_service

class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self):
        self.kelly_fraction = 0.25  # Quarter-Kelly for conservative approach
        self.max_position_pct = MAX_POSITION_SIZE_PCT
        self.max_portfolio_heat = MAX_PORTFOLIO_HEAT
        self.min_risk_reward_ratio = MIN_RISK_REWARD_RATIO
        self.trailing_stop_atr_multiplier = 2.0

    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Kelly fraction (0.0 to 1.0)

        Example:
        >>> risk = RiskManager()
        >>> kelly = risk.calculate_kelly_criterion(0.6, 100, 50)
        >>> print(f"Kelly fraction: {kelly:.2%}")
        Kelly fraction: 20.00%
        """
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0

        # Kelly formula: f = (bp * p - q) / b
        # where: f = fraction of bankroll to bet
        # p = probability of winning
        # q = probability of losing (1 - p)
        # b = odds received on the bet (b to 1)
        # In trading: b = avg_win / avg_loss

        win_loss_ratio = avg_win / avg_loss
        q = 1 - win_rate

        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio

        # Apply safety factor (quarter-Kelly)
        return max(0, kelly_fraction * self.kelly_fraction)

    def calculate_kelly_with_confidence(self, ml_confidence: float, rl_confidence: float,
                                      historical_win_rate: float, avg_win: float = 100,
                                      avg_loss: float = 50) -> float:
        """
        Calculate Kelly fraction weighted by ML/RL confidence

        Args:
            ml_confidence: ML model confidence (0.0 to 1.0)
            rl_confidence: RL agent confidence (0.0 to 1.0)
            historical_win_rate: Historical win rate
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Weighted Kelly fraction
        """
        # Calculate base Kelly from historical data
        base_kelly = self.calculate_kelly_criterion(historical_win_rate, avg_win, avg_loss)

        # Weight by model confidence
        combined_confidence = (ml_confidence + rl_confidence) / 2

        # Adjust Kelly based on confidence
        adjusted_kelly = base_kelly * combined_confidence

        return max(0, min(adjusted_kelly, self.kelly_fraction))

    def calculate_trailing_stop_atr(self, current_price: float, atr_value: float,
                                     multiplier: float = None) -> float:
        """
        Calculate trailing stop-loss based on ATR

        Args:
            current_price: Current stock price
            atr_value: ATR value
            multiplier: ATR multiplier (default: self.trailing_stop_atr_multiplier)

        Returns:
            Trailing stop-loss price

        Example:
        >>> risk = RiskManager()
        >>> stop = risk.calculate_trailing_stop_atr(100, 5, 2.0)
        >>> print(f"Trailing stop: {stop}")
        Trailing stop: 90.0
        """
        if multiplier is None:
            multiplier = self.trailing_stop_atr_multiplier

        if atr_value <= 0:
            return current_price

        return current_price - (atr_value * multiplier)

    def calculate_trailing_stop_percentage(self, entry_price: float, current_price: float,
                                         trailing_pct: float = 0.05) -> float:
        """
        Calculate percentage-based trailing stop-loss

        Args:
            entry_price: Entry price
            current_price: Current price
            trailing_pct: Trailing percentage (default: 5%)

        Returns:
            Trailing stop-loss price
        """
        if current_price <= entry_price:
            return entry_price * (1 - trailing_pct)

        # Calculate stop as percentage below highest price
        highest_price = current_price
        stop_price = highest_price * (1 - trailing_pct)

        return max(stop_price, entry_price * (1 - trailing_pct))

    def update_trailing_stop(self, current_stop: float, current_price: float,
                             highest_price: float, atr_value: float) -> float:
        """
        Update trailing stop-loss as price moves favorably

        Args:
            current_stop: Current stop-loss price
            current_price: Current stock price
            highest_price: Highest price since entry
            atr_value: ATR value

        Returns:
            Updated stop-loss price
        """
        # Calculate new potential stop
        new_stop = self.calculate_trailing_stop_atr(highest_price, atr_value)

        # Only move stop up (not down)
        return max(current_stop, new_stop)

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float,
                                  take_profit: float) -> float:
        """
        Calculate risk-reward ratio

        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price

        Returns:
            Risk-reward ratio

        Example:
        >>> risk = RiskManager()
        >>> ratio = risk.calculate_risk_reward_ratio(100, 95, 110)
        >>> print(f"Risk/Reward: {ratio:.2f}")
        Risk/Reward: 3.00
        """
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)

        if risk_amount == 0:
            return 0

        return reward_amount / risk_amount

    def validate_trade_risk_reward(self, entry_price: float, stop_loss: float,
                                   take_profit: float, min_ratio: float = None) -> bool:
        """
        Validate if trade meets minimum risk-reward criteria

        Args:
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            min_ratio: Minimum required ratio

        Returns:
            True if trade meets criteria
        """
        if min_ratio is None:
            min_ratio = self.min_risk_reward_ratio

        ratio = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        return ratio >= min_ratio

    def calculate_position_size(self, capital: float, risk_per_trade_pct: float,
                              entry_price: float, stop_loss: float) -> Dict:
        """
        Calculate position size based on risk parameters

        Args:
            capital: Total available capital
            risk_per_trade_pct: Risk percentage per trade (e.g., 0.02 for 2%)
            entry_price: Entry price
            stop_loss: Stop-loss price

        Returns:
            Dictionary with position details
        """
        risk_amount = capital * risk_per_trade_pct
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance == 0:
            return {
                'shares': 0,
                'position_value': 0,
                'risk_amount': 0,
                'max_loss': 0
            }

        shares = risk_amount / stop_distance
        position_value = shares * entry_price
        max_loss = shares * stop_distance

        return {
            'shares': shares,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'max_loss': max_loss
        }

    def calculate_max_position_size(self, capital: float, max_position_pct: float = None) -> float:
        """
        Calculate maximum allowed position size

        Args:
            capital: Total available capital
            max_position_pct: Maximum position percentage

        Returns:
            Maximum position value
        """
        if max_position_pct is None:
            max_position_pct = self.max_position_pct

        return capital * (max_position_pct / 100)

    def calculate_portfolio_heat(self, open_positions: List[Dict], capital: float) -> Dict:
        """
        Calculate portfolio risk exposure (heat)

        Args:
            open_positions: List of open position dictionaries
            capital: Total capital

        Returns:
            Portfolio heat statistics
        """
        total_risk = 0
        position_details = []

        for position in open_positions:
            shares = position.get('shares', 0)
            current_price = position.get('current_price', 0)
            stop_loss = position.get('stop_loss', 0)

            if shares > 0 and current_price > 0 and stop_loss > 0:
                position_value = shares * current_price
                position_risk = shares * (current_price - stop_loss)
                total_risk += position_risk

                position_details.append({
                    'ticker': position.get('ticker', ''),
                    'position_value': position_value,
                    'position_risk': position_risk,
                    'risk_pct': (position_risk / capital) * 100
                })

        portfolio_heat_pct = (total_risk / capital) * 100

        return {
            'total_risk': total_risk,
            'portfolio_heat_pct': portfolio_heat_pct,
            'max_allowed_heat': self.max_portfolio_heat,
            'is_overheated': portfolio_heat_pct > self.max_portfolio_heat,
            'position_details': position_details
        }

    def check_correlation_risk(self, ticker: str, existing_positions: List[Dict],
                           max_correlation: float = 0.7) -> Dict:
        """
        Check correlation risk with existing positions

        Args:
            ticker: New ticker symbol
            existing_positions: List of existing positions
            max_correlation: Maximum allowed correlation

        Returns:
            Correlation risk assessment
        """
        # Simplified correlation check based on industry sectors
        # In real implementation, would use historical correlation data

        sector_correlations = {
            'banking': ['PKO.WA', 'PEO.WA', 'SPL.WA', 'MBK.WA', 'BZWB.WA'],
            'energy': ['PKN.WA', 'PGE.WA', 'TPE.WA', 'KGH.WA', 'JSW.WA'],
            'telecom': ['CPS.WA', 'OPL.WA'],
            'retail': ['CCC.WA', 'DNP.WA', 'LPP.WA'],
            'tech': ['CDR.WA', '11B.WA'],
            'insurance': ['PZU.WA']
        }

        ticker_sector = None
        for sector, tickers in sector_correlations.items():
            if ticker in tickers:
                ticker_sector = sector
                break

        correlated_positions = []
        for position in existing_positions:
            pos_ticker = position.get('ticker', '')
            pos_sector = None

            for sector, tickers in sector_correlations.items():
                if pos_ticker in tickers:
                    pos_sector = sector
                    break

            if ticker_sector and pos_sector and ticker_sector == pos_sector:
                correlated_positions.append(pos_ticker)

        return {
            'ticker': ticker,
            'sector': ticker_sector,
            'correlated_positions': correlated_positions,
            'correlation_count': len(correlated_positions),
            'max_correlation': max_correlation,
            'is_high_correlation': len(correlated_positions) > 0
        }

    def calculate_optimal_stop_loss(self, entry_price: float, volatility: float,
                                   method: str = 'atr') -> float:
        """
        Calculate optimal stop-loss based on different methods

        Args:
            entry_price: Entry price
            volatility: Volatility measure (ATR or standard deviation)
            method: Method to use ('atr', 'percentage', 'volatility')

        Returns:
            Optimal stop-loss price
        """
        if method == 'atr':
            return self.calculate_trailing_stop_atr(entry_price, volatility)
        elif method == 'percentage':
            return self.calculate_trailing_stop_percentage(entry_price, entry_price, 0.05)
        elif method == 'volatility':
            # Use 2x volatility as stop distance
            return entry_price - (2 * volatility)
        else:
            return entry_price * 0.95  # Default 5% stop

    def generate_risk_report(self, trade_setup: Dict) -> Dict:
        """
        Generate comprehensive risk report for a trade setup

        Args:
            trade_setup: Dictionary with trade parameters

        Returns:
            Risk assessment report
        """
        entry_price = trade_setup.get('entry_price', 0)
        stop_loss = trade_setup.get('stop_loss', 0)
        take_profit = trade_setup.get('take_profit', 0)
        position_size = trade_setup.get('position_size', 0)

        risk_reward = self.calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        is_valid_risk_reward = self.validate_trade_risk_reward(entry_price, stop_loss, take_profit)

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': risk_reward,
            'is_valid_risk_reward': is_valid_risk_reward,
            'stop_distance_pct': ((entry_price - stop_loss) / entry_price) * 100,
            'target_distance_pct': ((take_profit - entry_price) / entry_price) * 100,
            'risk_per_trade': trade_setup.get('risk_per_trade_pct', 0),
            'recommendations': self._generate_risk_recommendations(risk_reward, is_valid_risk_reward)
        }

    def _generate_risk_recommendations(self, risk_reward: float, is_valid: bool) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        if not is_valid:
            recommendations.append(f"❌ Risk/reward ratio {risk_reward:.2f} below minimum threshold")
            recommendations.append("Consider improving entry or adjusting stop/target levels")
        else:
            recommendations.append(f"✅ Risk/reward ratio {risk_reward:.2f} meets minimum threshold")

        if risk_reward < 1:
            recommendations.append("⚠️ Risk exceeds potential reward - consider avoiding this trade")
        elif risk_reward < 2:
            recommendations.append("⚠️ Low risk-reward ratio - ensure high confidence in setup")

        return recommendations


def get_sector_for_ticker(ticker: str) -> str:
    """
    Get sector name for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'PKN.WA')

    Returns:
        Sector name or 'Other' if not found
    """
    sector_correlations = {
        'banking': ['PKO.WA', 'PEO.WA', 'SPL.WA', 'MBK.WA', 'BZWB.WA'],
        'energy': ['PKN.WA', 'PGE.WA', 'TPE.WA', 'KGH.WA', 'JSW.WA'],
        'telecom': ['CPS.WA', 'OPL.WA'],
        'insurance': ['PZU.WA', 'ALR.WA'],
        'retail': ['LPP.WA', 'CCC.WA'],
        'technology': ['CDR.WA', 'TXT.WA', '11B.WA'],
        'finance': ['XTB.WA', 'KTY.WA'],
        'industrial': ['KRU.WA', 'SNT.WA', 'RBW.WA']
    }

    for sector, tickers in sector_correlations.items():
        if ticker in tickers:
            return sector.capitalize()

    return 'Other'


# Utility functions for standalone usage
def calculate_kelly_position_size(capital: float, win_rate: float, avg_win: float,
                              avg_loss: float, kelly_fraction: float = 0.25) -> float:
    """
    Standalone function to calculate Kelly position size

    Args:
        capital: Available capital
        win_rate: Historical win rate
        avg_win: Average win amount
        avg_loss: Average loss amount
        kelly_fraction: Safety factor (default: 0.25 for quarter-Kelly)

    Returns:
        Recommended position size percentage
    """
    risk = RiskManager()
    kelly = risk.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    return kelly * kelly_fraction * 100


def calculate_trailing_stop(current_price: float, atr: float, multiplier: float = 2.0) -> float:
    """
    Standalone function to calculate trailing stop

    Args:
        current_price: Current price
        atr: ATR value
        multiplier: ATR multiplier

    Returns:
        Trailing stop price
    """
    risk = RiskManager()
    return risk.calculate_trailing_stop_atr(current_price, atr, multiplier)


if __name__ == "__main__":
    # Example usage
    risk = RiskManager()

    # Kelly Criterion example
    win_rate = 0.6
    avg_win = 100
    avg_loss = 50
    kelly = risk.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    print(f"Kelly Criterion: {kelly:.2%}")

    # Trailing stop example
    price = 100.0
    atr = 5.0
    stop = risk.calculate_trailing_stop_atr(price, atr)
    print(f"Trailing Stop: {stop}")

    # Risk-reward example
    entry = 100.0
    stop_loss = 95.0
    take_profit = 110.0
    ratio = risk.calculate_risk_reward_ratio(entry, stop_loss, take_profit)
    print(f"Risk/Reward Ratio: {ratio:.2f}")

    # Position sizing example
    capital = 100000
    risk_pct = 0.02
    position = risk.calculate_position_size(capital, risk_pct, entry, stop_loss)
    print(f"Position Details: {position}")