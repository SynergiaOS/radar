# -*- coding: utf-8 -*-
"""
Risk Management Service - Zarządzanie ryzykiem i pozycjami
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Position:
    """Klasa reprezentująca pozycję inwestycyjną"""
    ticker: str
    shares: int
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_date: datetime
    last_updated: datetime

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.shares

    @property
    def is_at_stop_loss(self) -> bool:
        return self.current_price <= self.stop_loss

    @property
    def is_at_take_profit(self) -> bool:
        return self.current_price >= self.take_profit

@dataclass
class RiskMetrics:
    """Metryki ryzyka portfolio"""
    total_capital: float
    invested_capital: float
    cash_available: float
    total_positions: int
    portfolio_heat: float
    max_position_size: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    beta: float

class RiskManagementService:
    """Service for risk management and position sizing"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.positions: List[Position] = []
        self.transaction_history: List[Dict] = []

    def _default_config(self) -> Dict:
        """Domyślna konfiguracja risk management"""
        return {
            'max_position_size': 0.02,      # 2% kapitału na pozycję
            'max_portfolio_heat': 0.20,    # 20% maksymalne zaangażowanie
            'stop_loss_pct': 0.05,         # 5% stop loss
            'take_profit_pct': 0.10,       # 10% take profit
            'rebalance_threshold': 0.05,   # 5% próg rebalancingu
            'max_positions': 10,           # Maksymalna liczba pozycji
            'min_position_size': 1000,     # Minimalna wielkość pozycji w PLN
            'risk_free_rate': 0.05,        # 5% stopa wolna od ryzyka
            'volatility_lookback': 20      # Okres do obliczania zmienności
        }

    def calculate_position_size(self, ticker: str, entry_price: float, volatility: float = None) -> int:
        """Oblicza wielkość pozycji na podstawie zarządzania ryzykiem"""
        total_capital = self.config['max_position_size'] * self._get_portfolio_value()
        risk_per_share = entry_price * self.config['stop_loss_pct']

        # Dostosuj rozmiar pozycji do zmienności
        if volatility:
            volatility_adjustment = min(1.0, 0.15 / volatility)  # Mniejsza pozycja przy wyższej zmienności
            total_capital *= volatility_adjustment

        shares = int(total_capital / entry_price)
        risk_amount = shares * risk_per_share

        # Sprawdź minimalną wielkość pozycji
        position_value = shares * entry_price
        if position_value < self.config['min_position_size']:
            return 0

        # Sprawdź czy ryzyko nie jest zbyt duże
        max_risk = self._get_portfolio_value() * 0.01  # Maksymalnie 1% ryzyka na pozycję
        if risk_amount > max_risk:
            shares = int(max_risk / risk_per_share)

        return shares

    def add_position(self, ticker: str, shares: int, entry_price: float,
                    stop_loss: float = None, take_profit: float = None) -> bool:
        """Dodaje nową pozycję do portfolio"""
        try:
            # Ustaw domyślne stop loss/take profit
            if stop_loss is None:
                stop_loss = entry_price * (1 - self.config['stop_loss_pct'])
            if take_profit is None:
                take_profit = entry_price * (1 + self.config['take_profit_pct'])

            # Sprawdź ograniczenia
            if not self._can_add_position(shares, entry_price):
                return False

            # Sprawdź czy nie mamy już pozycji w tym tickerze
            existing_position = self.get_position(ticker)
            if existing_position:
                return self._update_position(existing_position, shares, entry_price)

            # Stwórz nową pozycję
            position = Position(
                ticker=ticker,
                shares=shares,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_date=datetime.now(),
                last_updated=datetime.now()
            )

            self.positions.append(position)

            # Zapisz transakcję
            self._record_transaction('BUY', ticker, shares, entry_price)

            return True

        except Exception as e:
            print(f"Błąd dodawania pozycji {ticker}: {e}")
            return False

    def update_position_price(self, ticker: str, current_price: float) -> bool:
        """Aktualizuje cenę pozycji"""
        position = self.get_position(ticker)
        if not position:
            return False

        position.current_price = current_price
        position.last_updated = datetime.now()

        # Sprawdź czy pozycja osiągnęła stop loss lub take profit
        if position.is_at_stop_loss:
            self._close_position(position, 'STOP_LOSS')
        elif position.is_at_take_profit:
            self._close_position(position, 'TAKE_PROFIT')

        return True

    def close_position(self, ticker: str, reason: str = 'MANUAL') -> bool:
        """Zamyka pozycję"""
        position = self.get_position(ticker)
        if not position:
            return False

        return self._close_position(position, reason)

    def _close_position(self, position: Position, reason: str) -> bool:
        """Wewnętrzna metoda zamykania pozycji"""
        # Zapisz transakcję
        self._record_transaction('SELL', position.ticker, position.shares,
                                position.current_price, reason)

        # Usuń pozycję
        self.positions.remove(position)

        print(f"🔴 Pozycja zamknięta: {position.ticker} - {reason}")
        print(f"   P&L: {position.unrealized_pnl:.2f} PLN ({position.unrealized_pnl_pct:.2f}%)")

        return True

    def get_position(self, ticker: str) -> Optional[Position]:
        """Pobiera pozycję dla danego tickera"""
        for position in self.positions:
            if position.ticker == ticker:
                return position
        return None

    def _can_add_position(self, shares: int, entry_price: float) -> bool:
        """Sprawdza czy można dodać pozycję"""
        position_value = shares * entry_price

        # Sprawdź maksymalną liczbę pozycji
        if len(self.positions) >= self.config['max_positions']:
            print(f"❌ Osiągnięto maksymalną liczbę pozycji ({self.config['max_positions']})")
            return False

        # Sprawdź portfolio heat
        current_investment = sum(p.market_value for p in self.positions)
        total_portfolio = self._get_portfolio_value()

        if (current_investment + position_value) / total_portfolio > self.config['max_portfolio_heat']:
            print(f"❌ Przekroczono maksymalne zaangażowanie portfolio ({self.config['max_portfolio_heat']*100}%)")
            return False

        return True

    def _update_position(self, existing_position: Position, additional_shares: int, new_price: float) -> bool:
        """Aktualizuje istniejącą pozycję"""
        # Oblicz nową średnią cenę wejścia
        total_shares = existing_position.shares + additional_shares
        total_cost = (existing_position.shares * existing_position.entry_price) + (additional_shares * new_price)
        avg_entry_price = total_cost / total_shares

        existing_position.shares = total_shares
        existing_position.entry_price = avg_entry_price
        existing_position.current_price = new_price
        existing_position.last_updated = datetime.now()

        # Zapisz transakcję
        self._record_transaction('BUY', existing_position.ticker, additional_shares, new_price)

        return True

    def _record_transaction(self, action: str, ticker: str, shares: int, price: float, reason: str = None):
        """Zapisuje transakcję w historii"""
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'value': shares * price,
            'reason': reason
        }
        self.transaction_history.append(transaction)

    def _get_portfolio_value(self) -> float:
        """Pobiera całkowitą wartość portfolio"""
        return self.config.get('total_capital', 100000)

    def calculate_portfolio_metrics(self) -> RiskMetrics:
        """Oblicza metryki ryzyka portfolio"""
        total_capital = self._get_portfolio_value()
        invested_capital = sum(p.market_value for p in self.positions)
        cash_available = total_capital - invested_capital

        # Portfolio heat
        portfolio_heat = invested_capital / total_capital

        # Value at Risk (uproszczony)
        var_95 = self._calculate_var()

        # Maksymalne obsunięcie
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio (uproszczony)
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Beta (uproszczony)
        beta = self._calculate_beta()

        return RiskMetrics(
            total_capital=total_capital,
            invested_capital=invested_capital,
            cash_available=cash_available,
            total_positions=len(self.positions),
            portfolio_heat=portfolio_heat,
            max_position_size=self.config['max_position_size'],
            var_95=var_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            beta=beta
        )

    def _calculate_var(self) -> float:
        """Oblicza Value at Risk 95% (uproszczony)"""
        if not self.positions:
            return 0.0

        # Uproszczony VaR bazowany na stop loss
        total_risk = sum(p.risk_amount for p in self.positions)
        portfolio_value = sum(p.market_value for p in self.positions)

        if portfolio_value > 0:
            return (total_risk / portfolio_value) * 100
        return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Oblicza maksymalne obsunięcie (uproszczony)"""
        # W rzeczywistym systemie potrzebujemy historycznych danych
        # Tutaj uproszczona implementacja
        if not self.positions:
            return 0.0

        # Maksymalne potencjalne straty z stop loss
        max_potential_loss = sum(p.risk_amount for p in self.positions)
        portfolio_value = sum(p.market_value for p in self.positions)

        if portfolio_value > 0:
            return (max_potential_loss / portfolio_value) * 100
        return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """Oblicza Sharpe ratio (uproszczony)"""
        # W rzeczywistym systemie potrzebujemy historycznych zwrotów
        # Tutaj uproszczona implementacja bazowana na oczekiwaniach
        expected_return = 0.12  # 12% rocznie
        volatility = 0.20  # 20% rocznie

        risk_free_rate = self.config['risk_free_rate']
        excess_return = expected_return - risk_free_rate

        if volatility > 0:
            return excess_return / volatility
        return 0.0

    def _calculate_beta(self) -> float:
        """Oblicza beta (uproszczony)"""
        # W rzeczywistym systemie potrzebujemy korelacji z rynkiem
        # Tutaj przyjęcie beta = 1 dla portfela akcji GPW
        return 1.0

    def get_risk_alerts(self) -> List[Dict]:
        """Generuje alerty ryzyka"""
        alerts = []
        metrics = self.calculate_portfolio_metrics()

        # Alert o wysokim zaangażowaniu
        if metrics.portfolio_heat > 0.8:
            alerts.append({
                'type': 'HIGH_PORTFOLIO_HEAT',
                'message': f'Wysokie zaangażowanie portfolio: {metrics.portfolio_heat:.1%}',
                'severity': 'HIGH'
            })

        # Alert o pozycjach bliskich stop loss
        for position in self.positions:
            distance_to_sl = (position.current_price - position.stop_loss) / position.current_price
            if distance_to_sl < 0.02:  # Mniej niż 2% do stop loss
                alerts.append({
                    'type': 'NEAR_STOP_LOSS',
                    'ticker': position.ticker,
                    'message': f'{position.ticker} bliski stop loss: {distance_to_sl:.1%}',
                    'severity': 'MEDIUM'
                })

        # Alert o zbyt dużej liczbie pozycji
        if metrics.total_positions > self.config['max_positions'] * 0.8:
            alerts.append({
                'type': 'TOO_MANY_POSITIONS',
                'message': f'Zbyt wiele pozycji: {metrics.total_positions}',
                'severity': 'MEDIUM'
            })

        return alerts

    def rebalance_portfolio(self) -> List[Dict]:
        """Generuje rekomendacje rebalancingu"""
        recommendations = []
        metrics = self.calculate_portfolio_metrics()

        # Rebalancing jeśli portfolio heat jest zbyt wysoki
        if metrics.portfolio_heat > self.config['max_portfolio_heat']:
            # Prosta strategia: zamknij najgorsze pozycje
            sorted_positions = sorted(self.positions, key=lambda p: p.unrealized_pnl_pct)

            for position in sorted_positions:
                if metrics.portfolio_heat <= self.config['max_portfolio_heat'] * 0.8:
                    break

                recommendations.append({
                    'action': 'CLOSE',
                    'ticker': position.ticker,
                    'reason': 'Portfolio rebalancing - excessive heat',
                    'priority': 'HIGH'
                })

        return recommendations

    def generate_risk_report(self) -> Dict:
        """Generuje raport ryzyka"""
        metrics = self.calculate_portfolio_metrics()
        alerts = self.get_risk_alerts()
        recommendations = self.rebalance_portfolio()

        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_capital': metrics.total_capital,
                'invested_capital': metrics.invested_capital,
                'cash_available': metrics.cash_available,
                'portfolio_heat_pct': metrics.portfolio_heat * 100,
                'total_positions': metrics.total_positions,
                'var_95_pct': metrics.var_95,
                'max_drawdown_pct': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'beta': metrics.beta
            },
            'positions': [
                {
                    'ticker': p.ticker,
                    'shares': p.shares,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'market_value': p.market_value,
                    'unrealized_pnl': p.unrealized_pnl,
                    'unrealized_pnl_pct': p.unrealized_pnl_pct,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit,
                    'risk_amount': p.risk_amount
                }
                for p in self.positions
            ],
            'alerts': alerts,
            'recommendations': recommendations,
            'risk_score': self._calculate_risk_score(metrics)
        }

    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Oblicza ogólny score ryzyka (0-100)"""
        score = 50  # Bazowy score

        # Portfolio heat
        if metrics.portfolio_heat > 0.8:
            score += 20
        elif metrics.portfolio_heat > 0.6:
            score += 10
        elif metrics.portfolio_heat < 0.3:
            score -= 10

        # VaR
        if metrics.var_95 > 10:
            score += 15
        elif metrics.var_95 > 5:
            score += 8

        # Liczba pozycji
        if metrics.total_positions > 8:
            score += 10
        elif metrics.total_positions < 3:
            score -= 5

        # Maksymalne obsunięcie
        if metrics.max_drawdown > 15:
            score += 15
        elif metrics.max_drawdown > 10:
            score += 8

        return max(0, min(100, score))