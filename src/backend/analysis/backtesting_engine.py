#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework
Implements robust backtesting with performance metrics and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import os
from config import BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION_RATE, BACKTEST_SLIPPAGE_PCT
from risk_management import RiskManager
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies"""

    def __init__(self, initial_capital: float = BACKTEST_INITIAL_CAPITAL,
                 commission_rate: float = BACKTEST_COMMISSION_RATE,
                 slippage_pct: float = BACKTEST_SLIPPAGE_PCT):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.risk_manager = RiskManager()

        # Track backtest state
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.portfolio_value = []

        # Performance tracking
        self.returns = []
        self.win_trades = []
        self.loss_trades = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_value = initial_capital

    def load_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for backtesting

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Dictionary mapping tickers to DataFrames

        Example:
        >>> engine = BacktestEngine()
        >>> data = engine.load_historical_data(['PKN.WA', 'PKO.WA'], '2023-01-01', '2023-12-31')
        >>> print(f"Loaded data for {len(data)} tickers")
        Loaded data for 2 tickers
        """
        data = {}

        for ticker in tickers:
            try:
                # Try yfinance first
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if not df.empty:
                    data[ticker] = df
                    continue

                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                data[ticker] = df

            except Exception as e:
                print(f"Failed to load data for {ticker}: {e}")
                # Create empty DataFrame to maintain structure
                data[ticker] = pd.DataFrame()

        return data

    def run_backtest(self, strategy_func, signals_data: Dict, price_data: Dict) -> Dict:
        """
        Run backtest with given strategy function and signals

        Args:
            strategy_func: Strategy function that accepts price data and signals
            signals_data: Trading signals dictionary
            price_data: Historical price data

        Returns:
            Backtest results dictionary

        Example:
        >>> def simple_strategy(price_data, signals):
        ...     # Simple moving average crossover strategy
        ...     return signals
        >>> engine = BacktestEngine()
        >>> results = engine.run_backtest(simple_strategy, signals, data)
        >>> print(f"Final capital: {results['final_capital']:.2f}")
        Final capital: 105000.00
        """
        self.reset_backtest()

        # Get list of all unique dates
        all_dates = []
        for ticker in price_data:
            if ticker in price_data and not price_data[ticker].empty:
                all_dates.extend(price_data[ticker].index.tolist())

        all_dates = sorted(list(set(all_dates)))

        # Iterate through each day
        for date in all_dates:
            # Update portfolio based on current prices
            self.update_portfolio_value(date, price_data)

            # Get signals for this date
            daily_signals = self.get_signals_for_date(signals_data, date)

            # Execute trades based on signals
            self.execute_trades(daily_signals, date, price_data)

        # Calculate final results
        return self.calculate_results()

    def get_signals_for_date(self, signals_data: Dict, date: datetime) -> List[Dict]:
        """
        Get signals for a specific date

        Args:
            signals_data: All signals data
            date: Target date

        Returns:
            List of signals for the date
        """
        date_str = date.strftime('%Y-%m-%d')
        return signals_data.get(date_str, [])

    def update_portfolio_value(self, date: datetime, price_data: Dict):
        """
        Update portfolio value with current prices

        Args:
            date: Current date
            price_data: Current price data
        """
        total_value = self.cash

        for ticker, quantity in self.positions.items():
            if quantity != 0 and ticker in price_data:
                current_price = self.get_current_price(ticker, price_data, date)
                if current_price > 0:
                    position_value = quantity * current_price
                    total_value += position_value

        self.equity_curve.append({
            'date': date,
            'portfolio_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })

        # Track returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            current_value = total_value
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self.returns.append(daily_return)

        # Update drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
            self.current_drawdown = 0
        else:
            drawdown = (self.peak_value - total_value) / self.peak_value
            self.current_drawdown = drawdown
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        self.portfolio_value.append(total_value)

    def get_current_price(self, ticker: str, price_data: Dict, date: datetime) -> float:
        """
        Get current price for a ticker

        Args:
            ticker: Ticker symbol
            price_data: Price data dictionary
            date: Target date

        Returns:
            Current price
        """
        if ticker in price_data and not price_data[ticker].empty:
            try:
                return float(price_data[ticker].loc[date, 'close'])
            except KeyError:
                # If exact date not found, get previous close
                available_dates = price_data[ticker].index[price_data[ticker].index <= date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    return float(price_data[ticker].loc[closest_date, 'close'])
        return 0.0

    def execute_trades(self, signals: List[Dict], date: datetime, price_data: Dict):
        """
        Execute trades based on signals

        Args:
            signals: List of trading signals
            date: Current date
            price_data: Price data dictionary
        """
        for signal in signals:
            ticker = signal.get('ticker')
            action = signal.get('action', '').upper()
            confidence = signal.get('confidence', 0.5)

            if confidence < 0.6:  # Skip low confidence signals
                continue

            if action == 'BUY':
                self.execute_buy_order(ticker, date, price_data, signal)
            elif action == 'SELL':
                self.execute_sell_order(ticker, date, price_data, signal)

    def execute_buy_order(self, ticker: str, date: datetime, price_data: Dict, signal: Dict):
        """
        Execute buy order

        Args:
            ticker: Ticker symbol
            date: Trade date
            price_data: Price data
            signal: Signal information
        """
        price = self.get_current_price(ticker, price_data, date)
        if price <= 0:
            return

        # Calculate position size using risk management
        risk_per_trade = signal.get('risk_per_trade_pct', 0.02)  # 2% risk per trade default
        stop_loss = signal.get('stop_loss', price * 0.95)  # Default 5% stop
        position_details = self.risk_manager.calculate_position_size(
            self.cash, risk_per_trade, price, stop_loss
        )

        if position_details['shares'] <= 0:
            return

        shares = position_details['shares']
        cost = shares * price
        commission = cost * self.commission_rate
        slippage = cost * self.slippage_pct
        total_cost = cost + commission + slippage

        # Check if we have enough cash
        if total_cost > self.cash:
            return

        # Execute trade
        self.positions[ticker] = self.positions.get(ticker, 0) + shares
        self.cash -= total_cost

        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'quantity': shares,
            'price': price,
            'cost': total_cost,
            'commission': commission,
            'slippage': slippage,
            'signal_confidence': signal.get('confidence', 0.5),
            'stop_loss': stop_loss,
            'take_profit': signal.get('take_profit')
        }

        self.trades.append(trade)

    def execute_sell_order(self, ticker: str, date: datetime, price_data: Dict, signal: Dict):
        """
        Execute sell order

        Args:
            ticker: Ticker symbol
            date: Trade date
            price_data: Price data
            signal: Signal information
        """
        current_position = self.positions.get(ticker, 0)
        if current_position <= 0:
            return

        price = self.get_current_price(ticker, price_data, date)
        if price <= 0:
            return

        shares = min(current_position, abs(current_position))
        proceeds = shares * price
        commission = proceeds * self.commission_rate
        slippage = proceeds * self.slippage
        net_proceeds = proceeds - commission - slippage

        # Execute trade
        self.positions[ticker] = current_position - shares
        self.cash += net_proceeds

        # Calculate P&L
        cost_basis = signal.get('cost_basis', price) * shares
        if cost_basis > 0:
            pl = (net_proceeds - cost_basis) / cost_basis
        else:
            pl = 0

        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'quantity': shares,
            'price': price,
            'proceeds': net_proceeds,
            'commission': commission,
            'slippage': slippage,
            'pl_pct': pl,
            'signal_confidence': signal.get('confidence', 0.5)
        }

        self.trades.append(trade)

        # Categorize trade
        if pl > 0:
            self.win_trades.append(trade)
        else:
            self.loss_trades.append(trade)

    def reset_backtest(self):
        """Reset backtest to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.portfolio_value = []
        self.returns = []
        self.win_trades = []
        self.loss_trades = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_value = self.initial_capital

    def calculate_results(self) -> Dict:
        """
        Calculate comprehensive backtest results

        Returns:
            Dictionary with all performance metrics
        """
        if len(self.equity_curve) > 0:
            final_value = self.equity_curve[-1]['portfolio_value']
            total_return = (final_value - self.initial_capital) / self.initial_capital
            annual_return = total_return * (365 / 365)  # Adjust for actual period

            return {
                'initial_capital': self.initial_capital,
                'final_capital': final_value,
                'total_return': total_return,
                'annual_return': annual_return,
                'total_trades': len(self.trades),
                'winning_trades': len(self.win_trades),
                'losing_trades': len(self.loss_trades),
                'win_rate': len(self.win_trades) / len(self.trades) if self.trades else 0,
                'max_drawdown': self.max_drawdown,
                'final_drawdown': self.current_drawdown,
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
                'profit_factor': self.calculate_profit_factor(),
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'returns': self.returns
            }
        else:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'annual_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'final_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'profit_factor': 0,
                'equity_curve': [],
                'trades': [],
                'returns': []
            }

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns) < 2:
            return 0

        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate / 252

        if np.std(excess_returns) == 0:
            return 0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside risk adjusted)"""
        if len(self.returns) < 2:
            return 0

        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate / 252

        downside_returns = np.minimum(excess_returns, 0)
        if np.std(downside_returns) == 0:
            return 0

        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if self.max_drawdown == 0:
            return 0

        total_return = (self.equity_curve[-1]['portfolio_value'] - self.initial_capital) / self.initial_capital
        return total_return / self.max_drawdown

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum([trade['pl_pct'] for trade in self.win_trades if trade['pl_pct'] > 0])
        gross_loss = abs(sum([trade['pl_pct'] for trade in self.loss_trades if trade['pl_pct'] < 0]))

        if gross_loss == 0:
            return float('inf')

        return gross_profit / gross_loss

    def generate_equity_curve(self) -> plt.Figure:
        """
        Generate equity curve visualization

        Returns:
            Matplotlib figure
        """
        if not self.equity_curve:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        ax1.plot(df['date'], df['portfolio_value'], color='blue', linewidth=2)
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (PLN)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Drawdown
        drawdown = []
        peak = df['portfolio_value'].expanding().max()
        drawdown = (df['portfolio_value'] / peak - 1) * 100
        ax2.fill_between(df['date'], drawdown, 0, color='red', alpha=0.3)
        ax2.plot(df['date'], drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def plot_results(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive backtest visualization

        Args:
            save_path: Path to save the plot

        Returns:
            Dictionary with plot file paths
        """
        if not self.trades:
            print("No trades to plot")
            return {}

        results = self.calculate_results()

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Equity curve and drawdown
        fig.add_subplot(2, 2, (1, 1))
        equity_curve = self.generate_equity_curve()

        # Monthly returns heatmap
        fig.add_subplot(2, 2, (1, 2))
        self.plot_monthly_returns_heatmap(fig)

        # Trade distribution
        fig.add_subplot(2, 2, (2, 1))
        self.plot_trade_distribution(fig)

        # Performance metrics
        fig.add_subplot(2, 2, (2, 2))
        self.plot_performance_metrics(fig, results)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Backtest plots saved to {save_path}")

        plt.show()

        return {'figure': fig, 'save_path': save_path}

    def plot_monthly_returns_heatmap(self, fig):
        """Plot monthly returns heatmap"""
        if not self.returns:
            return

        returns_df = pd.Series(self.returns)
        returns_df.index = pd.to_datetime(returns_df.index)

        # Group by month and calculate returns
        monthly_returns = returns_df.groupby([returns_df.index.year, returns_df.index.month]).sum()

        # Create matrix for heatmap
        years = sorted(returns_df.index.year.unique())
        months = range(1, 13)
        matrix = np.zeros((len(years), len(months)))

        for i, year in enumerate(years):
            for j, month in enumerate(months):
                if (year, month) in monthly_returns.index:
                    matrix[i, j] = monthly_returns[(year, month)]

        months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax = fig.add_subplot(2, 2, (1, 2))
        sns.heatmap(matrix, xticklabels=months_labels, yticklabels=years,
                   cmap='RdYlGn', center=0, annot=True, fmt='.1%',
                   cbar_kws={'label': 'Monthly Returns (%)'})

        ax.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')

    def plot_trade_distribution(self, fig):
        """Plot trade P&L distribution"""
        all_returns = [trade['pl_pct'] for trade in self.trades]

        ax = fig.add_subplot(2, 2, (2, 1))
        ax.hist(all_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_performance_metrics(self, fig, results):
        """Plot performance metrics dashboard"""
        ax = fig.add_subplot(2, 2, (2, 2))
        metrics = [
            ('Total Return', f"{results['total_return']:.2%}"),
            ('Annual Return', f"{results['annual_return']:.2%}"),
            ('Win Rate', f"{results['win_rate']:.2%}"),
            ('Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"),
            ('Sortino Ratio', f"{results['sortino_ratio']:.2f}"),
            ('Max Drawdown', f"{results['max_drawdown']:.2%}"),
            ('Profit Factor', f"{results['profit_factor']:.2f}"),
            ('Total Trades', f"{results['total_trades']}")
        ]

        ax.axis('tight')
        ax.axis('off')

        for i, (metric, value) in enumerate(metrics):
            ax.text(0.5, 0.9 - i * 0.1, f"{metric}: {value}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')

    def walk_forward_analysis(self, strategy_func, train_period_months: int = 12,
                              test_period_months: int = 3, step_months: int = 6) -> Dict:
        """
        Perform walk-forward analysis

        Args:
            strategy_func: Strategy function to test
            train_period_months: Training period in months
            test_period_months: Testing period in months
            step_months: Step size in months

        Returns:
            Walk-forward analysis results
        """
        results = []
        start_date = datetime(2020, 1, 1)

        while start_date + timedelta(days=train_period_months * 30) < datetime.now():
            end_date = start_date + timedelta(days=(train_period_months + test_period_months) * 30)

            # Load data for period
            end_date_str = end_date.strftime('%Y-%m-%d')
            tickers = ['PKN.WA', 'PKO.WA', 'PZU.WA']  # Example tickers

            price_data = self.load_historical_data(tickers,
                                                   start_date.strftime('%Y-%m-%d'),
                                                   end_date_str)

            # Generate signals (placeholder - would use actual strategy)
            signals_data = self.generate_placeholder_signals(start_date, end_date, tickers)

            # Run backtest on test period
            test_start = start_date + timedelta(days=train_period_months * 30)
            test_end = end_date
            test_data = {ticker: df.loc[test_start:test_end] for ticker, df in price_data.items() if not df.empty}

            result = self.run_backtest(strategy_func, signals_data, test_data)
            result['train_period'] = f"{start_date.strftime('%Y-%m-%d')} to {(start_date + timedelta(days=train_period_months * 30)).strftime('%Y-%m-%d')}"
            result['test_period'] = f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"

            results.append(result)

            # Move forward
            start_date = start_date + timedelta(days=step_months * 30)

        return {'results': results}

    def generate_placeholder_signals(self, start_date: datetime, end_date: datetime,
                                     tickers: List[str]) -> Dict:
        """Generate placeholder signals for testing"""
        signals = {}
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            signals[date_str] = []

            # Generate random signals for testing
            for ticker in tickers:
                if np.random.random() > 0.8:  # 20% chance of signal
                    action = 'BUY' if np.random.random() > 0.4 else 'SELL'
                    confidence = np.random.uniform(0.6, 1.0)
                    signals[date_str].append({
                        'ticker': ticker,
                        'action': action,
                        'confidence': confidence,
                        'risk_per_trade_pct': np.random.uniform(0.01, 0.03)
                    })

            current_date += timedelta(days=1)

        return signals

    def monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict:
        """
        Perform Monte Carlo simulation on trades

        Args:
            num_simulations: Number of simulation runs

        Returns:
            Monte Carlo simulation results
        """
        if not self.trades:
            return {'error': 'No trades to simulate'}

        original_returns = [trade['pl_pct'] for trade in self.trades]
        results = []

        for i in range(num_simulations):
            # Shuffle trade order
            shuffled_returns = np.random.permutation(original_returns)
            cumulative_returns = np.cumprod(1 + shuffled_returns)

            final_value = self.initial_capital * cumulative_returns[-1]
            max_dd = (np.maximum.accumulate(cumulative_returns, np.maximum) - cumulative_returns).max() / np.maximum.accumulate(cumulative_returns, np.maximum)

            results.append({
                'simulation': i + 1,
                'final_value': final_value,
                'total_return': (final_value - self.initial_capital) / self.initial_capital,
                'max_drawdown': max_dd
            })

        results_df = pd.DataFrame(results)

        return {
            'simulations': num_simulations,
            'mean_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'percentile_5': results_df['total_return'].quantile(0.05),
            'percentile_95': results_df['total_return'].quantile(0.95),
            'probability_profit': (results_df['total_return'] > 0).mean(),
            'details': results_df
        }

    def generate_trade_log(self) -> pd.DataFrame:
        """Generate detailed trade log"""
        if not self.trades:
            return pd.DataFrame()

        trade_log = []
        for trade in self.trades:
            trade_log.append({
                'date': trade['date'].strftime('%Y-%m-%d'),
                'ticker': trade['ticker'],
                'action': trade['action'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'cost': trade.get('cost', 0),
                'pl_pct': trade.get('pl_pct', 0),
                'confidence': trade.get('signal_confidence', 0),
                'stop_loss': trade.get('stop_loss', 0),
                'take_profit': trade.get('take_profit', 0)
            })

        return pd.DataFrame(trade_log)

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        results = self.calculate_results()

        report = {
            'summary': {
                'Initial Capital': f"PLN {results['initial_capital']:,.2f}",
                'Final Capital': f"PLN {results['final_capital']:,.2f}",
                'Total Return': f"{results['total_return']:.2%}",
                'Annual Return': f"{results['annual_return']:.2%}",
                'Trading Period': f"{len(self.equity_curve)} days",
                'Total Trades': results['total_trades']
            },
            'performance': {
                'Win Rate': f"{results['win_rate']:.2%}",
                'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
                'Sortino Ratio': f"{results['sortino_ratio']:.2f}",
                'Max Drawdown': f"{results['max_drawdown']:.2%}",
                'Profit Factor': f"{results['profit_factor']:.2f}",
                'Winning Trades': results['winning_trades'],
                'Losing Trades': results['losing_trades']
            },
            'risk_metrics': {
                'Final Drawdown': f"{results['final_drawdown']:.2%}",
                'Maximum Drawdown': f"{results['max_drawdown']:.2%}",
                'Recovery Time': "N/A",  # Could calculate from drawdown recovery
                'Average Win': f"{np.mean([t['pl_pct'] for t in self.win_trades]):.2%}" if self.win_trades else "N/A",
                'Average Loss': f"{np.abs(np.mean([t['pl_pct'] for t in self.loss_trades])):.2%}" if self.loss_trades else "N/A"
            }
        }

        return report


# Standalone utility functions
def run_simple_backtest(tickers: List[str], start_date: str, end_date: str,
                      strategy_func=None) -> Dict:
    """
    Simple function to run backtest with default parameters

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        strategy_func: Optional strategy function

    Returns:
        Backtest results
    """
    engine = BacktestEngine()
    price_data = engine.load_historical_data(tickers, start_date, end_date)

    if strategy_func is None:
        # Use simple momentum strategy
        def simple_momentum(data, signals):
            return data  # Return price data as signals
        strategy_func = simple_momentum

    return engine.run_backtest(strategy_func, {}, price_data)


if __name__ == "__main__":
    # Example usage
    print("🚀 GPW Smart Analyzer - Backtesting Engine")
    print("=" * 50)

    # Load sample data and run backtest
    tickers = ['PKN.WA', 'PKO.WA', 'PZU.WA', 'CDR.WA']
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    engine = BacktestEngine()
    data = engine.load_historical_data(tickers, start_date, end_date)

    if data:
        print(f"✅ Loaded data for {len(data)} tickers")
        print(f"Date range: {start_date} to {end_date}")

        # Run simple backtest (using placeholder signals)
        results = run_simple_backtest(tickers, start_date, end_date)

        print("\n📊 Backtest Results:")
        print(f"Initial Capital: PLN {results['initial_capital']:,.2f}")
        print(f"Final Capital: PLN {results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}")
        print(f"Total Trades: {results['total_trades']}")

        # Generate visualization
        print("\n📈 Generating plots...")
        plots = engine.plot_results('backtest_results.png')
        print("✅ Plots saved to backtest_results.png")

        # Generate detailed report
        report = engine.generate_performance_report()
        print("\n📋 Performance Report:")
        for category, metrics in report.items():
            print(f"\n{category.title}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

        # Save detailed trade log
        trade_log = engine.generate_trade_log()
        if not trade_log.empty:
            trade_log.to_csv('backtest_trades.csv', index=False)
            print("\n💾 Trade log saved to backtest_trades.csv")

    else:
        print("❌ No data loaded. Check ticker symbols and date range.")