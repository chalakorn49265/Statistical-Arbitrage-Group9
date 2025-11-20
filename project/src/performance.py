"""
Performance evaluation and metrics computation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from utils import (
    annualize_return, annualize_volatility, sharpe_ratio,
    max_drawdown, calmar_ratio
)


def compute_performance_metrics(returns: pd.Series,
                               signals: Optional[pd.DataFrame] = None,
                               periods_per_year: int = 252,
                               risk_free_rate: float = 0.0) -> Dict:
    """
    Compute comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    signals : pd.DataFrame or None
        Signal dataframe (for trade-level metrics)
    periods_per_year : int
        Trading periods per year
    risk_free_rate : float
        Risk-free rate for Sharpe calculation
    
    Returns:
    --------
    dict : Dictionary of performance metrics
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return _empty_metrics()
    
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    ann_return = annualize_return(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    
    # Drawdown metrics
    equity = (1 + returns).cumprod()
    dd = max_drawdown(equity)
    calmar = calmar_ratio(returns, periods_per_year)
    
    # Win rate and trade metrics
    if signals is not None and 'signal' in signals.columns:
        trade_metrics = _compute_trade_metrics(signals, returns)
    else:
        trade_metrics = {}
    
    # Turnover and capacity
    if signals is not None and 'notional' in signals.columns:
        capacity_metrics = _compute_capacity_metrics(signals)
    else:
        capacity_metrics = {}
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': dd,
        'calmar_ratio': calmar,
        'n_periods': len(returns),
        'positive_periods': (returns > 0).sum(),
        'negative_periods': (returns < 0).sum(),
        'win_rate': (returns > 0).mean() if len(returns) > 0 else 0.0,
        **trade_metrics,
        **capacity_metrics
    }


def _compute_trade_metrics(signals: pd.DataFrame, returns: pd.Series) -> Dict:
    """Compute trade-level metrics."""
    if 'signal' not in signals.columns:
        return {}
    
    # Identify trades
    trades = []
    in_trade = False
    entry_date = None
    entry_signal = None
    
    for date in signals.index:
        signal = signals.loc[date, 'signal']
        
        if signal != 0 and not in_trade:
            # Entry
            in_trade = True
            entry_date = date
            entry_signal = signal
        elif signal == 0 and in_trade:
            # Exit
            exit_date = date
            trade_returns = returns.loc[entry_date:exit_date].sum()
            holding_period = (exit_date - entry_date).days
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'signal': entry_signal,
                'return': trade_returns,
                'holding_period': holding_period
            })
            
            in_trade = False
    
    if len(trades) == 0:
        return {
            'n_trades': 0,
            'avg_trade_return': 0.0,
            'avg_holding_period': 0.0,
            'win_rate_trades': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    
    return {
        'n_trades': len(trades),
        'avg_trade_return': trades_df['return'].mean(),
        'median_trade_return': trades_df['return'].median(),
        'std_trade_return': trades_df['return'].std(),
        'best_trade': trades_df['return'].max(),
        'worst_trade': trades_df['return'].min(),
        'avg_holding_period': trades_df['holding_period'].mean(),
        'median_holding_period': trades_df['holding_period'].median(),
        'win_rate_trades': (trades_df['return'] > 0).mean(),
        'profit_factor': abs(trades_df[trades_df['return'] > 0]['return'].sum()) / 
                        abs(trades_df[trades_df['return'] < 0]['return'].sum()) 
                        if (trades_df['return'] < 0).any() else np.inf
    }


def _compute_capacity_metrics(signals: pd.DataFrame) -> Dict:
    """Compute capacity and turnover metrics."""
    if 'notional' not in signals.columns:
        return {}
    
    notional = signals['notional']
    avg_notional = notional[notional > 0].mean() if (notional > 0).any() else 0.0
    max_notional = notional.max()
    
    # Turnover (approximate)
    if 'position_size_A' in signals.columns:
        pos_a = signals['position_size_A']
        pos_b = signals['position_size_B']
        turnover = (pos_a.diff().abs() + pos_b.diff().abs()).sum() / 2
        avg_turnover = turnover / len(signals) if len(signals) > 0 else 0.0
    else:
        avg_turnover = 0.0
    
    return {
        'avg_notional': avg_notional,
        'max_notional': max_notional,
        'avg_turnover': avg_turnover
    }


def _empty_metrics() -> Dict:
    """Return empty metrics dictionary."""
    return {
        'total_return': 0.0,
        'annualized_return': 0.0,
        'annualized_volatility': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'n_periods': 0,
        'positive_periods': 0,
        'negative_periods': 0,
        'win_rate': 0.0,
        'n_trades': 0,
        'avg_trade_return': 0.0,
        'avg_holding_period': 0.0,
        'win_rate_trades': 0.0
    }


def compute_rolling_metrics(returns: pd.Series,
                           window: int = 252,
                           periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute rolling performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    window : int
        Rolling window size
    periods_per_year : int
        Trading periods per year
    
    Returns:
    --------
    pd.DataFrame : Rolling metrics
    """
    results = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        metrics = compute_performance_metrics(
            window_returns, periods_per_year=periods_per_year
        )
        results.append(metrics)
    
    result_df = pd.DataFrame(results, index=returns.index[window-1:])
    return result_df

