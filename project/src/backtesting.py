"""
Backtesting framework with purged and embargoed walk-forward validation.

Implements:
- Walk-forward splits with training/test periods
- Purging and embargo to avoid data leakage
- Parameter optimization on training sets
- Performance evaluation on test sets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
from itertools import product

from .signal_generation import generate_signals, compute_position_sizes
from .performance import compute_performance_metrics


class WalkForwardBacktest:
    """
    Walk-forward backtesting with purged and embargoed splits.
    """
    
    def __init__(self,
                 train_window_years: float = 2.0,
                 test_window_years: float = 0.5,
                 purge_days: int = 10,
                 embargo_days: int = 5,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize walk-forward backtest.
        
        Parameters:
        -----------
        train_window_years : float
            Training window size in years
        test_window_years : float
            Test window size in years
        purge_days : int
            Days to purge between train/test to avoid overlap
        embargo_days : int
            Days to embargo around test periods
        start_date : str or None
            Start date for backtest
        end_date : str or None
            End date for backtest
        """
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        self.folds = []
    
    def create_folds(self, data: pd.DataFrame) -> List[Dict]:
        """
        Create walk-forward folds with purging and embargo.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with Date index
        
        Returns:
        --------
        list : List of fold dictionaries with 'train_start', 'train_end', 
               'test_start', 'test_end' keys
        """
        if self.start_date:
            data = data[data.index >= self.start_date]
        if self.end_date:
            data = data[data.index <= self.end_date]
        
        if len(data) == 0:
            return []
        
        # Convert years to trading days (approximate)
        train_days = int(self.train_window_years * 252)
        test_days = int(self.test_window_years * 252)
        
        folds = []
        current_date = data.index[0]
        end_date = data.index[-1]
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end_idx = data.index.get_indexer([train_start], method='nearest')[0] + train_days
            
            if train_end_idx >= len(data):
                break
            
            train_end = data.index[train_end_idx]
            
            # Purge period
            purge_start = train_end
            purge_end_idx = data.index.get_indexer([purge_start], method='nearest')[0] + self.purge_days
            
            if purge_end_idx >= len(data):
                break
            
            purge_end = data.index[purge_end_idx]
            
            # Test period
            test_start = purge_end
            test_end_idx = data.index.get_indexer([test_start], method='nearest')[0] + test_days
            
            if test_end_idx >= len(data):
                test_end = data.index[-1]
            else:
                test_end = data.index[test_end_idx]
            
            # Embargo
            embargo_end = data.index[min(
                data.index.get_indexer([test_end], method='nearest')[0] + self.embargo_days,
                len(data) - 1
            )]
            
            # Check if we have enough data
            if test_end <= train_end:
                break
            
            folds.append({
                'fold': len(folds) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_end,
                'purge_end': purge_end,
                'test_start': test_start,
                'test_end': test_end,
                'embargo_end': embargo_end
            })
            
            # Move to next fold (start after embargo)
            current_date = embargo_end
        
        self.folds = folds
        return folds
    
    def run_backtest(self,
                    pair_data: pd.DataFrame,
                    spread: pd.Series,
                    param_grid: Dict,
                    signal_config: Dict,
                    cost_config: Dict) -> pd.DataFrame:
        """
        Run backtest on all folds.
        
        Parameters:
        -----------
        pair_data : pd.DataFrame
            Pair dataset
        spread : pd.Series
            Spread series
        param_grid : dict
            Parameter grid for optimization
        signal_config : dict
            Base signal configuration
        cost_config : dict
            Cost configuration (transaction costs, etc.)
        
        Returns:
        --------
        pd.DataFrame : Backtest results with columns:
            - fold, train_start, train_end, test_start, test_end
            - best_params: Best parameters from training
            - test_returns: Test period returns
            - test_sharpe: Test period Sharpe ratio
            - test_drawdown: Test period max drawdown
            - etc.
        """
        if not self.folds:
            self.create_folds(pair_data)
        
        results = []
        
        for fold in self.folds:
            # Training data
            train_mask = (pair_data.index >= fold['train_start']) & (pair_data.index <= fold['train_end'])
            train_data = pair_data[train_mask]
            train_spread = spread[train_mask]
            
            # Test data
            test_mask = (pair_data.index >= fold['test_start']) & (pair_data.index <= fold['test_end'])
            test_data = pair_data[test_mask]
            test_spread = spread[test_mask]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Optimize parameters on training set
            best_params = self._optimize_parameters(
                train_data, train_spread, param_grid, signal_config, cost_config
            )
            
            # Run backtest on test set with best parameters
            test_result = self._run_single_backtest(
                test_data, test_spread, best_params, signal_config, cost_config
            )
            
            # Combine results
            fold_result = {
                **fold,
                'best_params': best_params,
                **test_result
            }
            results.append(fold_result)
        
        return pd.DataFrame(results)
    
    def _optimize_parameters(self,
                            train_data: pd.DataFrame,
                            train_spread: pd.Series,
                            param_grid: Dict,
                            base_config: Dict,
                            cost_config: Dict) -> Dict:
        """
        Optimize parameters on training set.
        
        Uses grid search and selects based on Sharpe ratio.
        """
        best_sharpe = -np.inf
        best_params = None
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            
            # Merge with base config
            config = {**base_config, **params}
            
            # Run backtest on training set
            result = self._run_single_backtest(
                train_data, train_spread, params, base_config, cost_config
            )
            
            sharpe = result.get('sharpe_ratio', -np.inf)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params if best_params else base_config
    
    def _run_single_backtest(self,
                            data: pd.DataFrame,
                            spread: pd.Series,
                            signal_params: Dict,
                            base_config: Dict,
                            cost_config: Dict) -> Dict:
        """
        Run single backtest period.
        
        Returns performance metrics.
        """
        # Generate signals
        config = {**base_config, **signal_params}
        signals = generate_signals(data, spread, config)
        
        # Compute position sizes
        signals = compute_position_sizes(signals, data, cost_config)
        
        # Compute returns
        returns = self._compute_returns(signals, data, cost_config)
        
        # Compute performance metrics
        metrics = compute_performance_metrics(returns, signals)
        
        return metrics
    
    def _compute_returns(self,
                        signals: pd.DataFrame,
                        pair_data: pd.DataFrame,
                        cost_config: Dict) -> pd.Series:
        """
        Compute strategy returns including transaction costs.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Signals with position sizes
        pair_data : pd.DataFrame
            Pair data with returns
        cost_config : dict
            Cost configuration
        
        Returns:
        --------
        pd.Series : Strategy returns
        """
        transaction_cost_bps = cost_config.get('transaction_cost_bps', 10) / 10000
        spread_cost_bps = cost_config.get('spread_cost_bps', 5) / 10000
        
        # Align data
        common_idx = signals.index.intersection(pair_data.index)
        
        # Get returns
        returns_a = pair_data.loc[common_idx, 'returns_A']
        returns_b = pair_data.loc[common_idx, 'returns_B']
        
        # Position sizes
        pos_a = signals.loc[common_idx, 'position_size_A']
        pos_b = signals.loc[common_idx, 'position_size_B']
        
        # Compute P&L
        pnl = pd.Series(index=common_idx, data=0.0)
        
        prev_position_a = 0.0
        prev_position_b = 0.0
        
        for i, date in enumerate(common_idx):
            curr_pos_a = pos_a.loc[date] if not pd.isna(pos_a.loc[date]) else 0.0
            curr_pos_b = pos_b.loc[date] if not pd.isna(pos_b.loc[date]) else 0.0
            
            # Position change
            delta_a = curr_pos_a - prev_position_a
            delta_b = curr_pos_b - prev_position_b
            
            # Transaction costs
            turnover_a = abs(delta_a)
            turnover_b = abs(delta_b)
            total_cost = (turnover_a + turnover_b) * (transaction_cost_bps + spread_cost_bps)
            
            # P&L from positions
            if i > 0:
                prev_date = common_idx[i-1]
                pnl_from_positions = (
                    prev_position_a * returns_a.loc[date] +
                    prev_position_b * returns_b.loc[date]
                )
            else:
                pnl_from_positions = 0.0
            
            # Net P&L
            notional = abs(curr_pos_a) + abs(curr_pos_b)
            if notional > 0:
                pnl.loc[date] = (pnl_from_positions - total_cost) / notional
            else:
                pnl.loc[date] = 0.0
            
            prev_position_a = curr_pos_a
            prev_position_b = curr_pos_b
        
        return pnl


def run_backtest(pair_data: pd.DataFrame,
                spread: pd.Series,
                signal_config: Dict,
                cost_config: Dict,
                walk_forward_config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to run full backtest.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset
    spread : pd.Series
        Spread series
    signal_config : dict
        Signal configuration
    cost_config : dict
        Cost configuration
    walk_forward_config : dict or None
        Walk-forward configuration
    
    Returns:
    --------
    dict : Backtest results
    """
    if walk_forward_config is None:
        walk_forward_config = {}
    
    backtest = WalkForwardBacktest(**walk_forward_config)
    results = backtest.run_backtest(
        pair_data, spread, {}, signal_config, cost_config
    )
    
    return {
        'results': results,
        'folds': backtest.folds,
        'summary': results.describe() if len(results) > 0 else {}
    }

