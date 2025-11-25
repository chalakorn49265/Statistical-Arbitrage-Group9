"""
Portfolio optimization for pairs trading strategies.

Implements:
- Portfolio weight optimization (w_i for each pair)
- Hedge ratio optimization (beta for each pair)
- Risk-based and return-based optimization objectives
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
import warnings


def optimize_portfolio_weights(
    pair_returns: Dict[str, pd.Series],
    objective: str = 'sharpe',
    constraints: Optional[Dict] = None,
    method: str = 'SLSQP',
    initial_weights: Optional[np.ndarray] = None) -> Dict:
    """
    Optimize portfolio weights for multiple pairs.
    
    Parameters:
    -----------
    pair_returns : dict
        Dictionary of {pair_name: returns_series} for each pair
    objective : str
        Objective function: 'sharpe', 'return', 'min_vol', 'max_sharpe'
    constraints : dict or None
        Constraints with keys:
            - min_weight: Minimum weight per pair (default: 0)
            - max_weight: Maximum weight per pair (default: 1)
            - sum_to_one: Whether weights must sum to 1 (default: True)
            - max_leverage: Maximum leverage (default: None)
    method : str
        Optimization method: 'SLSQP', 'trust-constr', 'differential_evolution'
    initial_weights : np.ndarray or None
        Initial guess for weights (default: equal weights)
    
    Returns:
    --------
    dict : Optimization results with keys:
        - weights: Optimal weights (dict of {pair_name: weight})
        - portfolio_returns: Portfolio return series
        - metrics: Performance metrics
        - optimization_result: Full optimization result
    """
    if constraints is None:
        constraints = {}
    
    # Align all return series on common dates
    pair_names = list(pair_returns.keys())
    returns_df = pd.DataFrame(pair_returns)
    returns_df = returns_df.dropna()
    
    if len(returns_df) == 0:
        raise ValueError("No overlapping dates between pair returns")
    
    n_pairs = len(pair_names)
    
    # Default constraints
    min_weight = constraints.get('min_weight', 0.0)
    max_weight = constraints.get('max_weight', 1.0)
    sum_to_one = constraints.get('sum_to_one', True)
    max_leverage = constraints.get('max_leverage', None)
    
    # Initial weights
    if initial_weights is None:
        initial_weights = np.ones(n_pairs) / n_pairs
    
    # Bounds
    bounds = [(min_weight, max_weight) for _ in range(n_pairs)]
    
    # Constraints
    constraint_list = []
    if sum_to_one:
        constraint_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
    
    if max_leverage is not None:
        constraint_list.append({
            'type': 'ineq',
            'fun': lambda w: max_leverage - np.sum(np.abs(w))
        })
    
    # Objective function
    if objective == 'sharpe' or objective == 'max_sharpe':
        def objective_func(w):
            portfolio_ret = (returns_df.values @ w).mean() * 252
            portfolio_vol = (returns_df.values @ w).std() * np.sqrt(252)
            if portfolio_vol == 0:
                return -np.inf
            sharpe = portfolio_ret / portfolio_vol
            return -sharpe  # Minimize negative Sharpe
    
    elif objective == 'return' or objective == 'max_return':
        def objective_func(w):
            portfolio_ret = (returns_df.values @ w).mean() * 252
            return -portfolio_ret  # Minimize negative return
    
    elif objective == 'min_vol' or objective == 'volatility':
        def objective_func(w):
            portfolio_vol = (returns_df.values @ w).std() * np.sqrt(252)
            return portfolio_vol
    
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Optimize
    if method == 'differential_evolution':
        result = differential_evolution(
            objective_func,
            bounds=bounds,
            constraints=constraint_list if constraint_list else None,
            seed=42,
            maxiter=1000
        )
    else:
        result = minimize(
            objective_func,
            x0=initial_weights,
            method=method,
            bounds=bounds,
            constraints=constraint_list if constraint_list else None,
            options={'maxiter': 1000}
        )
    
    # Extract optimal weights
    optimal_weights = result.x
    weights_dict = {pair_names[i]: optimal_weights[i] for i in range(n_pairs)}
    
    # Compute portfolio returns
    portfolio_returns = pd.Series(
        returns_df.values @ optimal_weights,
        index=returns_df.index
    )
    
    # Compute metrics
    metrics = _compute_portfolio_metrics(portfolio_returns)
    
    return {
        'weights': weights_dict,
        'portfolio_returns': portfolio_returns,
        'metrics': metrics,
        'optimization_result': result,
        'pair_names': pair_names
    }


def optimize_hedge_ratios(
    pair_data: pd.DataFrame,
    kalman_config: Dict,
    objective: str = 'sharpe',
    param_bounds: Optional[Dict] = None) -> Dict:
    """
    Optimize Kalman filter parameters (hedge ratio estimation) for a pair.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset with log_price_A, log_price_B, returns_A, returns_B
    kalman_config : dict
        Base Kalman filter configuration
    objective : str
        Objective: 'sharpe', 'return', 'min_vol'
    param_bounds : dict or None
        Bounds for parameters to optimize:
            - observation_noise: (min, max)
            - state_noise: (min, max)
            - initial_beta: (min, max)
    
    Returns:
    --------
    dict : Optimization results with optimal Kalman config and metrics
    """
    from kalman_filter import estimate_hedge_ratio_kalman
    from signal_generation import generate_signals, compute_position_sizes
    import config
    
    if param_bounds is None:
        param_bounds = {
            'observation_noise': (0.0001, 0.01),
            'state_noise': (0.00001, 0.001),
            'initial_beta': (0.5, 1.5)
        }
    
    # Extract data
    log_price_a = pair_data['log_price_A']
    log_price_b = pair_data['log_price_B']
    
    def objective_func(params):
        obs_noise, state_noise, init_beta = params
        
        # Update Kalman config
        test_config = kalman_config.copy()
        test_config['observation_noise'] = obs_noise
        test_config['state_noise'] = state_noise
        test_config['initial_beta'] = init_beta
        
        try:
            # Run Kalman filter
            kalman_result = estimate_hedge_ratio_kalman(pair_data, test_config)
            spread = kalman_result['spread']
            
            # Generate signals
            signals = generate_signals(pair_data, spread, config.SIGNAL_CONFIG)
            signals = compute_position_sizes(signals, pair_data, config.BACKTEST_CONFIG)
            
            # Compute returns
            strategy_returns = _compute_strategy_returns_simple(signals, pair_data)
            
            if len(strategy_returns) == 0:
                return np.inf
            
            # Objective
            if objective == 'sharpe':
                ann_ret = strategy_returns.mean() * 252
                ann_vol = strategy_returns.std() * np.sqrt(252)
                if ann_vol == 0:
                    return np.inf
                return -ann_ret / ann_vol  # Negative Sharpe
            
            elif objective == 'return':
                ann_ret = strategy_returns.mean() * 252
                return -ann_ret
            
            elif objective == 'min_vol':
                ann_vol = strategy_returns.std() * np.sqrt(252)
                return ann_vol
            
        except Exception as e:
            return np.inf
    
    # Bounds
    bounds = [
        param_bounds['observation_noise'],
        param_bounds['state_noise'],
        param_bounds['initial_beta']
    ]
    
    # Optimize
    result = differential_evolution(
        objective_func,
        bounds=bounds,
        seed=42,
        maxiter=100,
        popsize=15
    )
    
    # Get optimal config
    optimal_config = kalman_config.copy()
    optimal_config['observation_noise'] = result.x[0]
    optimal_config['state_noise'] = result.x[1]
    optimal_config['initial_beta'] = result.x[2]
    
    # Re-run with optimal config to get final results
    kalman_result = estimate_hedge_ratio_kalman(pair_data, optimal_config)
    spread = kalman_result['spread']
    
    signals = generate_signals(pair_data, spread, config.SIGNAL_CONFIG)
    signals = compute_position_sizes(signals, pair_data, config.BACKTEST_CONFIG)
    strategy_returns = _compute_strategy_returns_simple(signals, pair_data)
    
    metrics = _compute_portfolio_metrics(strategy_returns)
    
    return {
        'optimal_config': optimal_config,
        'kalman_result': kalman_result,
        'spread': spread,
        'signals': signals,
        'strategy_returns': strategy_returns,
        'metrics': metrics,
        'optimization_result': result
    }


def _compute_strategy_returns_simple(signals: pd.DataFrame, pair_data: pd.DataFrame) -> pd.Series:
    """Simple strategy returns computation."""
    common_idx = signals.index.intersection(pair_data.index)
    
    pos_a = signals.loc[common_idx, 'position_size_A']
    pos_b = signals.loc[common_idx, 'position_size_B']
    returns_a = pair_data.loc[common_idx, 'returns_A']
    returns_b = pair_data.loc[common_idx, 'returns_B']
    
    dollar_pnl = pos_a * returns_a + pos_b * returns_b
    notional = pos_a.abs() + pos_b.abs()
    
    prev_notional = notional.shift(1).fillna(notional)
    strategy_returns = dollar_pnl / prev_notional.replace(0, np.nan)
    strategy_returns = strategy_returns.fillna(0.0)
    
    return strategy_returns


def _compute_portfolio_metrics(returns: pd.Series) -> Dict:
    """Compute portfolio performance metrics."""
    if len(returns) == 0:
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0
        }
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    total_ret = (1 + returns).prod() - 1
    
    equity = (1 + returns).cumprod()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'total_return': total_ret,
        'max_drawdown': max_dd,
        'n_periods': len(returns)
    }


def compute_portfolio_returns(
    pair_returns: Dict[str, pd.Series],
    weights: Dict[str, float]) -> pd.Series:
    """
    Compute portfolio returns from pair returns and weights.
    
    Parameters:
    -----------
    pair_returns : dict
        Dictionary of {pair_name: returns_series}
    weights : dict
        Dictionary of {pair_name: weight}
    
    Returns:
    --------
    pd.Series : Portfolio returns
    """
    # Align all series
    returns_df = pd.DataFrame(pair_returns)
    returns_df = returns_df.dropna()
    
    # Get weights array in same order as columns
    weight_array = np.array([weights.get(pair, 0.0) for pair in returns_df.columns])
    
    # Compute portfolio returns
    portfolio_returns = pd.Series(
        returns_df.values @ weight_array,
        index=returns_df.index
    )
    
    return portfolio_returns

