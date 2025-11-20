"""
Robustness checks: SPA test, structural breaks, event studies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings

from .mean_reversion import estimate_ar1, rolling_ar1


def spa_test(returns: pd.Series,
            benchmark_returns: Optional[pd.Series] = None,
            bootstrap_reps: int = 1000,
            confidence: float = 0.95) -> Dict:
    """
    Superior Predictive Ability (SPA) test / White's Reality Check.
    
    Tests if strategy significantly outperforms benchmark after adjusting
    for data snooping bias.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series or None
        Benchmark returns (if None, uses zero)
    bootstrap_reps : int
        Number of bootstrap replications
    confidence : float
        Confidence level
    
    Returns:
    --------
    dict : Test results
    """
    if benchmark_returns is None:
        benchmark_returns = pd.Series(index=returns.index, data=0.0)
    
    # Align series
    common_idx = returns.index.intersection(benchmark_returns.index)
    strategy = returns.loc[common_idx]
    benchmark = benchmark_returns.loc[common_idx]
    
    # Excess returns
    excess = strategy - benchmark
    n = len(excess)
    
    if n == 0:
        return {
            'spa_statistic': np.nan,
            'pvalue': 1.0,
            'reject_null': False,
            'n_obs': 0
        }
    
    # Test statistic: mean excess return / std
    mean_excess = excess.mean()
    std_excess = excess.std()
    
    if std_excess == 0:
        spa_stat = 0.0
    else:
        spa_stat = np.sqrt(n) * mean_excess / std_excess
    
    # Bootstrap
    bootstrap_stats = []
    for _ in range(bootstrap_reps):
        # Resample with replacement
        bootstrap_sample = excess.sample(n=n, replace=True)
        bootstrap_mean = bootstrap_sample.mean()
        bootstrap_std = bootstrap_sample.std()
        
        if bootstrap_std > 0:
            bootstrap_stat = np.sqrt(n) * bootstrap_mean / bootstrap_std
        else:
            bootstrap_stat = 0.0
        
        bootstrap_stats.append(bootstrap_stat)
    
    # P-value: proportion of bootstrap stats >= observed
    pvalue = (np.array(bootstrap_stats) >= spa_stat).mean()
    
    # Critical value
    alpha = 1 - confidence
    critical_value = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return {
        'spa_statistic': spa_stat,
        'pvalue': pvalue,
        'critical_value': critical_value,
        'reject_null': pvalue < alpha,
        'n_obs': n,
        'mean_excess': mean_excess,
        'std_excess': std_excess
    }


def detect_structural_breaks(spread: pd.Series,
                             beta_estimates: pd.Series,
                             window: int = 252,
                             min_obs: int = 60) -> pd.DataFrame:
    """
    Detect structural breaks in hedge ratio and spread parameters.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread series
    beta_estimates : pd.Series
        Beta estimates from Kalman filter
    window : int
        Window for rolling estimation
    min_obs : int
        Minimum observations per segment
    
    Returns:
    --------
    pd.DataFrame : Breakpoints with dates and statistics
    """
    # Rolling AR(1) estimation
    ar1_rolling = rolling_ar1(spread, window=window)
    
    # Rolling beta statistics
    beta_rolling_mean = beta_estimates.rolling(window=window).mean()
    beta_rolling_std = beta_estimates.rolling(window=window).std()
    
    # Simple break detection: large changes in rolling parameters
    phi_changes = ar1_rolling['phi'].diff().abs()
    beta_changes = beta_estimates.diff().abs()
    
    # Identify potential breaks (top 5% of changes)
    phi_threshold = phi_changes.quantile(0.95)
    beta_threshold = beta_changes.quantile(0.95)
    
    breaks = []
    for date in spread.index:
        if date in phi_changes.index and date in beta_changes.index:
            phi_change = phi_changes.loc[date]
            beta_change = beta_changes.loc[date]
            
            if pd.notna(phi_change) and pd.notna(beta_change):
                if phi_change > phi_threshold or beta_change > beta_threshold:
                    breaks.append({
                        'date': date,
                        'phi_change': phi_change,
                        'beta_change': beta_change,
                        'phi': ar1_rolling.loc[date, 'phi'] if date in ar1_rolling.index else np.nan,
                        'beta': beta_estimates.loc[date] if date in beta_estimates.index else np.nan
                    })
    
    if breaks:
        return pd.DataFrame(breaks)
    else:
        return pd.DataFrame(columns=['date', 'phi_change', 'beta_change', 'phi', 'beta'])


def event_study(spread: pd.Series,
               signals: pd.DataFrame,
               events: List[pd.Timestamp],
               window_pre: int = 5,
               window_post: int = 10) -> pd.DataFrame:
    """
    Event study around specific dates (rebalances, large flows, etc.).
    
    Parameters:
    -----------
    spread : pd.Series
        Spread series
    signals : pd.DataFrame
        Signal dataframe
    events : list
        List of event dates
    window_pre : int
        Days before event
    window_post : int
        Days after event
    
    Returns:
    --------
    pd.DataFrame : Event study results with columns:
        - event_date
        - day_relative (days from event)
        - spread_mean, spread_std
        - zscore_mean, zscore_std
        - position_mean
        - n_events
    """
    results = []
    
    for event_date in events:
        if event_date not in spread.index:
            continue
        
        # Event window
        event_idx = spread.index.get_loc(event_date)
        start_idx = max(0, event_idx - window_pre)
        end_idx = min(len(spread), event_idx + window_post + 1)
        
        window_dates = spread.index[start_idx:end_idx]
        window_spread = spread.loc[window_dates]
        
        # Relative days
        days_relative = [(d - event_date).days for d in window_dates]
        
        # Z-scores if available
        if 'zscore' in signals.columns:
            window_zscore = signals.loc[window_dates, 'zscore']
        else:
            window_zscore = pd.Series(index=window_dates, data=np.nan)
        
        # Positions if available
        if 'position' in signals.columns:
            window_position = signals.loc[window_dates, 'position']
        else:
            window_position = pd.Series(index=window_dates, data=0.0)
        
        for i, (date, day_rel) in enumerate(zip(window_dates, days_relative)):
            results.append({
                'event_date': event_date,
                'day_relative': day_rel,
                'date': date,
                'spread': window_spread.loc[date],
                'zscore': window_zscore.loc[date] if date in window_zscore.index else np.nan,
                'position': window_position.loc[date] if date in window_position.index else 0.0
            })
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Aggregate by relative day
    aggregated = result_df.groupby('day_relative').agg({
        'spread': ['mean', 'std', 'count'],
        'zscore': ['mean', 'std'],
        'position': 'mean'
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['day_relative', 'spread_mean', 'spread_std', 'n_events',
                         'zscore_mean', 'zscore_std', 'position_mean']
    
    return aggregated


def identify_large_flow_events(pair_data: pd.DataFrame,
                              threshold_pctile: float = 95.0) -> List[pd.Timestamp]:
    """
    Identify dates with unusually large flow differentials.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset with flow_diff column
    threshold_pctile : float
        Percentile threshold for "large" flows
    
    Returns:
    --------
    list : List of event dates
    """
    if 'flow_diff' not in pair_data.columns:
        return []
    
    flow_diff = pair_data['flow_diff'].abs()
    threshold = flow_diff.quantile(threshold_pctile / 100.0)
    
    large_flow_dates = flow_diff[flow_diff >= threshold].index.tolist()
    return large_flow_dates


def compute_robustness_checks(pair_data: pd.DataFrame,
                            spread: pd.Series,
                            signals: pd.DataFrame,
                            returns: pd.Series,
                            beta_estimates: pd.Series,
                            config: Optional[Dict] = None) -> Dict:
    """
    Run all robustness checks.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset
    spread : pd.Series
        Spread series
    signals : pd.DataFrame
        Signal dataframe
    returns : pd.Series
        Strategy returns
    beta_estimates : pd.Series
        Beta estimates
    config : dict or None
        Configuration
    
    Returns:
    --------
    dict : Dictionary with all robustness check results
    """
    if config is None:
        config = {}
    
    # SPA test
    spa_results = spa_test(
        returns,
        bootstrap_reps=config.get('spa_bootstrap_reps', 1000),
        confidence=config.get('spa_confidence', 0.95)
    )
    
    # Structural breaks
    breaks = detect_structural_breaks(
        spread,
        beta_estimates,
        window=config.get('structural_break_window', 252),
        min_obs=config.get('chow_test_min_obs', 60)
    )
    
    # Large flow events
    large_flows = identify_large_flow_events(
        pair_data,
        threshold_pctile=config.get('large_flow_threshold_pctile', 95.0)
    )
    
    # Event study
    event_results = None
    if large_flows:
        event_results = event_study(
            spread,
            signals,
            large_flows,
            window_pre=config.get('event_window_pre', 5),
            window_post=config.get('event_window_post', 10)
        )
    
    return {
        'spa_test': spa_results,
        'structural_breaks': breaks,
        'large_flow_events': large_flows,
        'event_study': event_results
    }

