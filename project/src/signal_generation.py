"""
Signal generation for pairs trading strategy.

Generates trading signals based on:
- Z-scores of spread
- Flow differentials
- Holdings overlap and weighting distance conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score of spread.
    
    z_t = (s_t - mean(s_{t-window:t})) / std(s_{t-window:t})
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    window : int
        Rolling window for mean/std calculation
    
    Returns:
    --------
    pd.Series : Z-score series
    """
    rolling_mean = spread.rolling(window=window, min_periods=window//2).mean()
    rolling_std = spread.rolling(window=window, min_periods=window//2).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def generate_signals(pair_data: pd.DataFrame,
                    spread: pd.Series,
                    config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Generate trading signals for pairs trading strategy.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset with flow, volume, etc.
    spread : pd.Series
        Spread series (indexed by date)
    config : dict or None
        Signal configuration with keys:
            - z_entry: Z-score threshold for entry
            - z_exit: Z-score threshold for exit
            - z_window: Window for z-score calculation
            - max_holding_days: Maximum holding period
            - stop_loss_pct: Stop loss as % of entry spread
            - flow_threshold_pctile: Percentile threshold for flow differential
            - use_flow_filter: Whether to use flow conditioning
    
    Returns:
    --------
    pd.DataFrame : Signal dataframe with columns:
        - zscore: Z-score of spread
        - signal: Trading signal (-1, 0, 1)
        - position: Current position (-1, 0, 1)
        - entry_date: Date of entry (if in position)
        - entry_zscore: Z-score at entry
        - entry_spread: Spread value at entry
        - days_held: Days since entry
        - flow_diff: Flow differential
        - flow_diff_pctile: Percentile of flow differential
    """
    if config is None:
        config = {}
    
    # Extract parameters
    z_entry = config.get('z_entry', 2.0)
    z_exit = config.get('z_exit', 0.5)
    z_window = config.get('z_window', 60)
    max_holding_days = config.get('max_holding_days', 60)
    stop_loss_pct = config.get('stop_loss_pct', 0.05)
    flow_threshold_pctile = config.get('flow_threshold_pctile', 75)
    use_flow_filter = config.get('use_flow_filter', True)
    
    # Align data
    common_idx = spread.index.intersection(pair_data.index)
    spread_aligned = spread.loc[common_idx]
    
    # Compute z-score
    zscore = compute_zscore(spread_aligned, window=z_window)
    
    # Initialize result dataframe
    result = pd.DataFrame(index=common_idx)
    result['spread'] = spread_aligned
    result['zscore'] = zscore
    
    # Flow differential
    if 'flow_diff' in pair_data.columns:
        result['flow_diff'] = pair_data.loc[common_idx, 'flow_diff']
        result['flow_diff_pct'] = pair_data.loc[common_idx, 'flow_diff_pct']
        # Compute percentile
        flow_window = min(252, len(result))
        result['flow_diff_pctile'] = result['flow_diff'].rolling(
            window=flow_window, min_periods=flow_window//2
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
    else:
        result['flow_diff'] = 0.0
        result['flow_diff_pctile'] = 50.0
    
    # Initialize signal and position columns
    result['signal'] = 0
    result['position'] = 0
    result['entry_date'] = pd.NaT
    result['entry_zscore'] = np.nan
    result['entry_spread'] = np.nan
    result['days_held'] = 0
    
    # Generate signals
    position = 0
    entry_date = None
    entry_zscore = None
    entry_spread = None
    
    for i, date in enumerate(common_idx):
        z = zscore.loc[date]
        s = spread_aligned.loc[date]
        flow_pctile = result.loc[date, 'flow_diff_pctile']
        
        if pd.isna(z) or pd.isna(s):
            result.loc[date, 'position'] = position
            continue
        
        # Check exit conditions first
        if position != 0:
            days_held = (date - entry_date).days if entry_date else 0
            result.loc[date, 'days_held'] = days_held
            
            # Exit conditions
            exit_signal = False
            
            # Z-score exit
            if abs(z) < z_exit:
                exit_signal = True
            
            # Maximum holding period
            if days_held >= max_holding_days:
                exit_signal = True
            
            # Stop loss
            if entry_spread is not None:
                if position > 0:  # Long spread
                    if s < entry_spread * (1 - stop_loss_pct):
                        exit_signal = True
                else:  # Short spread
                    if s > entry_spread * (1 + stop_loss_pct):
                        exit_signal = True
            
            if exit_signal:
                result.loc[date, 'signal'] = -position
                position = 0
                entry_date = None
                entry_zscore = None
                entry_spread = None
            else:
                result.loc[date, 'position'] = position
                result.loc[date, 'entry_date'] = entry_date
                result.loc[date, 'entry_zscore'] = entry_zscore
                result.loc[date, 'entry_spread'] = entry_spread
            continue
        
        # Entry conditions (only if not in position)
        if position == 0:
            # Check flow filter
            flow_ok = True
            if use_flow_filter and not pd.isna(flow_pctile):
                # Only enter if flow differential is extreme
                flow_ok = abs(flow_pctile - 50) > (100 - flow_threshold_pctile)
            
            # Entry signals
            if z < -z_entry and flow_ok:
                # Long spread (long A, short B)
                position = 1
                entry_date = date
                entry_zscore = z
                entry_spread = s
                result.loc[date, 'signal'] = 1
            
            elif z > z_entry and flow_ok:
                # Short spread (short A, long B)
                position = -1
                entry_date = date
                entry_zscore = z
                entry_spread = s
                result.loc[date, 'signal'] = -1
        
        result.loc[date, 'position'] = position
        if entry_date:
            result.loc[date, 'entry_date'] = entry_date
            result.loc[date, 'entry_zscore'] = entry_zscore
            result.loc[date, 'entry_spread'] = entry_spread
    
    return result


def compute_position_sizes(signals: pd.DataFrame,
                          pair_data: pd.DataFrame,
                          config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Compute position sizes based on capacity constraints.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Signal dataframe from generate_signals
    pair_data : pd.DataFrame
        Pair dataset with ADV information
    config : dict or None
        Configuration with keys:
            - capacity_pct_adv: Max position as % of ADV
            - initial_capital: Initial capital
            - max_leverage: Maximum leverage
    
    Returns:
    --------
    pd.DataFrame : Signals with position size columns:
        - position_size_A: Dollar position in ETF A
        - position_size_B: Dollar position in ETF B
        - notional: Total notional exposure
    """
    if config is None:
        config = {}
    
    capacity_pct = config.get('capacity_pct_adv', 0.10)
    initial_capital = config.get('initial_capital', 1_000_000)
    max_leverage = config.get('max_leverage', 2.0)
    
    # Align data
    common_idx = signals.index.intersection(pair_data.index)
    result = signals.loc[common_idx].copy()
    
    # Get ADV
    adv_a = pair_data.loc[common_idx, 'adv_A'] if 'adv_A' in pair_data.columns else pd.Series(index=common_idx, data=np.nan)
    adv_b = pair_data.loc[common_idx, 'adv_B'] if 'adv_B' in pair_data.columns else pd.Series(index=common_idx, data=np.nan)
    
    # Initialize position size columns
    result['position_size_A'] = 0.0
    result['position_size_B'] = 0.0
    result['notional'] = 0.0
    
    for date in common_idx:
        position = result.loc[date, 'position']
        
        if position == 0:
            continue
        
        # Capacity constraint: min of ADV_A and ADV_B
        adv_min = min(adv_a.loc[date], adv_b.loc[date]) if not (pd.isna(adv_a.loc[date]) or pd.isna(adv_b.loc[date])) else initial_capital
        
        if pd.isna(adv_min) or adv_min <= 0:
            adv_min = initial_capital
        
        # Maximum notional based on capacity
        max_notional = adv_min * capacity_pct * max_leverage
        
        # Position sizing: equal dollar amounts for long/short legs
        if position > 0:
            # Long A, short B
            result.loc[date, 'position_size_A'] = max_notional / 2
            result.loc[date, 'position_size_B'] = -max_notional / 2
        else:
            # Short A, long B
            result.loc[date, 'position_size_A'] = -max_notional / 2
            result.loc[date, 'position_size_B'] = max_notional / 2
        
        result.loc[date, 'notional'] = max_notional
    
    return result

