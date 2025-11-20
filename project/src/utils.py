"""
Utility functions for data processing, metrics, and common operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


def compute_flow_proxy(shares_outstanding: pd.Series, price: pd.Series) -> pd.Series:
    """
    Compute primary market flow proxy.
    
    Flow_t ≈ (SO_t - SO_{t-1}) × P_t
    
    Parameters:
    -----------
    shares_outstanding : pd.Series
        Shares outstanding time series
    price : pd.Series
        Price time series (same index as shares_outstanding)
    
    Returns:
    --------
    pd.Series : Flow proxy in dollar terms
    """
    flow = (shares_outstanding.diff() * price).fillna(0)
    return flow


def compute_flow_pct_aum(shares_outstanding: pd.Series, price: pd.Series) -> pd.Series:
    """
    Compute flow as percentage of AUM.
    
    Flow%_t = (SO_t - SO_{t-1}) / SO_{t-1}
    
    Parameters:
    -----------
    shares_outstanding : pd.Series
        Shares outstanding time series
    price : pd.Series
        Price time series
    
    Returns:
    --------
    pd.Series : Flow as percentage of AUM
    """
    aum = shares_outstanding * price
    flow_pct = (aum.diff() / aum.shift(1)).fillna(0)
    return flow_pct


def compute_adv(volume: pd.Series, price: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute Average Daily Dollar Volume (ADV).
    
    Parameters:
    -----------
    volume : pd.Series
        Trading volume time series
    price : pd.Series
        Price time series
    window : int
        Rolling window for average (default: 20 days)
    
    Returns:
    --------
    pd.Series : ADV in dollars
    """
    dollar_volume = volume * price
    adv = dollar_volume.rolling(window=window, min_periods=1).mean()
    return adv


def compute_holdings_overlap(weights_a: Dict[str, float], 
                             weights_b: Dict[str, float]) -> float:
    """
    Compute holdings overlap between two ETFs.
    
    Uses Jaccard-like index: sum of min(weight_A, weight_B) across common holdings.
    
    Parameters:
    -----------
    weights_a : dict
        Dictionary of {ticker: weight} for ETF A
    weights_b : dict
        Dictionary of {ticker: weight} for ETF B
    
    Returns:
    --------
    float : Overlap metric (0 to 1, where 1 = identical weights)
    """
    if not weights_a or not weights_b:
        return 0.0
    
    common_tickers = set(weights_a.keys()) & set(weights_b.keys())
    if not common_tickers:
        return 0.0
    
    overlap = sum(min(weights_a.get(t, 0), weights_b.get(t, 0)) for t in common_tickers)
    return overlap


def compute_weighting_distance(weights_a: Dict[str, float], 
                               weights_b: Dict[str, float]) -> float:
    """
    Compute L1 distance between weight vectors.
    
    Parameters:
    -----------
    weights_a : dict
        Dictionary of {ticker: weight} for ETF A
    weights_b : dict
        Dictionary of {ticker: weight} for ETF B
    
    Returns:
    --------
    float : L1 distance (0 = identical, larger = more different)
    """
    if not weights_a or not weights_b:
        return float('inf')
    
    all_tickers = set(weights_a.keys()) | set(weights_b.keys())
    distance = sum(abs(weights_a.get(t, 0) - weights_b.get(t, 0)) for t in all_tickers)
    return distance


def align_dataframes(df_list: List[pd.DataFrame], 
                    how: str = 'inner',
                    fill_method: Optional[str] = None) -> pd.DataFrame:
    """
    Align multiple dataframes on index (date).
    
    Parameters:
    -----------
    df_list : list of DataFrames
        DataFrames to align
    how : str
        How to align: 'inner', 'outer', 'left', 'right'
    fill_method : str or None
        Forward fill method: 'ffill', 'bfill', or None
    
    Returns:
    --------
    pd.DataFrame : Aligned dataframe
    """
    if not df_list:
        return pd.DataFrame()
    
    result = df_list[0]
    for df in df_list[1:]:
        result = result.join(df, how=how, rsuffix='_other')
    
    if fill_method:
        result = result.fillna(method=fill_method)
    
    return result


def remove_outliers(series: pd.Series, method: str = 'iqr', 
                   factor: float = 3.0) -> pd.Series:
    """
    Remove outliers from a series.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
    method : str
        Method: 'iqr' (interquartile range) or 'zscore'
    factor : float
        Factor for outlier detection
    
    Returns:
    --------
    pd.Series : Series with outliers set to NaN
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (series >= lower_bound) & (series <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores < factor
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return series.where(mask)


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize return."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize volatility."""
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, 
                 periods_per_year: int = 252) -> float:
    """Compute Sharpe ratio."""
    ann_return = annualize_return(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute Calmar ratio (annual return / max drawdown)."""
    equity = (1 + returns).cumprod()
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return 0.0
    ann_return = annualize_return(returns, periods_per_year)
    return ann_return / dd

