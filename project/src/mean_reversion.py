"""
Mean reversion modeling: AR(1) / Ornstein-Uhlenbeck process estimation.

Models spread as: s_t = mu + phi * (s_{t-1} - mu) + epsilon_t
Computes half-life and other mean reversion metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import warnings


def estimate_ar1(spread: pd.Series, 
                 window: Optional[int] = None,
                 method: str = 'ols') -> Dict:
    """
    Estimate AR(1) model: s_t = mu + phi * (s_{t-1} - mu) + epsilon_t.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    window : int or None
        Rolling window (if None, uses all data)
    method : str
        Estimation method: 'ols' or 'statsmodels'
    
    Returns:
    --------
    dict : Dictionary with keys:
        - mu: Mean reversion level
        - phi: Mean reversion speed (AR coefficient)
        - sigma: Residual standard deviation
        - half_life: Half-life in periods
        - theta: Continuous-time OU parameter (if phi < 1)
        - sigma_ou: Continuous-time OU volatility
    """
    if window is None:
        # Use all data
        data = spread.dropna()
    else:
        # Use rolling window
        if len(spread) < window:
            warnings.warn(f"Window {window} larger than data length {len(spread)}")
            data = spread.dropna()
        else:
            data = spread.iloc[-window:].dropna()
    
    if len(data) < 2:
        return {
            'mu': np.nan,
            'phi': np.nan,
            'sigma': np.nan,
            'half_life': np.nan,
            'theta': np.nan,
            'sigma_ou': np.nan,
            'n_obs': len(data)
        }
    
    # Remove mean for estimation
    s_mean = data.mean()
    s_centered = data - s_mean
    
    # Estimate AR(1)
    if method == 'statsmodels':
        try:
            model = AutoReg(s_centered, lags=1, trend='n')
            res = model.fit()
            phi = res.params[0]
            sigma = np.sqrt(res.sigma2)
        except:
            # Fallback to OLS
            method = 'ols'
    
    if method == 'ols':
        # Manual OLS: s_t = phi * s_{t-1} + epsilon_t
        s_lag = s_centered.shift(1).dropna()
        s_current = s_centered.loc[s_lag.index]
        
        if len(s_current) < 2:
            return {
                'mu': s_mean,
                'phi': np.nan,
                'sigma': np.nan,
                'half_life': np.nan,
                'theta': np.nan,
                'sigma_ou': np.nan,
                'n_obs': len(data)
            }
        
        # OLS regression
        phi = np.dot(s_lag, s_current) / np.dot(s_lag, s_lag)
        residuals = s_current - phi * s_lag
        sigma = residuals.std()
    
    # Ensure stationarity
    if abs(phi) >= 1:
        phi = np.sign(phi) * 0.99  # Clip to stationary region
    
    # Compute half-life
    if phi > 0:
        half_life = -np.log(2) / np.log(phi)
    else:
        half_life = np.nan
    
    # Continuous-time OU parameters (if we assume daily observations)
    # dS_t = theta * (mu - S_t) * dt + sigma_ou * dW_t
    # Discrete: S_t = mu + phi * (S_{t-1} - mu) + epsilon_t
    # phi = exp(-theta * dt), where dt = 1 day
    if phi > 0 and phi < 1:
        theta = -np.log(phi)  # Mean reversion speed (per day)
        sigma_ou = sigma / np.sqrt((1 - phi**2) / (2 * theta)) if theta > 0 else np.nan
    else:
        theta = np.nan
        sigma_ou = np.nan
    
    return {
        'mu': s_mean,
        'phi': phi,
        'sigma': sigma,
        'half_life': half_life,
        'theta': theta,
        'sigma_ou': sigma_ou,
        'n_obs': len(data)
    }


def rolling_ar1(spread: pd.Series, 
                window: int = 252,
                min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Estimate rolling AR(1) parameters.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    window : int
        Rolling window size
    min_periods : int or None
        Minimum periods required (default: window // 2)
    
    Returns:
    --------
    pd.DataFrame : Rolling estimates with columns:
        - mu, phi, sigma, half_life, theta, sigma_ou
    """
    if min_periods is None:
        min_periods = max(window // 2, 10)
    
    results = []
    dates = []
    
    for i in range(min_periods, len(spread) + 1):
        window_data = spread.iloc[:i]
        if len(window_data) >= window:
            window_data = spread.iloc[i-window:i]
        
        est = estimate_ar1(window_data, window=None)
        results.append(est)
        dates.append(spread.index[i-1])
    
    result_df = pd.DataFrame(results, index=dates)
    return result_df


def adf_test(spread: pd.Series, 
             lags: Optional[int] = None,
             maxlags: int = 10) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    lags : int or None
        Number of lags (if None, auto-selected)
    maxlags : int
        Maximum lags for auto-selection
    
    Returns:
    --------
    dict : Test results with keys:
        - adf_statistic: ADF test statistic
        - pvalue: p-value
        - critical_values: Dict of critical values
        - is_stationary: Boolean (p-value < 0.05)
    """
    data = spread.dropna()
    if len(data) < maxlags + 5:
        return {
            'adf_statistic': np.nan,
            'pvalue': 1.0,
            'critical_values': {},
            'is_stationary': False
        }
    
    if lags is None:
        # Auto-select lags using AIC
        result = adfuller(data, maxlags=maxlags, autolag='AIC')
    else:
        result = adfuller(data, maxlags=lags, autolag=None)
    
    adf_stat, pvalue, usedlag, nobs, critical_values, icbest = result
    
    return {
        'adf_statistic': adf_stat,
        'pvalue': pvalue,
        'critical_values': critical_values,
        'is_stationary': pvalue < 0.05,
        'used_lags': usedlag,
        'n_obs': nobs
    }


def compute_mean_reversion_metrics(spread: pd.Series,
                                   config: Optional[Dict] = None) -> Dict:
    """
    Compute comprehensive mean reversion metrics.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    config : dict or None
        Configuration with keys:
            - ar1_window: Window for AR(1) estimation
            - adf_lags: Lags for ADF test
    
    Returns:
    --------
    dict : Dictionary with all mean reversion metrics
    """
    if config is None:
        config = {}
    
    ar1_window = config.get('ar1_window', 252)
    adf_lags = config.get('adf_lags', 5)
    
    # AR(1) estimation
    ar1_results = estimate_ar1(spread, window=ar1_window)
    
    # ADF test
    adf_results = adf_test(spread, maxlags=adf_lags)
    
    # Combine results
    results = {
        **ar1_results,
        **adf_results,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
        'spread_min': spread.min(),
        'spread_max': spread.max(),
    }
    
    return results

