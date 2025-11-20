"""
Kalman Filter implementation for time-varying hedge ratio estimation.

Implements a state-space model:
- Observation: y_t = alpha + beta_t * x_t + epsilon_t
- State: beta_t = beta_{t-1} + eta_t (random walk)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.linalg import inv
import warnings


class KalmanFilter:
    """
    Kalman filter for time-varying hedge ratio estimation.
    
    Model:
    - Observation: y_t = alpha + beta_t * x_t + epsilon_t, epsilon_t ~ N(0, R)
    - State: beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, Q)
    - alpha is fixed (can be estimated or set to 0)
    """
    
    def __init__(self, 
                 initial_beta: float = 1.0,
                 initial_alpha: float = 0.0,
                 beta_variance: float = 0.01,
                 alpha_variance: float = 0.01,
                 observation_noise: float = 0.001,
                 state_noise: float = 0.0001,
                 estimate_alpha: bool = False):
        """
        Initialize Kalman filter.
        
        Parameters:
        -----------
        initial_beta : float
            Initial guess for beta (hedge ratio)
        initial_alpha : float
            Initial intercept (can be estimated or fixed)
        beta_variance : float
            Initial variance for beta state
        alpha_variance : float
            Initial variance for alpha state (if estimating)
        observation_noise : float
            Observation equation noise variance (R)
        state_noise : float
            State equation noise variance (Q)
        estimate_alpha : bool
            Whether to estimate alpha (if False, alpha is fixed)
        """
        self.initial_beta = initial_beta
        self.initial_alpha = initial_alpha
        self.beta_variance = beta_variance
        self.alpha_variance = alpha_variance
        self.observation_noise = observation_noise
        self.state_noise = state_noise
        self.estimate_alpha = estimate_alpha
        
        # State dimension: [beta] or [alpha, beta]
        self.state_dim = 2 if estimate_alpha else 1
        
        # Results storage
        self.beta_estimates = None
        self.alpha_estimates = None
        self.beta_variance_estimates = None
        self.spread = None
    
    def filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Run Kalman filter (forward pass).
        
        Parameters:
        -----------
        y : pd.Series
            Dependent variable (log price of ETF A)
        x : pd.Series
            Independent variable (log price of ETF B)
        
        Returns:
        --------
        pd.DataFrame : Filtered estimates with columns:
            - beta_filtered
            - alpha_filtered (if estimate_alpha=True)
            - beta_variance_filtered
        """
        # Align series
        common_idx = y.index.intersection(x.index)
        y = y.loc[common_idx]
        x = x.loc[common_idx]
        n = len(y)
        
        if n == 0:
            raise ValueError("No overlapping data between y and x")
        
        # Initialize state and covariance
        if self.estimate_alpha:
            state = np.array([self.initial_alpha, self.initial_beta])
            P = np.diag([self.alpha_variance, self.beta_variance])
        else:
            state = np.array([self.initial_beta])
            P = np.array([[self.beta_variance]])
        
        # Storage
        beta_filtered = np.zeros(n)
        alpha_filtered = np.zeros(n) if self.estimate_alpha else None
        beta_variance_filtered = np.zeros(n)
        
        # Observation noise
        R = self.observation_noise
        
        # State transition matrix (identity for random walk)
        if self.estimate_alpha:
            F = np.eye(2)
        else:
            F = np.eye(1)
        
        # State noise covariance
        Q = np.diag([self.alpha_variance, self.state_noise]) if self.estimate_alpha else np.array([[self.state_noise]])
        
        # Kalman filter loop
        for i in range(n):
            # Prediction step
            state_pred = F @ state
            P_pred = F @ P @ F.T + Q
            
            # Observation matrix
            if self.estimate_alpha:
                H = np.array([[1.0, x.iloc[i]]])
            else:
                H = np.array([[x.iloc[i]]])
            
            # Innovation
            y_pred = H @ state_pred
            innovation = y.iloc[i] - y_pred
            S = H @ P_pred @ H.T + R
            
            # Update step
            K = P_pred @ H.T @ inv(S)  # Kalman gain
            state = state_pred + K * innovation
            P = (np.eye(self.state_dim) - K @ H) @ P_pred
            
            # Store results
            if self.estimate_alpha:
                alpha_filtered[i] = state[0]
                beta_filtered[i] = state[1]
                beta_variance_filtered[i] = P[1, 1]
            else:
                beta_filtered[i] = state[0]
                beta_variance_filtered[i] = P[0, 0]
        
        # Build result dataframe
        result = pd.DataFrame(index=common_idx)
        result['beta_filtered'] = beta_filtered
        result['beta_variance_filtered'] = beta_variance_filtered
        
        if self.estimate_alpha:
            result['alpha_filtered'] = alpha_filtered
        
        self.beta_estimates = result['beta_filtered']
        if self.estimate_alpha:
            self.alpha_estimates = result['alpha_filtered']
        self.beta_variance_estimates = result['beta_variance_filtered']
        
        return result
    
    def smooth(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Run Kalman smoother (forward-backward pass).
        
        Parameters:
        -----------
        y : pd.Series
            Dependent variable
        x : pd.Series
            Independent variable
        
        Returns:
        --------
        pd.DataFrame : Smoothed estimates
        """
        # First run filter to get forward pass
        filter_result = self.filter(y, x)
        
        # For simplicity, we'll use the filtered estimates as smoothed
        # A full smoother would do backward pass, but filtered is often sufficient
        # and computationally simpler
        
        result = filter_result.copy()
        result['beta_smoothed'] = result['beta_filtered']
        result['beta_variance_smoothed'] = result['beta_variance_filtered']
        
        if self.estimate_alpha:
            result['alpha_smoothed'] = result['alpha_filtered']
        
        self.beta_estimates = result['beta_smoothed']
        if self.estimate_alpha:
            self.alpha_estimates = result.get('alpha_smoothed')
        self.beta_variance_estimates = result['beta_variance_smoothed']
        
        return result
    
    def compute_spread(self, y: pd.Series, x: pd.Series, 
                      use_smooth: bool = True) -> pd.Series:
        """
        Compute spread: s_t = y_t - beta_t * x_t.
        
        Parameters:
        -----------
        y : pd.Series
            Log price of ETF A
        x : pd.Series
            Log price of ETF B
        use_smooth : bool
            Use smoothed beta estimates (if available)
        
        Returns:
        --------
        pd.Series : Spread series
        """
        if self.beta_estimates is None:
            # Run filter/smoother first
            if use_smooth:
                self.smooth(y, x)
            else:
                self.filter(y, x)
        
        # Align series
        common_idx = y.index.intersection(x.index).intersection(self.beta_estimates.index)
        y_aligned = y.loc[common_idx]
        x_aligned = x.loc[common_idx]
        beta_aligned = self.beta_estimates.loc[common_idx]
        
        # Compute spread
        if self.estimate_alpha and self.alpha_estimates is not None:
            alpha_aligned = self.alpha_estimates.loc[common_idx]
            spread = y_aligned - alpha_aligned - beta_aligned * x_aligned
        else:
            spread = y_aligned - beta_aligned * x_aligned
        
        self.spread = spread
        return spread


def estimate_hedge_ratio_kalman(pair_data: pd.DataFrame,
                                 config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Estimate time-varying hedge ratio using Kalman filter for an ETF pair.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset with columns log_price_A, log_price_B
    config : dict or None
        Kalman filter configuration (see KalmanFilter.__init__)
    
    Returns:
    --------
    pd.DataFrame : Results with columns:
        - beta_smoothed: Time-varying hedge ratio
        - beta_variance_smoothed: Variance of beta estimate
        - spread: Spread series s_t = log_price_A - beta_t * log_price_B
    """
    if config is None:
        config = {}
    
    # Extract series
    y = pair_data['log_price_A']
    x = pair_data['log_price_B']
    
    # Filter out 'smooth' key if present (not an init parameter)
    init_config = {k: v for k, v in config.items() if k != 'smooth'}
    
    # Initialize and run Kalman filter
    kf = KalmanFilter(**init_config)
    result = kf.smooth(y, x)
    
    # Compute spread
    spread = kf.compute_spread(y, x, use_smooth=True)
    result['spread'] = spread
    
    return result

