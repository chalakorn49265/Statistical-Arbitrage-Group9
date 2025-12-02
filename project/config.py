"""
Configuration parameters for the pairs trading project.
Modify these to adjust strategy parameters, data sources, and backtest settings.
"""

import pandas as pd
from datetime import datetime

# ============================================================================
# ETF PAIRS CONFIGURATION
# ============================================================================

# Define ETF pairs: (ETF_A, ETF_B, sector_name)
# Original 5 pairs (for backward compatibility)
ETF_PAIRS = [
    ('XLE', 'VDE', 'Energy'),
    ('XLK', 'FTEC', 'Technology'),
    ('XLF', 'VFH', 'Financials'),
    ('XLI', 'VIS', 'Industrials'),
    ('XLV', 'VHT', 'Healthcare'),
]

# Expanded candidate pairs (20-50 pairs for correlation analysis and selection)
# Import from pair_selection module for full list
# Use EXPANDED_ETF_PAIRS from src.pair_selection for comprehensive analysis

# Alternative: specify pairs as dictionary with metadata
# ETF_PAIRS_DICT = {
#     ('XLE', 'VDE'): {'sector': 'Energy', 'start_date': '2010-01-01'},
#     ('XLK', 'FTEC'): {'sector': 'Technology', 'start_date': '2013-10-24'},
# }

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Date range for analysis
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'

# Data file paths (modify based on your data source)
DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

# Data columns expected in price data
PRICE_COLS = {
    'date': 'Date',
    'symbol': 'Symbol',
    'close': 'Close',
    'adj_close': 'Adj Close',
    'volume': 'Volume',
    'shares_outstanding': 'Shares Outstanding',
}

# ============================================================================
# KALMAN FILTER PARAMETERS
# ============================================================================

KALMAN_CONFIG = {
    'initial_beta': 1.0,           # Initial hedge ratio guess
    'initial_alpha': 0.0,           # Initial intercept
    'beta_variance': 0.01,          # Initial variance for beta state
    'alpha_variance': 0.01,         # Initial variance for alpha state
    'observation_noise': 0.001,      # Observation equation noise variance
    'state_noise': 0.0001,          # State equation noise variance (beta random walk)
    'smooth': True,                  # Use Kalman smoother (vs. filter only)
}

# ============================================================================
# MEAN REVERSION PARAMETERS
# ============================================================================

MEAN_REVERSION_CONFIG = {
    'ar1_window': 252,              # Rolling window for AR(1) estimation (trading days)
    'min_half_life': 5,              # Minimum half-life (days) to consider pair
    'max_half_life': 252,            # Maximum half-life (days) to consider pair
    'adf_lags': 5,                   # Lags for ADF test
    'adf_critical': 0.05,            # Critical p-value for stationarity
}

# ============================================================================
# SIGNAL GENERATION PARAMETERS
# ============================================================================

SIGNAL_CONFIG = {
    'z_entry': 2.0,                  # Z-score threshold for entry
    'z_exit': 0.5,                  # Z-score threshold for exit
    'z_window': 60,                  # Rolling window for z-score calculation (trading days)
    'max_holding_days': 60,          # Maximum holding period (days)
    'stop_loss_pct': 0.05,           # Stop loss as % of entry spread
    'flow_threshold_pctile': 75,     # Percentile threshold for flow differential
    'min_overlap': 0.3,              # Minimum holdings overlap to trade pair
    'max_weighting_distance': 0.5,   # Maximum L1 weighting distance to trade pair
}

# ============================================================================
# BACKTESTING PARAMETERS
# ============================================================================

BACKTEST_CONFIG = {
    'train_window_years': 2,         # Training window size (years)
    'test_window_years': 0.5,        # Test window size (years)
    'purge_days': 10,                # Days to purge between train/test
    'embargo_days': 5,               # Days to embargo around test periods
    'rebalance_freq': 'M',           # Rebalance frequency: 'D', 'W', 'M'
    'transaction_cost_bps': 10,      # Transaction cost in basis points (5 bps per side = 10 bps round trip)
    'spread_cost_bps': 5,            # Half-spread cost in bps (per side)
    'capacity_pct_adv': 0.10,        # Max position size as % of 20-day ADV
    'min_adv_days': 20,              # Days for ADV calculation
    'initial_capital': 1_000_000,    # Initial capital for backtest
    'max_leverage': 2.0,             # Maximum leverage (2x = long $2 for every $1 capital)
}

# Parameter grid for optimization
PARAM_GRID = {
    'z_entry': [1.5, 2.0, 2.5, 3.0],
    'z_exit': [0.3, 0.5, 0.7, 1.0],
    'z_window': [30, 60, 90, 120],
    'flow_threshold_pctile': [70, 75, 80, 85],
    'max_holding_days': [30, 45, 60, 90],
}

# ============================================================================
# ROBUSTNESS CONFIGURATION
# ============================================================================

ROBUSTNESS_CONFIG = {
    'spa_bootstrap_reps': 1000,      # Bootstrap replications for SPA test
    'spa_confidence': 0.95,         # Confidence level for SPA
    'structural_break_window': 252,  # Window for rolling parameter estimation
    'chow_test_min_obs': 60,          # Minimum observations per segment for Chow test
    'event_window_pre': 5,            # Days before event for event study
    'event_window_post': 10,          # Days after event for event study
    'large_flow_threshold_pctile': 95, # Percentile for "large" flow events
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
TABLES_DIR = 'results/tables'
BACKTESTS_DIR = 'results/backtests'

# Plotting style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 300

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_trading_calendar(start_date, end_date):
    """Get trading calendar (business days) for date range."""
    return pd.bdate_range(start=start_date, end=end_date)

def validate_config():
    """Validate configuration parameters."""
    assert SIGNAL_CONFIG['z_entry'] > SIGNAL_CONFIG['z_exit'], \
        "z_entry must be greater than z_exit"
    assert BACKTEST_CONFIG['train_window_years'] > 0, \
        "train_window_years must be positive"
    assert BACKTEST_CONFIG['transaction_cost_bps'] >= 0, \
        "transaction_cost_bps must be non-negative"
    return True

