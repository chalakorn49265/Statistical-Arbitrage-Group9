# Pairs Trading Across Sector ETFs: Mean Reversion from Weighting & Flow Frictions

**MGMTMFE-413 Statistical Arbitrage Research Project**

## Project Overview

This project implements a pairs trading strategy on same-sector ETF pairs (e.g., XLE–VDE, XLK–FTEC) that:
- Track similar sector benchmarks but differ in holdings, weights, and primary-market flows
- Uses time-varying hedge ratios via Kalman filtering
- Models spread mean reversion as an OU/AR(1) process
- Conditions signals on flow differentials and holdings overlap metrics
- Implements robust backtesting with purged & embargoed walk-forward splits

## Project Structure

```
project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration parameters
├── data/                     # Data storage
│   ├── raw/                  # Raw data files
│   └── processed/           # Processed data files
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── kalman_filter.py      # Kalman filter for hedge ratio estimation
│   ├── mean_reversion.py     # OU/AR(1) modeling and half-life
│   ├── signal_generation.py  # Z-score signals with flow/weighting conditions
│   ├── backtesting.py        # Walk-forward backtesting framework
│   ├── performance.py        # Performance metrics and evaluation
│   ├── robustness.py         # SPA, structural breaks, event studies
│   └── utils.py              # Utility functions
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_hedge_ratio_estimation.ipynb
│   ├── 03_mean_reversion_analysis.ipynb
│   ├── 04_signal_generation.ipynb
│   ├── 05_backtesting.ipynb
│   └── 06_robustness_checks.ipynb
├── results/                  # Output files
│   ├── figures/              # Plots and visualizations
│   ├── tables/               # Summary tables
│   └── backtests/            # Backtest results
└── final_proj.ipynb          # Main project notebook

```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up data directories:
```bash
mkdir -p data/raw data/processed results/figures results/tables results/backtests
```

## Usage

1. **Data Preparation**: Place your ETF data in `data/raw/` or modify `src/data_loader.py` to connect to your data source (WRDS/Refinitiv).

2. **Run Analysis**: Use the notebooks in `notebooks/` or the main `final_proj.ipynb` to execute the full pipeline.

3. **Configuration**: Adjust parameters in `config.py` for different ETF pairs, date ranges, and strategy parameters.

## Key Features

- **Time-Varying Hedge Ratios**: Kalman filter implementation for dynamic beta estimation
- **Mean Reversion Modeling**: AR(1) estimation with half-life calculation
- **Flow-Based Signals**: Incorporates primary market flow differentials
- **Holdings Overlap Metrics**: Jaccard and L1 distance measures for pair similarity
- **Robust Backtesting**: Purged and embargoed walk-forward validation
- **Transaction Costs**: Realistic cost assumptions (commissions + spreads)
- **Capacity Constraints**: Position sizing based on ADV limits
- **Robustness Checks**: SPA test, structural break detection, event studies

## Data Requirements

- Daily ETF prices (adjusted close, volume)
- Shares outstanding (for flow calculation)
- Holdings snapshots (ticker, weight) - periodic updates
- Sector classifications (optional, for beta neutrality)

## Output

The project generates:
- Hedge ratio time series
- Spread series and z-scores
- Trade lists with entry/exit points
- Performance metrics (Sharpe, drawdown, turnover, etc.)
- Robustness diagnostics
- Event study results

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule

