# Pairs Trading Strategy: Methodology Steps

## Slide 1: Steps 1-3

### Step 1: Kalman Filter Hedge Ratio Estimation
**Objective:** Estimate time-varying hedge ratios (β_t) for each ETF pair

**Method:**
- State-space model: y_t = α + β_t × x_t + ε_t (observation equation)
- State evolution: β_t = β_{t-1} + η_t (random walk)
- Kalman smoother to estimate β_t using full information set
- Parameters: observation_noise (0.001), state_noise (0.0001), initial_beta (1.0)

**Outputs:**
- Time-varying hedge ratio series (β_smoothed) for each pair
- Spread series: s_t = log_price_A - β_t × log_price_B
- Beta variance estimates (uncertainty quantification)

**Pairs Analyzed:** 5 same-sector ETF pairs (Energy, Technology, Financials, Industrials, Healthcare)

---

### Step 2: Trade Signal Generation
**Objective:** Generate entry/exit signals based on mean reversion and flow conditions

**Signal Logic:**
- Compute rolling z-score: z_t = (s_t - μ_{t-window}) / σ_{t-window} (60-day window)
- **Entry Conditions:**
  - Long spread when: z < -2.0 AND flow differential > 75th percentile
  - Short spread when: z > +2.0 AND flow differential > 75th percentile
- **Exit Conditions:**
  - |z| < 0.5 (mean reversion complete)
  - Maximum holding period: 60 days
  - Stop loss: 5% of entry spread value

**Position Sizing:**
- Capacity constraint: 10% of 20-day average daily volume (ADV)
- Maximum leverage: 2.0x
- Equal dollar amounts for long/short legs

**Outputs:**
- Signal series (-1, 0, 1) for each pair
- Position sizes (dollar amounts) for ETF A and ETF B
- Entry/exit dates and z-scores

---

### Step 3: Strategy Returns & Performance Metrics
**Objective:** Compute strategy returns and evaluate performance

**Return Calculation:**
- Dollar P&L: P&L_t = position_A × return_A + position_B × return_B
- Strategy return: r_t = P&L_t / notional_{t-1}
- Transaction costs: 10 bps transaction + 5 bps spread (20 bps round-trip)

**Performance Metrics Computed:**
- Annualized return and volatility
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown
- Total return over period
- Win rate and trade statistics

**Outputs:**
- Daily strategy returns for each pair
- Performance metrics dictionary
- Individual pair Sharpe ratios: XLE-VDE (0.128), XLK-FTEC (-0.145), XLF-VFH (0.292), XLI-VIS (0.373), XLV-VHT (0.215)

---

## Slide 2: Steps 4-5

### Step 4: Portfolio Weight Optimization
**Objective:** Optimize capital allocation across pairs to maximize risk-adjusted returns

**Optimization Method:**
- Objective: Maximize Sharpe ratio
- Constraints:
  - Weights sum to 1.0 (fully invested)
  - Individual weights: 0 ≤ w_i ≤ 1.0
  - No leverage constraint (can allow negative weights for shorting)
- Algorithm: Sequential Least Squares Programming (SLSQP)

**Optimization Results:**
- **Original Portfolio Weights:**
  - XLI-VIS: 36.0% (highest allocation)
  - XLF-VFH: 31.7%
  - XLV-VHT: 18.4%
  - XLE-VDE: 13.9%
  - XLK-FTEC: 0.0% (excluded)
- **Portfolio Sharpe Ratio:** 0.432

**Hedge Ratio Optimization (Optional):**
- Optimize Kalman filter parameters (observation_noise, state_noise, initial_beta) for each pair
- Grid search + local optimization to maximize individual pair Sharpe ratios
- **Optimized Portfolio Weights:**
  - XLE-VDE: 28.9%
  - XLI-VIS: 21.3%
  - XLF-VFH: 19.6%
  - XLK-FTEC: 17.9%
  - XLV-VHT: 12.3%
- **Optimized Portfolio Sharpe Ratio:** 1.00

**Outputs:**
- Optimal weights dictionary for each pair
- Portfolio return series (weighted combination)
- Portfolio performance metrics

---

### Step 5: Cumulative Returns & Final Performance
**Objective:** Calculate cumulative returns and evaluate overall strategy performance

**Cumulative Return Calculation:**
- Cumulative return: CR_t = ∏(1 + r_i) from i=1 to t
- Equity curve: E_t = E_0 × CR_t
- Drawdown: DD_t = (E_t - max(E_{0:t})) / max(E_{0:t})

**Final Performance Summary:**
- **Original Portfolio (No Costs):**
  - Annualized Return: 0.09%
  - Annualized Volatility: 0.21%
  - Sharpe Ratio: 0.432
  - Total Return: 1.03%
  - Max Drawdown: -0.27%

- **With Transaction Costs (20 bps):**
  - Annualized Return: -1.49%
  - Sharpe Ratio: -2.09
  - Total Return: -15.36%
  - Max Drawdown: -15.42%

**Visualizations Generated:**
- Cumulative returns over time
- Drawdown analysis
- Rolling Sharpe ratio (252-day window)
- Return distribution
- Monthly returns heatmap
- Portfolio weights comparison
- Correlation matrix of pair returns

**Key Insights:**
- Strategy profitable before transaction costs
- Transaction costs significantly impact returns (break-even ~0 bps)
- Portfolio diversification improves risk-adjusted returns
- Optimized hedge ratios improve Sharpe ratio from 0.43 to 1.00

---

## Summary
**Data Period:** 2010-2024 (common period: 2013-2024, 2,815 trading days)
**Pairs:** 5 same-sector ETF pairs
**Methodology:** Kalman filter → Signal generation → Returns computation → Portfolio optimization → Performance evaluation
**Key Innovation:** Time-varying hedge ratios + flow-conditioned signals + portfolio optimization

