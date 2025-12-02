# Why Kalman Filter Instead of Static Regression for Hedge Ratio Estimation

## Key Advantages of Kalman Filter

### 1. **Time-Varying Hedge Ratios**
**Problem with Regression:** OLS regression estimates a single, static hedge ratio (β) that assumes the relationship between two ETFs is constant over time.

**Kalman Filter Solution:** The hedge ratio β_t evolves over time, allowing the model to adapt to changing market conditions, structural breaks, or shifts in the relationship between the two ETFs.

**Example:** If ETF A and ETF B's relationship changes due to sector rotation, regulatory changes, or different market regimes, the Kalman filter will track this evolution, while regression would give you an average that may not be accurate at any point in time.

---

### 2. **Adaptive to Market Regimes**
**Problem with Regression:** A single regression estimate is an average across all time periods, which may not be representative during volatile periods or regime changes.

**Kalman Filter Solution:** The state-space model assumes β_t = β_{t-1} + η_t (random walk), meaning the hedge ratio can drift gradually. This captures:
- Gradual changes in the relationship
- Market regime shifts
- Structural breaks
- Evolving correlation patterns

---

### 3. **Uncertainty Quantification**
**Problem with Regression:** While regression provides standard errors, they assume the parameter is constant.

**Kalman Filter Solution:** Provides time-varying variance estimates for β_t, giving you:
- Confidence intervals that evolve over time
- Higher uncertainty during volatile periods
- Lower uncertainty during stable periods
- Better risk management through uncertainty-aware position sizing

---

### 4. **Uses Full Information Set (Smoother)**
**Problem with Regression:** Uses all data points equally, but doesn't account for the temporal structure.

**Kalman Filter Solution:** The Kalman smoother uses both past and future information (in-sample) to estimate β_t at each point, providing:
- More efficient estimates (lower variance)
- Better tracking of the true underlying relationship
- Smoother estimates that reduce noise

---

### 5. **Handles Non-Stationarity**
**Problem with Regression:** Assumes the relationship is stationary. If the true relationship drifts, regression gives a biased average.

**Kalman Filter Solution:** Explicitly models non-stationarity through the random walk state equation, making it robust to:
- Trending relationships
- Structural breaks
- Evolving market dynamics

---

## Practical Example

**Static Regression:**
- Estimates: β = 0.95 (constant for entire period)
- If true β changes from 0.90 to 1.00 over time, regression gives you 0.95, which is wrong at both the beginning and end

**Kalman Filter:**
- Estimates: β_t evolves from 0.90 → 0.92 → 0.94 → 0.96 → 0.98 → 1.00
- Tracks the actual relationship at each point in time
- Provides more accurate spread calculation: s_t = log_price_A - β_t × log_price_B

---

## When Regression Might Be Sufficient

Static regression is adequate if:
- The relationship is truly constant over time
- You're only interested in a long-term average
- The time period is short and stable
- You want a simpler, more interpretable model

However, for pairs trading where:
- Relationships can evolve
- You need accurate hedge ratios for spread calculation
- Market conditions change
- You trade frequently and need current estimates

**Kalman filter is the superior choice.**

---

## Mathematical Intuition

**Regression Model:**
```
y_t = α + β × x_t + ε_t
```
- β is constant for all t
- Assumes relationship doesn't change

**Kalman Filter Model:**
```
Observation: y_t = α + β_t × x_t + ε_t
State:       β_t = β_{t-1} + η_t
```
- β_t evolves over time
- Adapts to changing relationships
- More realistic for financial markets

---

## Summary

| Aspect | Static Regression | Kalman Filter |
|--------|------------------|---------------|
| **Hedge Ratio** | Constant (β) | Time-varying (β_t) |
| **Adaptability** | None | Adapts to changes |
| **Uncertainty** | Constant SE | Time-varying variance |
| **Information Use** | All data equally | Temporal structure |
| **Non-stationarity** | Assumes stationary | Handles non-stationary |
| **Complexity** | Simple | More complex |
| **Computational Cost** | Low | Moderate |

**Bottom Line:** For pairs trading where relationships evolve and you need accurate, current hedge ratios, Kalman filter provides significant advantages over static regression.

