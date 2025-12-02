# Q&A Preparation: Pairs Trading Strategy Presentation

## Sample Questions and Answers (20 Questions)

---

### **Methodology Questions**

#### **Q1: Why did you choose Kalman filter instead of static regression for hedge ratio estimation?**

**A:** Static regression assumes a constant hedge ratio over time, but ETF relationships evolve due to market conditions, sector rotation, and structural changes. Kalman filter allows the hedge ratio β_t to adapt over time, tracking the actual relationship at each point. This provides more accurate spread calculations and better captures non-stationary relationships. Additionally, Kalman filter gives us time-varying uncertainty estimates, which helps with risk management.

---

#### **Q2: How did you choose the Kalman filter parameters (observation_noise, state_noise, initial_beta)?**

**A:** We started with standard values from the literature: observation_noise of 0.001 and state_noise of 0.0001, with initial_beta of 1.0. However, we also performed optimization for each pair, using grid search combined with local optimization to maximize Sharpe ratio. The optimized parameters varied by pair, with observation_noise ranging from 0.0026 to 0.0075, showing that different pairs require different parameter settings. This optimization improved portfolio Sharpe from 0.43 to 1.00.

---

#### **Q3: Why use z-scores instead of absolute spread values for signal generation?**

**A:** Z-scores normalize the spread by its recent mean and standard deviation, making signals comparable across different pairs and time periods. A spread of $0.50 might be extreme for one pair but normal for another. Z-scores account for the natural volatility of each pair, allowing us to use consistent thresholds (like ±2.0) across all pairs. This standardization is crucial when combining multiple pairs into a portfolio.

---

#### **Q4: What is the rationale behind the flow differential condition for entry signals?**

**A:** Flow differentials capture market stress and temporary mispricings. When one ETF experiences extreme flows relative to its pair, it often creates temporary price dislocations that mean revert. By requiring flow differentials in the top 25th percentile, we only enter trades when there's evidence of market stress that's likely to reverse. This filters out trades during normal market conditions where mean reversion may be weaker.

---

#### **Q5: How did you determine the position sizing constraints (10% of ADV, 2x leverage)?**

**A:** The 10% of average daily volume constraint ensures we can enter and exit positions without significant market impact. This is a standard practice in quantitative trading to maintain tradeability. The 2x leverage limit provides some flexibility while keeping risk manageable. These constraints are conservative compared to some strategies but are appropriate given the transaction cost sensitivity we observed.

---

### **Results and Performance Questions**

#### **Q6: Why did the strategy perform poorly after including transaction costs?**

**A:** Mean reversion strategies generate small, frequent profits. With 20 basis points per round trip (10 bps transaction + 5 bps per side spread), these small profits are quickly eroded. Our sensitivity analysis showed the strategy becomes unprofitable even at very low cost levels. This highlights a fundamental challenge: the strategy's edge is too small relative to execution costs. For viability, we'd need either extremely low costs (below 5 bps) or significantly larger mean reversion opportunities.

---

#### **Q7: Why did the portfolio weight optimization exclude XLK-FTEC (Technology pair) in the original portfolio?**

**A:** XLK-FTEC had a negative Sharpe ratio of -0.145, meaning it was unprofitable even before transaction costs. The optimization algorithm correctly identified this and allocated zero weight to it, focusing capital on the profitable pairs. This demonstrates the value of portfolio optimization—it automatically filters out underperforming pairs and allocates more to stronger performers like XLI-VIS (Sharpe 0.373) and XLF-VFH (Sharpe 0.292).

---

#### **Q8: How did optimizing Kalman filter parameters improve the Sharpe ratio from 0.43 to 1.00?**

**A:** Optimizing the Kalman parameters for each pair individually improved their individual Sharpe ratios, which then improved the portfolio when reoptimized. For example, XLE-VDE improved from 0.128 to 0.599 Sharpe. Better hedge ratio estimates lead to more accurate spreads, which generate better signals and returns. The optimization also led to more balanced portfolio weights, reducing concentration risk while improving overall performance.

---

#### **Q9: What explains the wide range in individual pair Sharpe ratios (from -0.145 to 0.373)?**

**A:** Different pairs have different mean reversion characteristics. XLI-VIS (Industrials) showed the strongest mean reversion with a Sharpe of 0.373, likely due to stable sector relationships and good liquidity. XLK-FTEC (Technology) had negative Sharpe, possibly due to higher volatility, different market dynamics, or weaker mean reversion in that sector. This variation is expected and highlights the importance of pair selection and portfolio diversification.

---

#### **Q10: Why did the portfolio with transaction costs allocate 100% to XLV-VHT (Healthcare)?**

**A:** After transaction costs, most pairs became unprofitable. XLV-VHT had the lowest cost impact (1.38% vs 1.44-1.75% for others), making it the least bad option. However, even this allocation resulted in negative 1.49% annual return and negative 2.09 Sharpe. This extreme concentration reflects the optimizer's attempt to minimize losses, but it's not a viable strategy—it's essentially choosing the least unprofitable option.

---

### **Data and Implementation Questions**

#### **Q11: What data sources did you use, and what was the time period?**

**A:** We used WRDS (Wharton Research Data Services) to obtain daily price, volume, and flow data for 10 ETFs across 5 sector pairs. The data period spans from 2010-01-01 to 2024-12-31, though the common trading period after accounting for different ETF launch dates is 2013-10-24 to 2024-12-31 (2,815 trading days). We used adjusted closing prices to account for dividends and splits.

---

#### **Q12: How did you handle missing data or different trading calendars between ETFs?**

**A:** We aligned all series on common trading dates using pandas intersection. For any missing values, we forward-filled prices but excluded days where either ETF in a pair had missing data. This ensures we only trade when both ETFs have valid prices. The Kalman filter naturally handles missing observations by skipping updates on those days, though we ensured sufficient overlap for meaningful analysis.

---

#### **Q13: Did you perform any data preprocessing or cleaning?**

**A:** Yes. We computed log prices for spread calculation, calculated returns as log differences, computed 20-day rolling average daily volume for capacity constraints, and calculated flow differentials as the difference in net flows between pairs. We also handled outliers by winsorizing extreme values at the 1st and 99th percentiles for flow data to prevent spurious signals from data errors.

---

### **Robustness and Validation Questions**

#### **Q14: How did you validate that your results aren't due to overfitting or data snooping?**

**A:** We used several robustness checks: (1) Walk-forward validation with purged and embargoed periods to prevent look-ahead bias, (2) Out-of-sample testing on recent data, (3) Sensitivity analysis showing consistent results across parameter ranges, (4) Structural break detection to identify regime changes, and (5) Event studies around large flow events. While we optimized parameters, we did so on training data and validated on separate test periods.

---

#### **Q15: What would happen if you tested this strategy on different time periods or market conditions?**

**A:** Mean reversion strategies tend to perform better in range-bound, low-volatility markets and worse during trending markets or financial crises. Our period (2013-2024) includes both bull markets and the COVID-19 volatility, providing a reasonable test. However, the strategy would likely struggle during extended trends or high-volatility periods. We'd need to test on additional periods and potentially add regime filters to avoid trading during unfavorable conditions.

---

#### **Q16: How sensitive are your results to the choice of parameters (z_entry, z_exit, holding period)?**

**A:** We performed sensitivity analysis across parameter grids. The strategy is moderately sensitive—small changes in z_entry (1.5 vs 2.0) can significantly affect trade frequency and returns. However, the fundamental finding—that transaction costs make the strategy unprofitable—holds across all parameter combinations we tested. This suggests the issue is structural (costs too high relative to edge) rather than parameter-specific.

---

### **Practical Implementation Questions**

#### **Q17: How would you implement this strategy in practice, given the transaction cost issues?**

**A:** Several approaches: (1) Focus on pairs with larger mean reversion opportunities, (2) Negotiate better execution (target <5 bps total costs), (3) Reduce trading frequency by using wider entry thresholds, (4) Use limit orders to capture spread instead of paying it, (5) Focus on less liquid pairs where the edge might be larger, or (6) Combine with other strategies to diversify. The key is either reducing costs or increasing the edge.

---

#### **Q18: What are the main risks of this strategy beyond transaction costs?**

**A:** Key risks include: (1) **Divergence risk**—pairs may not mean revert if the relationship fundamentally changes, (2) **Liquidity risk**—difficulty exiting positions during stress, (3) **Model risk**—Kalman filter may not capture all relationship changes, (4) **Regime risk**—strategy fails during trending markets, (5) **Correlation breakdown**—pairs may decouple permanently, and (6) **Capacity constraints**—strategy may not scale beyond certain size.

---

#### **Q19: How would you scale this strategy to a larger portfolio with more pairs?**

**A:** We'd: (1) Expand to more sector pairs and cross-sector pairs, (2) Use hierarchical optimization to manage correlation, (3) Implement risk budgeting across pairs, (4) Add sector exposure limits, (5) Use factor models to understand common drivers, and (6) Implement dynamic rebalancing based on market conditions. However, scaling also increases operational complexity and may reduce the edge if too many participants trade similar strategies.

---

#### **Q20: What are the main limitations of your approach, and how would you address them?**

**A:** **Limitations:** (1) Transaction costs make strategy unprofitable, (2) Assumes mean reversion will continue, (3) Limited to 5 pairs in one study, (4) No regime filtering, (5) Static parameter choices (though we optimized), and (6) Assumes sufficient liquidity. **Addressing them:** (1) Focus on cost reduction or larger opportunities, (2) Add regime detection and pause trading during trends, (3) Expand to more pairs and sectors, (4) Implement dynamic parameter adjustment, (5) Add real-time liquidity monitoring, and (6) Consider alternative execution strategies like TWAP/VWAP algorithms.

---

## Additional Quick Answers for Common Follow-ups

**Q: Why not use machine learning instead of Kalman filter?**
**A:** ML could work but requires more data and is less interpretable. Kalman filter is well-suited for this problem and provides uncertainty estimates. We could combine both—use ML for regime detection and Kalman for hedge ratios.

**Q: Did you test different rebalancing frequencies?**
**A:** We used daily rebalancing, but weekly or monthly could reduce costs. However, this also reduces responsiveness to mean reversion opportunities. The optimal frequency depends on the half-life of mean reversion.

**Q: How does this compare to other pairs trading papers?**
**A:** Our approach is similar to Gatev et al. (2006) but adds time-varying hedge ratios and flow conditioning. The transaction cost findings align with recent literature showing costs are critical for high-frequency mean reversion.

**Q: What's the minimum capital required?**
**A:** Given our position sizing (10% of ADV) and 2x leverage, minimum would be around $500K-$1M to trade all 5 pairs effectively, though this depends on the specific ETFs' ADV.

**Q: Would this work with cryptocurrencies or other asset classes?**
**A:** The methodology is general, but cryptocurrency pairs might have different dynamics (higher volatility, 24/7 trading, different cost structure). The flow conditioning might not apply directly, but the mean reversion framework could work.

---

## Tips for Q&A Delivery

1. **Listen carefully** to the full question before answering
2. **Pause briefly** to think if needed—better than rushing
3. **Be honest** about limitations—shows intellectual honesty
4. **Refer to slides** when discussing specific numbers
5. **Keep answers concise**—aim for 30-60 seconds per answer
6. **If you don't know**, say so and offer to follow up
7. **Connect answers** back to your main findings when possible
8. **Stay calm**—difficult questions show the audience is engaged

