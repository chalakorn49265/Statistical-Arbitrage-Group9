# Transaction Cost Analysis Script (1 minute)

## Timing Guide
- **Total time:** 1 minute (60 seconds)
- **Pace:** ~150 words per minute
- **Word count:** ~150 words

---

## Script

**[0:00 - 0:20] Impact of Transaction Costs**

"Transaction costs are critical for mean reversion strategies. We model costs as 20 basis points per round trip—10 basis points for transaction fees and 5 basis points per side for bid-ask spreads. When we include these costs, the results are dramatic. Our portfolio's annual return drops from positive 0.09% to negative 1.49%. The Sharpe ratio collapses from 0.43 to negative 2.09, making the strategy unprofitable."

**[0:20 - 0:40] Individual Pair Impact**

"At the individual pair level, transaction costs reduce returns by 1.1 to 1.7 percentage points annually. For example, the Energy pair goes from 0.05% return to negative 1.05%. The Healthcare pair, which had the lowest cost impact at 1.47 percentage points, still becomes unprofitable. This occurs because mean reversion strategies require frequent trading, and each trade incurs costs that erode the small expected profits."

**[0:40 - 1:00] Key Takeaways**

"Our sensitivity analysis shows the strategy becomes unprofitable even at very low cost levels—essentially zero basis points. This highlights a fundamental challenge: mean reversion strategies generate small, frequent profits that are easily eliminated by transaction costs. For this strategy to be viable, we would need either extremely low costs—below 5 basis points—or significantly larger mean reversion opportunities. This finding emphasizes the importance of realistic cost assumptions in backtesting and the need for institutional-quality execution to make such strategies profitable."

---

## Alternative Shorter Version (if needed)

**[0:00 - 0:15] Impact**

"Transaction costs of 20 basis points per round trip—10 bps fees plus 5 bps per side for spreads—dramatically impact our strategy. Portfolio returns drop from 0.09% to negative 1.49%, and Sharpe ratio falls from 0.43 to negative 2.09."

**[0:15 - 0:35] Why It Matters**

"Individual pairs lose 1.1 to 1.7 percentage points annually. Mean reversion strategies trade frequently, and each trade's small profit is eroded by costs. Our sensitivity analysis shows the strategy becomes unprofitable even at near-zero costs."

**[0:35 - 1:00] Implications**

"This demonstrates a fundamental challenge: small, frequent profits are easily eliminated by transaction costs. For viability, we'd need either extremely low costs—below 5 basis points—or larger mean reversion opportunities. This emphasizes the critical importance of realistic cost assumptions and institutional-quality execution."

---

## Key Points to Emphasize

1. **Magnitude:** "20 basis points" → "negative 1.49%"
2. **Contrast:** "0.43 Sharpe" → "negative 2.09 Sharpe"
3. **Frequency:** "Frequent trading" → "each trade incurs costs"
4. **Reality check:** "Unprofitable even at near-zero costs"
5. **Solution:** "Need below 5 bps or larger opportunities"

---

## Delivery Tips

1. **Pause after numbers:** Give audience time to process the dramatic changes
2. **Emphasize contrasts:** "From positive... to negative"
3. **Slow down on:** "Negative 2.09" and "unprofitable"
4. **Visual cue:** Point to slide showing before/after comparison
5. **Tone:** Matter-of-fact, not apologetic—this is an important finding

---

## Word Count
- Full version: ~250 words (1.7 minutes at 150 wpm)
- Shorter version: ~150 words (1.0 minute at 150 wpm)

