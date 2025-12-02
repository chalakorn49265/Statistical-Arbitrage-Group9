# Plot Placement Guide for LaTeX Report

This document explains where to place each plot from your Jupyter notebook in the LaTeX report section `report_optimization_comparison.tex`.

## Overview

The LaTeX file `report_optimization_comparison.tex` contains comments marked with `% PLACE PLOT HERE:` indicating where to insert figures. Each location is labeled with a figure caption and label.

## Plot Locations and Sources

### 1. Figure: Portfolio Performance (No Costs in Optimizer)
**Location:** After line with `% PLACE PLOT HERE: Figure showing portfolio returns visualization from Cell 32`

**Source:** Cell 32 - The 2x2 subplot showing:
- **Top Left:** Cumulative Returns (Equity Curve) - comparing optimized vs original
- **Top Right:** Daily Returns Distribution - histograms of both portfolios
- **Bottom Left:** Rolling Sharpe Ratio (252-day window)
- **Bottom Right:** Drawdown Analysis

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/optimization_no_costs.png}
\caption{Portfolio Performance: Optimization Without Transaction Costs in Objective Function. The figure shows (a) cumulative returns comparing optimized and original portfolios, (b) daily returns distribution, (c) rolling Sharpe ratio, and (d) drawdown analysis.}
\label{fig:optimization_no_costs}
\end{figure}
```

**How to Export:**
1. Run Cell 32 in your notebook
2. Right-click on the figure → "Save image as..."
3. Save as `figures/optimization_no_costs.png` (create `figures/` folder in your LaTeX project)

---

### 2. Figure: End-to-End Optimization Results
**Location:** After line with `% PLACE PLOT HERE: Figure showing end-to-end optimization results from Cell 45`

**Source:** Cell 45 - The comparison visualization showing:
- Cumulative Returns Comparison (End-to-End vs Optimized Then Add Costs)
- Performance Metrics Comparison
- Other relevant visualizations

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/optimization_with_costs.png}
\caption{Portfolio Performance: End-to-End Optimization With Transaction Costs in Objective Function. This approach optimizes Kalman parameters and portfolio weights while explicitly accounting for transaction costs, resulting in more realistic performance estimates.}
\label{fig:optimization_with_costs}
\end{figure}
```

**How to Export:**
1. Run Cell 45 in your notebook
2. Right-click on the figure → "Save image as..."
3. Save as `figures/optimization_with_costs.png`

---

### 3. Table: Performance Comparison
**Location:** After line with `% PLACE TABLE HERE: Use the comparison DataFrame from Cell 32 and Cell 44`

**Source:** 
- Cell 32: `comparison` DataFrame (Original vs Optimized)
- Cell 44: `comparison_df` DataFrame (all approaches)

**LaTeX Code:**
```latex
\begin{table}[h]
\centering
\caption{Performance Comparison: Optimization Approaches}
\label{tab:optimization_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Approach} & \textbf{Ann. Return} & \textbf{Sharpe Ratio} & \textbf{Max DD} & \textbf{Total Return} \\
\midrule
No Costs in Optimizer & 0.59\% & 2.34 & -X.XX\% & 3.90\% \\
End-to-End (Costs in Obj) & [VALUE]\% & [VALUE] & [VALUE]\% & [VALUE]\% \\
\bottomrule
\end{tabular}
\end{table}
```

**How to Fill:**
1. Copy values from Cell 32 output (`portfolio_metrics_optimized`)
2. Copy values from Cell 44 output (`portfolio_metrics_end_to_end`)
3. Replace `[VALUE]` placeholders with actual numbers

---

### 4. Figure: Trading Frequency Comparison
**Location:** After line with `% PLACE PLOT HERE: If available, show trading frequency/turnover comparison`

**Source:** If you have trading frequency data, create a plot showing:
- Number of trades per period
- Average holding period
- Turnover ratio

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/trading_frequency.png}
\caption{Trading Frequency Comparison: Impact of Transaction Cost Optimization. The end-to-end optimization approach reduces trading frequency by selecting parameters that minimize unnecessary position changes.}
\label{fig:trading_frequency}
\end{figure}
```

**Note:** This plot may not exist yet. You can create it by analyzing the signals from both approaches.

---

### 5. Figure: Return Distributions
**Location:** After line with `% PLACE PLOT HERE: Return distribution comparison (from Cell 32, subplot 2)`

**Source:** Cell 32, Top Right subplot (Daily Returns Distribution)

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{figures/return_distributions.png}
\caption{Daily Returns Distribution: Comparison of Optimization Approaches. The end-to-end optimization approach shows a tighter distribution with reduced tail risk.}
\label{fig:return_distributions}
\end{figure}
```

**How to Export:**
1. You can extract just the top-right subplot from Cell 32
2. Or create a separate plot focusing on return distributions

---

### 6. Figure: Drawdown Comparison
**Location:** After line with `% PLACE PLOT HERE: Drawdown comparison`

**Source:** Cell 32, Bottom Right subplot (Drawdown Analysis)

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/drawdown_comparison.png}
\caption{Drawdown Analysis: Optimization Approaches Comparison. The end-to-end optimization approach exhibits lower maximum drawdown and faster recovery.}
\label{fig:drawdown_comparison}
\end{figure}
```

---

### 7. Table: Cost Sensitivity Analysis
**Location:** After line with `% PLACE PLOT HERE: Sensitivity analysis plot`

**Source:** If you have sensitivity analysis results (from transaction cost analysis cells)

**LaTeX Code:**
```latex
\begin{table}[h]
\centering
\caption{Sensitivity Analysis: Transaction Cost Levels}
\label{tab:cost_sensitivity}
\begin{tabular}{lccccc}
\toprule
\textbf{Cost Level} & \textbf{Ann. Return} & \textbf{Sharpe} & \textbf{Max DD} & \textbf{Turnover} \\
\midrule
0 bps & [VALUE]\% & [VALUE] & [VALUE]\% & [VALUE] \\
10 bps & [VALUE]\% & [VALUE] & [VALUE]\% & [VALUE] \\
20 bps & [VALUE]\% & [VALUE] & [VALUE]\% & [VALUE] \\
50 bps & [VALUE]\% & [VALUE] & [VALUE]\% & [VALUE] \\
\bottomrule
\end{tabular}
\end{table}
```

---

### 8. Figure: Comprehensive Comparison
**Location:** After line with `% PLACE PLOT HERE: Overall comparison visualization (from Cell 45)`

**Source:** Cell 45 - The main comparison visualization

**LaTeX Code:**
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/comprehensive_comparison.png}
\caption{Comprehensive Performance Comparison: All Optimization Approaches. This figure compares the original portfolio, optimization without costs, and end-to-end optimization with costs.}
\label{fig:comprehensive_comparison}
\end{figure}
```

---

## Quick Reference: Cell Numbers

- **Cell 32:** Optimization without transaction costs + visualization (2x2 subplot)
- **Cell 42:** End-to-end optimization with transaction costs (Step 1)
- **Cell 43:** Portfolio weight optimization with cost-adjusted returns (Step 2)
- **Cell 44:** Comparison of all approaches (table)
- **Cell 45:** Comprehensive visualization comparison

## Steps to Complete the LaTeX Report

1. **Export all figures:**
   - Run each cell in your notebook
   - Save figures as PNG files in a `figures/` folder
   - Use descriptive filenames

2. **Fill in table values:**
   - Copy metrics from notebook outputs
   - Replace `[VALUE]` placeholders in tables

3. **Insert figures in LaTeX:**
   - Find each `% PLACE PLOT HERE:` comment
   - Replace with the appropriate `\begin{figure}...\end{figure}` block
   - Adjust `width` parameter as needed

4. **Compile and check:**
   - Compile the LaTeX document
   - Ensure all figures are visible
   - Verify table values are correct

## Tips

- Use `[h]` placement for figures (here) or `[htbp]` for more flexibility
- Adjust `width` parameter (0.7, 0.8, 0.95) based on figure size
- Ensure figure files are in the correct path relative to your main `.tex` file
- Use `\centering` for centered figures
- Always include descriptive captions and labels

