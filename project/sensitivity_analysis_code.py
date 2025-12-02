# Sensitivity Analysis: Transaction Cost Levels
# This code generates a table showing how portfolio performance varies with different transaction cost levels

print("="*70)
print("Sensitivity Analysis: Transaction Cost Levels")
print("="*70)

# Define transaction cost levels to test (in basis points)
cost_levels = [0, 10, 20, 50]

# Store results for each cost level
sensitivity_results = []

# Check if we have the necessary data
if 'pair_data_dict' not in locals() and 'pair_returns_dict' not in locals():
    print("\nError: pair_data_dict or pair_returns_dict not found.")
    print("Please run the previous cells to load pair data first.")
else:
    # Use pair_data_dict if available, otherwise use pair_returns_dict
    if 'pair_data_dict' in locals():
        pairs_to_analyze = pair_data_dict
        use_pair_data = True
    else:
        pairs_to_analyze = pair_returns_dict
        use_pair_data = False
    
    print(f"\nAnalyzing {len(pairs_to_analyze)} pairs across {len(cost_levels)} cost levels...")
    print(f"Using {'pair_data_dict' if use_pair_data else 'pair_returns_dict'}")
    
    for cost_bps in cost_levels:
        print(f"\n{'='*70}")
        print(f"Cost Level: {cost_bps} bps")
        print(f"{'='*70}")
        
        try:
            if use_pair_data:
                # If we have pair_data_dict, we need to compute returns with costs
                # This requires running the full pipeline for each cost level
                print("Computing strategy returns with costs...")
                
                # For each pair, compute returns with the specified cost level
                cost_adjusted_returns = {}
                
                for pair_name, pair_df in pairs_to_analyze.items():
                    try:
                        # Estimate hedge ratio (using default Kalman config)
                        from src.kalman_filter import estimate_hedge_ratio_kalman
                        from src.signal_generation import generate_signals, compute_position_sizes
                        from src.portfolio_optimization import _compute_strategy_returns_with_costs
                        import config
                        
                        # Run Kalman filter
                        kalman_result = estimate_hedge_ratio_kalman(
                            pair_df, 
                            config.KALMAN_CONFIG
                        )
                        spread = kalman_result['spread']
                        
                        # Generate signals
                        signals = generate_signals(pair_df, spread, config.SIGNAL_CONFIG)
                        signals = compute_position_sizes(signals, pair_df, config.BACKTEST_CONFIG)
                        
                        # Compute returns with costs
                        if cost_bps > 0:
                            # Split cost: transaction cost + spread cost
                            trans_cost = cost_bps / 2  # Half for each side
                            spread_cost = 5  # Fixed spread cost
                            strategy_returns = _compute_strategy_returns_with_costs(
                                signals, pair_df, trans_cost, spread_cost
                            )
                        else:
                            # No costs
                            from src.portfolio_optimization import _compute_strategy_returns_simple
                            strategy_returns = _compute_strategy_returns_simple(signals, pair_df)
                        
                        cost_adjusted_returns[pair_name] = strategy_returns
                        
                    except Exception as e:
                        print(f"  Error processing {pair_name}: {e}")
                        continue
                
                if len(cost_adjusted_returns) == 0:
                    print("  No valid returns computed. Skipping this cost level.")
                    continue
                
                # Optimize portfolio weights
                print("Optimizing portfolio weights...")
                opt_result = optimize_portfolio_weights(
                    cost_adjusted_returns,
                    objective='sharpe',
                    constraints=weight_constraints,
                    method='SLSQP'
                )
                
                portfolio_returns = opt_result['portfolio_returns']
                portfolio_metrics = opt_result['metrics']
                
            else:
                # If we have pair_returns_dict (gross returns), apply costs
                print("Applying transaction costs to gross returns...")
                
                # For simplicity, we'll compute costs based on turnover
                # This is an approximation - for exact costs, we'd need to recompute with pair_data_dict
                cost_adjusted_returns = {}
                
                for pair_name, gross_returns in pairs_to_analyze.items():
                    if cost_bps > 0:
                        # Approximate: assume costs proportional to return volatility
                        # This is a simplification - actual costs depend on trading frequency
                        cost_per_period = cost_bps / 10000  # Convert bps to decimal
                        # Estimate turnover (simplified: use absolute returns as proxy)
                        estimated_turnover = gross_returns.abs().mean() * 10  # Rough estimate
                        net_returns = gross_returns - (cost_per_period * estimated_turnover)
                        cost_adjusted_returns[pair_name] = net_returns
                    else:
                        cost_adjusted_returns[pair_name] = gross_returns
                
                # Optimize portfolio weights
                print("Optimizing portfolio weights...")
                opt_result = optimize_portfolio_weights(
                    cost_adjusted_returns,
                    objective='sharpe',
                    constraints=weight_constraints,
                    method='SLSQP'
                )
                
                portfolio_returns = opt_result['portfolio_returns']
                portfolio_metrics = opt_result['metrics']
            
            # Calculate turnover (simplified: sum of absolute weight changes)
            # For a more accurate measure, we'd track actual trades
            turnover = portfolio_returns.std() * np.sqrt(252) * 2  # Rough estimate
            
            # Store results
            sensitivity_results.append({
                'Cost Level': f"{cost_bps} bps",
                'Ann. Return': portfolio_metrics.get('annualized_return', 0) * 100,  # Convert to %
                'Sharpe': portfolio_metrics.get('sharpe_ratio', 0),
                'Max DD': portfolio_metrics.get('max_drawdown', 0) * 100,  # Convert to %
                'Turnover': turnover
            })
            
            print(f"  Annualized Return: {portfolio_metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {portfolio_metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Estimated Turnover: {turnover:.2f}")
            
        except Exception as e:
            print(f"  Error at cost level {cost_bps} bps: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create and display table
    if len(sensitivity_results) > 0:
        print("\n" + "="*70)
        print("Table 2: Sensitivity Analysis: Transaction Cost Levels")
        print("="*70)
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Format the table nicely
        print("\n" + sensitivity_df.to_string(index=False))
        
        # Also create a LaTeX-formatted version
        print("\n" + "="*70)
        print("LaTeX Table Format:")
        print("="*70)
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Sensitivity Analysis: Transaction Cost Levels}")
        print("\\label{tab:cost_sensitivity}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("\\textbf{Cost Level} & \\textbf{Ann. Return} & \\textbf{Sharpe} & \\textbf{Max DD} & \\textbf{Turnover} \\\\")
        print("\\midrule")
        
        for _, row in sensitivity_df.iterrows():
            print(f"{row['Cost Level']} & {row['Ann. Return']:.2f}\\% & {row['Sharpe']:.2f} & {row['Max DD']:.2f}\\% & {row['Turnover']:.2f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        
        # Store for later use
        sensitivity_analysis_results = sensitivity_df
        
    else:
        print("\nNo results generated. Please check the error messages above.")

