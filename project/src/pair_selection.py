"""
Pair selection and ranking module.

This module:
- Generates candidate ETF pairs
- Computes correlation and cointegration metrics
- Ranks pairs by quality
- Selects best pairs for trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, coint
import warnings
warnings.filterwarnings('ignore')


# Comprehensive list of sector ETF pairs
# Format: (ETF_A, ETF_B, sector, description)
EXPANDED_ETF_PAIRS = [
    # Energy
    ('XLE', 'VDE', 'Energy', 'SPDR vs Vanguard Energy'),
    ('XLE', 'IYE', 'Energy', 'SPDR vs iShares Energy'),
    ('VDE', 'IYE', 'Energy', 'Vanguard vs iShares Energy'),
    
    # Technology
    ('XLK', 'FTEC', 'Technology', 'SPDR vs Fidelity Technology'),
    ('XLK', 'VGT', 'Technology', 'SPDR vs Vanguard Technology'),
    ('FTEC', 'VGT', 'Technology', 'Fidelity vs Vanguard Technology'),
    ('XLK', 'IGV', 'Technology', 'SPDR vs iShares Software'),
    
    # Financials
    ('XLF', 'VFH', 'Financials', 'SPDR vs Vanguard Financials'),
    ('XLF', 'IYF', 'Financials', 'SPDR vs iShares Financials'),
    ('VFH', 'IYF', 'Financials', 'Vanguard vs iShares Financials'),
    ('XLF', 'KBE', 'Financials', 'SPDR vs SPDR Banks'),
    
    # Industrials
    ('XLI', 'VIS', 'Industrials', 'SPDR vs Vanguard Industrials'),
    ('XLI', 'IYJ', 'Industrials', 'SPDR vs iShares Industrials'),
    ('VIS', 'IYJ', 'Industrials', 'Vanguard vs iShares Industrials'),
    
    # Healthcare
    ('XLV', 'VHT', 'Healthcare', 'SPDR vs Vanguard Healthcare'),
    ('XLV', 'IYH', 'Healthcare', 'SPDR vs iShares Healthcare'),
    ('VHT', 'IYH', 'Healthcare', 'Vanguard vs iShares Healthcare'),
    ('XLV', 'IBB', 'Healthcare', 'SPDR vs iShares Biotech'),
    
    # Consumer Discretionary
    ('XLY', 'VCR', 'Consumer Discretionary', 'SPDR vs Vanguard Consumer Disc'),
    ('XLY', 'IYC', 'Consumer Discretionary', 'SPDR vs iShares Consumer Disc'),
    ('VCR', 'IYC', 'Consumer Discretionary', 'Vanguard vs iShares Consumer Disc'),
    
    # Consumer Staples
    ('XLP', 'VDC', 'Consumer Staples', 'SPDR vs Vanguard Consumer Staples'),
    ('XLP', 'IYK', 'Consumer Staples', 'SPDR vs iShares Consumer Staples'),
    ('VDC', 'IYK', 'Consumer Staples', 'Vanguard vs iShares Consumer Staples'),
    
    # Materials
    ('XLB', 'VAW', 'Materials', 'SPDR vs Vanguard Materials'),
    ('XLB', 'IYM', 'Materials', 'SPDR vs iShares Materials'),
    ('VAW', 'IYM', 'Materials', 'Vanguard vs iShares Materials'),
    
    # Utilities
    ('XLU', 'VPU', 'Utilities', 'SPDR vs Vanguard Utilities'),
    ('XLU', 'IDU', 'Utilities', 'SPDR vs iShares Utilities'),
    ('VPU', 'IDU', 'Utilities', 'Vanguard vs iShares Utilities'),
    
    # Real Estate
    ('XLRE', 'VNQ', 'Real Estate', 'SPDR vs Vanguard Real Estate'),
    ('XLRE', 'IYR', 'Real Estate', 'SPDR vs iShares Real Estate'),
    ('VNQ', 'IYR', 'Real Estate', 'Vanguard vs iShares Real Estate'),
    
    # Communication Services
    ('XLC', 'VOX', 'Communication', 'SPDR vs Vanguard Communication'),
    ('XLC', 'IYZ', 'Communication', 'SPDR vs iShares Communication'),
    ('VOX', 'IYZ', 'Communication', 'Vanguard vs iShares Communication'),
    
    # Cross-sector pairs (related sectors)
    ('XLE', 'XOP', 'Energy', 'SPDR Energy vs SPDR Oil & Gas'),
    ('XLK', 'SOXX', 'Technology', 'SPDR Tech vs iShares Semiconductors'),
    ('XLF', 'KRE', 'Financials', 'SPDR Financials vs SPDR Regional Banks'),
    ('XLI', 'IYT', 'Industrials', 'SPDR Industrials vs iShares Transportation'),
    ('XLV', 'XBI', 'Healthcare', 'SPDR Healthcare vs SPDR Biotech'),
]


def compute_pair_correlation(pair_data: pd.DataFrame, 
                            window: int = 252,
                            method: str = 'pearson') -> Dict:
    """
    Compute correlation metrics for an ETF pair.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset with returns_A and returns_B columns
    window : int
        Rolling window for correlation (default: 252 trading days)
    method : str
        Correlation method: 'pearson', 'spearman'
    
    Returns:
    --------
    dict : Correlation metrics
    """
    if 'returns_A' not in pair_data.columns or 'returns_B' not in pair_data.columns:
        return {
            'mean_correlation': np.nan,
            'min_correlation': np.nan,
            'max_correlation': np.nan,
            'std_correlation': np.nan,
            'stability_score': np.nan
        }
    
    returns_a = pair_data['returns_A'].dropna()
    returns_b = pair_data['returns_B'].dropna()
    
    # Align on common dates
    common_idx = returns_a.index.intersection(returns_b.index)
    if len(common_idx) < window:
        return {
            'mean_correlation': np.nan,
            'min_correlation': np.nan,
            'max_correlation': np.nan,
            'std_correlation': np.nan,
            'stability_score': np.nan
        }
    
    returns_a_aligned = returns_a.loc[common_idx]
    returns_b_aligned = returns_b.loc[common_idx]
    
    # Rolling correlation
    rolling_corr = returns_a_aligned.rolling(window=window).corr(returns_b_aligned)
    rolling_corr = rolling_corr.dropna()
    
    if len(rolling_corr) == 0:
        # Fallback to overall correlation
        overall_corr, _ = pearsonr(returns_a_aligned, returns_b_aligned)
        return {
            'mean_correlation': overall_corr,
            'min_correlation': overall_corr,
            'max_correlation': overall_corr,
            'std_correlation': 0.0,
            'stability_score': 1.0 if not np.isnan(overall_corr) else 0.0
        }
    
    return {
        'mean_correlation': rolling_corr.mean(),
        'min_correlation': rolling_corr.min(),
        'max_correlation': rolling_corr.max(),
        'std_correlation': rolling_corr.std(),
        'stability_score': 1.0 - rolling_corr.std()  # Higher is better (more stable)
    }


def test_cointegration(price_a: pd.Series, price_b: pd.Series) -> Dict:
    """
    Test for cointegration between two price series.
    
    Parameters:
    -----------
    price_a : pd.Series
        Price series for ETF A
    price_b : pd.Series
        Price series for ETF B
    
    Returns:
    --------
    dict : Cointegration test results
    """
    # Align on common dates
    common_idx = price_a.index.intersection(price_b.index)
    if len(common_idx) < 60:  # Need minimum observations
        return {
            'cointegrated': False,
            'pvalue': 1.0,
            'test_statistic': np.nan,
            'critical_value_1pct': np.nan
        }
    
    price_a_aligned = price_a.loc[common_idx]
    price_b_aligned = price_b.loc[common_idx]
    
    try:
        # Engle-Granger cointegration test
        score, pvalue, _ = coint(price_a_aligned, price_b_aligned)
        
        # Critical values (approximate)
        critical_1pct = -3.90
        critical_5pct = -3.34
        critical_10pct = -3.04
        
        is_cointegrated = pvalue < 0.05  # 5% significance
        
        return {
            'cointegrated': is_cointegrated,
            'pvalue': pvalue,
            'test_statistic': score,
            'critical_value_1pct': critical_1pct,
            'critical_value_5pct': critical_5pct,
            'critical_value_10pct': critical_10pct,
            'significance_level': '1%' if score < critical_1pct else 
                                  '5%' if score < critical_5pct else
                                  '10%' if score < critical_10pct else 'Not significant'
        }
    except Exception as e:
        return {
            'cointegrated': False,
            'pvalue': 1.0,
            'test_statistic': np.nan,
            'critical_value_1pct': np.nan,
            'error': str(e)
        }


def compute_spread_stationarity(spread: pd.Series) -> Dict:
    """
    Test spread for stationarity using ADF test.
    
    Parameters:
    -----------
    spread : pd.Series
        Spread series
    
    Returns:
    --------
    dict : Stationarity test results
    """
    spread_clean = spread.dropna()
    
    if len(spread_clean) < 60:
        return {
            'is_stationary': False,
            'adf_pvalue': 1.0,
            'adf_statistic': np.nan
        }
    
    try:
        result = adfuller(spread_clean, maxlag=5, autolag='AIC')
        adf_statistic, pvalue, _, _, critical_values, _ = result
        
        is_stationary = pvalue < 0.05  # 5% significance
        
        return {
            'is_stationary': is_stationary,
            'adf_pvalue': pvalue,
            'adf_statistic': adf_statistic,
            'critical_value_1pct': critical_values['1%'],
            'critical_value_5pct': critical_values['5%'],
            'critical_value_10pct': critical_values['10%']
        }
    except Exception as e:
        return {
            'is_stationary': False,
            'adf_pvalue': 1.0,
            'adf_statistic': np.nan,
            'error': str(e)
        }


def compute_pair_quality_score(pair_data: pd.DataFrame,
                              spread: Optional[pd.Series] = None,
                              correlation_metrics: Optional[Dict] = None,
                              cointegration_metrics: Optional[Dict] = None,
                              stationarity_metrics: Optional[Dict] = None) -> Dict:
    """
    Compute overall quality score for a pair.
    
    Parameters:
    -----------
    pair_data : pd.DataFrame
        Pair dataset
    spread : pd.Series or None
        Spread series (if available)
    correlation_metrics : dict or None
        Correlation metrics from compute_pair_correlation
    cointegration_metrics : dict or None
        Cointegration test results
    stationarity_metrics : dict or None
        Stationarity test results
    
    Returns:
    --------
    dict : Quality score and metrics
    """
    score = 0.0
    max_score = 0.0
    details = {}
    
    # 1. Correlation score (0-30 points)
    if correlation_metrics:
        mean_corr = correlation_metrics.get('mean_correlation', 0)
        stability = correlation_metrics.get('stability_score', 0)
        # High correlation (0.7+) is good, but not too high (0.99+ suggests duplicates)
        if 0.7 <= abs(mean_corr) < 0.99:
            corr_score = min(30, abs(mean_corr) * 30 * stability)
            score += corr_score
            details['correlation_score'] = corr_score
        max_score += 30
    
    # 2. Cointegration score (0-30 points)
    if cointegration_metrics:
        if cointegration_metrics.get('cointegrated', False):
            pvalue = cointegration_metrics.get('pvalue', 1.0)
            # Lower p-value is better
            coint_score = 30 * (1.0 - min(pvalue, 0.05) / 0.05)
            score += coint_score
            details['cointegration_score'] = coint_score
        max_score += 30
    
    # 3. Stationarity score (0-20 points)
    if stationarity_metrics:
        if stationarity_metrics.get('is_stationary', False):
            pvalue = stationarity_metrics.get('adf_pvalue', 1.0)
            stat_score = 20 * (1.0 - min(pvalue, 0.05) / 0.05)
            score += stat_score
            details['stationarity_score'] = stat_score
        max_score += 20
    
    # 4. Data quality score (0-20 points)
    if not pair_data.empty:
        # Check for sufficient data
        n_days = len(pair_data)
        if n_days >= 1000:
            data_score = 20
        elif n_days >= 500:
            data_score = 15
        elif n_days >= 252:
            data_score = 10
        else:
            data_score = 5
        score += data_score
        details['data_quality_score'] = data_score
        max_score += 20
    
    # Normalize to 0-100
    quality_score = (score / max_score * 100) if max_score > 0 else 0.0
    
    return {
        'quality_score': quality_score,
        'raw_score': score,
        'max_score': max_score,
        'details': details
    }


def rank_pairs(pair_datasets: Dict[Tuple[str, str], pd.DataFrame],
               spreads: Optional[Dict[Tuple[str, str], pd.Series]] = None) -> pd.DataFrame:
    """
    Rank all pairs by quality metrics.
    
    Parameters:
    -----------
    pair_datasets : dict
        Dictionary of {(symbol_a, symbol_b): pair_data}
    spreads : dict or None
        Dictionary of {(symbol_a, symbol_b): spread_series} (optional)
    
    Returns:
    --------
    pd.DataFrame : Ranked pairs with quality metrics
    """
    results = []
    
    for (symbol_a, symbol_b), pair_data in pair_datasets.items():
        pair_name = f"{symbol_a}-{symbol_b}"
        
        # Compute correlation
        corr_metrics = compute_pair_correlation(pair_data)
        
        # Test cointegration (if price data available)
        coint_metrics = None
        if 'price_A' in pair_data.columns and 'price_B' in pair_data.columns:
            coint_metrics = test_cointegration(
                pair_data['price_A'],
                pair_data['price_B']
            )
        
        # Test spread stationarity
        stat_metrics = None
        if spreads and (symbol_a, symbol_b) in spreads:
            stat_metrics = compute_spread_stationarity(spreads[(symbol_a, symbol_b)])
        elif 'spread' in pair_data.columns:
            stat_metrics = compute_spread_stationarity(pair_data['spread'])
        
        # Compute quality score
        quality = compute_pair_quality_score(
            pair_data,
            spread=spreads.get((symbol_a, symbol_b)) if spreads else None,
            correlation_metrics=corr_metrics,
            cointegration_metrics=coint_metrics,
            stationarity_metrics=stat_metrics
        )
        
        # Collect results
        result = {
            'pair': pair_name,
            'symbol_a': symbol_a,
            'symbol_b': symbol_b,
            'n_days': len(pair_data),
            'quality_score': quality['quality_score'],
            'mean_correlation': corr_metrics.get('mean_correlation', np.nan),
            'correlation_stability': corr_metrics.get('stability_score', np.nan),
            'cointegrated': coint_metrics.get('cointegrated', False) if coint_metrics else False,
            'cointegration_pvalue': coint_metrics.get('pvalue', 1.0) if coint_metrics else 1.0,
            'spread_stationary': stat_metrics.get('is_stationary', False) if stat_metrics else False,
            'adf_pvalue': stat_metrics.get('adf_pvalue', 1.0) if stat_metrics else 1.0,
        }
        result.update(quality['details'])
        results.append(result)
    
    # Create DataFrame and sort by quality score
    df = pd.DataFrame(results)
    df = df.sort_values('quality_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def select_best_pairs(ranked_pairs: pd.DataFrame,
                      n_pairs: int = 15,
                      min_quality_score: float = 50.0,
                      min_correlation: float = 0.7,
                      require_cointegration: bool = False) -> List[Tuple[str, str]]:
    """
    Select best pairs based on ranking and criteria.
    
    Parameters:
    -----------
    ranked_pairs : pd.DataFrame
        Ranked pairs from rank_pairs()
    n_pairs : int
        Maximum number of pairs to select
    min_quality_score : float
        Minimum quality score (0-100)
    min_correlation : float
        Minimum mean correlation
    require_cointegration : bool
        Whether to require cointegration
    
    Returns:
    --------
    list : List of (symbol_a, symbol_b) tuples
    """
    # Filter by criteria
    filtered = ranked_pairs[
        (ranked_pairs['quality_score'] >= min_quality_score) &
        (ranked_pairs['mean_correlation'].abs() >= min_correlation)
    ]
    
    if require_cointegration:
        filtered = filtered[filtered['cointegrated'] == True]
    
    # Select top N
    selected = filtered.head(n_pairs)
    
    # Return as list of tuples
    pairs = [
        (row['symbol_a'], row['symbol_b'])
        for _, row in selected.iterrows()
    ]
    
    return pairs

