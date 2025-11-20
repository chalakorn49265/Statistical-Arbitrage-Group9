"""
Data loading and preprocessing module.

This module handles:
- Loading ETF price and volume data
- Computing log prices, returns, flows
- Aligning data across pairs
- Computing holdings overlap and weighting distance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

from utils import (
    compute_flow_proxy, compute_flow_pct_aum, compute_adv,
    compute_holdings_overlap, compute_weighting_distance
)


class ETFDataLoader:
    """
    Load and preprocess ETF data for pairs trading.
    
    This class is designed to work with dataframes that have the expected structure.
    In production, you would connect this to WRDS/Refinitiv APIs.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        data_dir : str or None
            Directory containing data files (if loading from files)
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.price_data = {}
        self.holdings_data = {}
    
    def load_price_data(self, symbol: str, 
                       file_path: Optional[str] = None,
                       df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load price data for a single ETF.
        
        Parameters:
        -----------
        symbol : str
            ETF symbol (e.g., 'XLE')
        file_path : str or None
            Path to CSV file (if loading from file)
        df : pd.DataFrame or None
            Pre-loaded dataframe (if data already in memory)
        
        Expected columns:
        - Date (or date-like column)
        - Close or Adj Close
        - Volume
        - Shares Outstanding (optional, for flow calculation)
        
        Returns:
        --------
        pd.DataFrame : Processed price data with Date index
        """
        if df is not None:
            data = df.copy()
        elif file_path:
            data = pd.read_csv(file_path)
        elif self.data_dir:
            # Try to find file in data directory
            possible_files = list(self.data_dir.glob(f"{symbol}*.csv"))
            if possible_files:
                data = pd.read_csv(possible_files[0])
            else:
                raise FileNotFoundError(f"No data file found for {symbol}")
        else:
            raise ValueError("Must provide either file_path, df, or data_dir")
        
        # Standardize date column
        date_cols = ['Date', 'date', 'DATE', 'timestamp', 'Timestamp']
        date_col = None
        for col in date_cols:
            if col in data.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found in data for {symbol}")
        
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col).sort_index()
        
        # Standardize price column
        price_cols = ['Adj Close', 'adj_close', 'Close', 'close', 'AdjClose']
        price_col = None
        for col in price_cols:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found in data for {symbol}")
        
        # Standardize volume column
        volume_cols = ['Volume', 'volume', 'VOL', 'Vol']
        volume_col = None
        for col in volume_cols:
            if col in data.columns:
                volume_col = col
                break
        
        # Build standardized dataframe
        result = pd.DataFrame(index=data.index)
        result['price'] = data[price_col]
        result['close'] = data.get('Close', data[price_col])  # Use raw close if available
        
        if volume_col:
            result['volume'] = data[volume_col]
        else:
            result['volume'] = np.nan
        
        # Shares outstanding (optional)
        so_cols = ['Shares Outstanding', 'shares_outstanding', 'SO', 'SharesOutstanding']
        so_col = None
        for col in so_cols:
            if col in data.columns:
                so_col = col
                break
        
        if so_col:
            result['shares_outstanding'] = data[so_col]
        else:
            result['shares_outstanding'] = np.nan
        
        # Compute derived metrics
        result['log_price'] = np.log(result['price'])
        result['returns'] = result['price'].pct_change()
        result['log_returns'] = result['log_price'].diff()
        
        # Compute flow proxy if shares outstanding available
        if result['shares_outstanding'].notna().any():
            result['flow'] = compute_flow_proxy(
                result['shares_outstanding'], 
                result['price']
            )
            result['flow_pct'] = compute_flow_pct_aum(
                result['shares_outstanding'],
                result['price']
            )
        else:
            result['flow'] = np.nan
            result['flow_pct'] = np.nan
        
        # Compute ADV
        if result['volume'].notna().any():
            result['adv'] = compute_adv(result['volume'], result['price'], window=20)
        else:
            result['adv'] = np.nan
        
        result['symbol'] = symbol
        
        self.price_data[symbol] = result
        return result
    
    def load_holdings_data(self, symbol: str,
                          file_path: Optional[str] = None,
                          df: Optional[pd.DataFrame] = None,
                          date_col: str = 'Date',
                          ticker_col: str = 'Ticker',
                          weight_col: str = 'Weight') -> pd.DataFrame:
        """
        Load holdings data for an ETF.
        
        Parameters:
        -----------
        symbol : str
            ETF symbol
        file_path : str or None
            Path to CSV file
        df : pd.DataFrame or None
            Pre-loaded dataframe
        date_col : str
            Name of date column
        ticker_col : str
            Name of ticker/holding identifier column
        weight_col : str
            Name of weight column
        
        Returns:
        --------
        pd.DataFrame : Holdings data with Date index and holdings as nested dict
        """
        if df is not None:
            data = df.copy()
        elif file_path:
            data = pd.read_csv(file_path)
        elif self.data_dir:
            possible_files = list(self.data_dir.glob(f"{symbol}_holdings*.csv"))
            if possible_files:
                data = pd.read_csv(possible_files[0])
            else:
                warnings.warn(f"No holdings file found for {symbol}")
                return pd.DataFrame()
        else:
            warnings.warn(f"No holdings data source provided for {symbol}")
            return pd.DataFrame()
        
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col).sort_index()
        
        # Group by date and create weight dictionary
        holdings_dict = {}
        for date in data.index.unique():
            date_data = data.loc[date]
            if isinstance(date_data, pd.Series):
                # Single row
                holdings_dict[date] = {date_data[ticker_col]: date_data[weight_col]}
            else:
                # Multiple rows
                holdings_dict[date] = dict(zip(date_data[ticker_col], date_data[weight_col]))
        
        result = pd.DataFrame(index=data.index.unique())
        result['holdings'] = result.index.map(holdings_dict.get)
        
        self.holdings_data[symbol] = result
        return result
    
    def create_pair_dataset(self, symbol_a: str, symbol_b: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create aligned dataset for an ETF pair.
        
        Parameters:
        -----------
        symbol_a : str
            First ETF symbol
        symbol_b : str
            Second ETF symbol
        start_date : str or None
            Start date (YYYY-MM-DD)
        end_date : str or None
            End date (YYYY-MM-DD)
        
        Returns:
        --------
        pd.DataFrame : Aligned pair dataset with columns:
            - log_price_A, log_price_B
            - price_A, price_B
            - returns_A, returns_B
            - flow_A, flow_B
            - flow_pct_A, flow_pct_B
            - volume_A, volume_B
            - adv_A, adv_B
            - flow_diff (flow_A - flow_B)
            - flow_diff_pct (flow_pct_A - flow_pct_B)
        """
        if symbol_a not in self.price_data:
            raise ValueError(f"Price data not loaded for {symbol_a}")
        if symbol_b not in self.price_data:
            raise ValueError(f"Price data not loaded for {symbol_b}")
        
        data_a = self.price_data[symbol_a].copy()
        data_b = self.price_data[symbol_b].copy()
        
        # Align on common index
        common_dates = data_a.index.intersection(data_b.index)
        data_a = data_a.loc[common_dates]
        data_b = data_b.loc[common_dates]
        
        # Filter by date range if specified
        if start_date:
            data_a = data_a[data_a.index >= start_date]
            data_b = data_b[data_b.index >= start_date]
        if end_date:
            data_a = data_a[data_a.index <= end_date]
            data_b = data_b[data_b.index <= end_date]
        
        # Build pair dataset
        pair_data = pd.DataFrame(index=data_a.index)
        
        # Log prices
        pair_data['log_price_A'] = data_a['log_price']
        pair_data['log_price_B'] = data_b['log_price']
        pair_data['price_A'] = data_a['price']
        pair_data['price_B'] = data_b['price']
        
        # Returns
        pair_data['returns_A'] = data_a['returns']
        pair_data['returns_B'] = data_b['returns']
        
        # Flows
        pair_data['flow_A'] = data_a['flow']
        pair_data['flow_B'] = data_b['flow']
        pair_data['flow_pct_A'] = data_a['flow_pct']
        pair_data['flow_pct_B'] = data_b['flow_pct']
        pair_data['flow_diff'] = pair_data['flow_A'] - pair_data['flow_B']
        pair_data['flow_diff_pct'] = pair_data['flow_pct_A'] - pair_data['flow_pct_B']
        
        # Volume and ADV
        pair_data['volume_A'] = data_a['volume']
        pair_data['volume_B'] = data_b['volume']
        pair_data['adv_A'] = data_a['adv']
        pair_data['adv_B'] = data_b['adv']
        
        # Metadata
        pair_data['symbol_A'] = symbol_a
        pair_data['symbol_B'] = symbol_b
        
        return pair_data
    
    def compute_pair_holdings_metrics(self, symbol_a: str, symbol_b: str,
                                     date: Optional[str] = None) -> Dict[str, float]:
        """
        Compute holdings overlap and weighting distance for a pair.
        
        Parameters:
        -----------
        symbol_a : str
            First ETF symbol
        symbol_b : str
            Second ETF symbol
        date : str or None
            Specific date to compute metrics (if None, uses most recent)
        
        Returns:
        --------
        dict : Dictionary with 'overlap' and 'weighting_distance' keys
        """
        if symbol_a not in self.holdings_data or self.holdings_data[symbol_a].empty:
            return {'overlap': 0.0, 'weighting_distance': float('inf')}
        if symbol_b not in self.holdings_data or self.holdings_data[symbol_b].empty:
            return {'overlap': 0.0, 'weighting_distance': float('inf')}
        
        holdings_a = self.holdings_data[symbol_a]
        holdings_b = self.holdings_data[symbol_b]
        
        if date:
            target_date = pd.to_datetime(date)
        else:
            # Use most recent common date
            common_dates = holdings_a.index.intersection(holdings_b.index)
            if common_dates.empty:
                return {'overlap': 0.0, 'weighting_distance': float('inf')}
            target_date = common_dates.max()
        
        # Get holdings for target date
        if target_date not in holdings_a.index or target_date not in holdings_b.index:
            return {'overlap': 0.0, 'weighting_distance': float('inf')}
        
        weights_a = holdings_a.loc[target_date, 'holdings']
        weights_b = holdings_b.loc[target_date, 'holdings']
        
        if not isinstance(weights_a, dict) or not isinstance(weights_b, dict):
            return {'overlap': 0.0, 'weighting_distance': float('inf')}
        
        overlap = compute_holdings_overlap(weights_a, weights_b)
        distance = compute_weighting_distance(weights_a, weights_b)
        
        return {
            'overlap': overlap,
            'weighting_distance': distance,
            'date': target_date
        }

