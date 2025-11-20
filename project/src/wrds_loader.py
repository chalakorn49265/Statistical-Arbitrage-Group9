"""
WRDS data loader for ETF pairs trading project.

Loads ETF price, volume, shares outstanding, and other data from WRDS.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
import wrds
from typing import Optional, List, Dict
import warnings

from .data_loader import ETFDataLoader


class WRDSETFLoader(ETFDataLoader):
    """
    WRDS-based ETF data loader.
    Extends ETFDataLoader with WRDS connection and data pulling capabilities.
    """
    
    def __init__(self, wrds_username: str, data_dir: Optional[str] = None):
        """
        Initialize WRDS ETF loader.
        
        Parameters:
        -----------
        wrds_username : str
            WRDS username
        data_dir : str or None
            Directory to cache data (optional)
        """
        super().__init__(data_dir)
        self.wrds_username = wrds_username
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish WRDS connection."""
        try:
            self.conn = wrds.Connection(wrds_username=self.wrds_username)
            self.conn.create_pgpass_file()
            print("✓ WRDS connection established")
        except Exception as e:
            warnings.warn(f"Failed to connect to WRDS: {e}")
            self.conn = None
    
    def close(self):
        """Close WRDS connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def load_etf_price_data_wrds(self, 
                                symbol: str,
                                start_date: str = '2010-01-01',
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load ETF price data from WRDS CRSP.
        
        Parameters:
        -----------
        symbol : str
            ETF ticker symbol (e.g., 'XLE')
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str or None
            End date (YYYY-MM-DD), if None uses current date
        
        Returns:
        --------
        pd.DataFrame : Price data with Date index
        """
        if not self.conn:
            raise ConnectionError("WRDS connection not established")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert dates for SQL
        start_date_sql = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m/%d/%Y')
        end_date_sql = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m/%d/%Y')
        
        # Query CRSP for ETF data
        # Try both dsf (daily stock file) and dse (daily ETF file) using UNION
        # This handles ETFs that might be in either table
        query = f"""
        SELECT 
            a.date,
            a.ticker,
            a.prc as close,
            a.ret,
            a.retx,
            a.shrout as shares_outstanding,
            a.vol as volume,
            a.cfacshr,
            a.cfacpr,
            b.comnam as company_name
        FROM crsp.dsf as a
        LEFT JOIN crsp.stocknames as b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
        WHERE a.ticker = '{symbol}'
        AND a.date BETWEEN '{start_date_sql}' AND '{end_date_sql}'
        
        UNION
        
        SELECT 
            a.date,
            a.ticker,
            a.prc as close,
            a.ret,
            a.retx,
            a.shrout as shares_outstanding,
            a.vol as volume,
            a.cfacshr,
            a.cfacpr,
            b.comnam as company_name
        FROM crsp.dse as a
        LEFT JOIN crsp.stocknames as b
        ON a.permno = b.permno
        AND b.namedt <= a.date
        AND a.date <= b.nameendt
        WHERE a.ticker = '{symbol}'
        AND a.date BETWEEN '{start_date_sql}' AND '{end_date_sql}'
        
        ORDER BY date
        """
        
        try:
            df = self.conn.raw_sql(query)
        except Exception as e:
            # Fallback: try dsf only
            try:
                query_simple = f"""
                SELECT 
                    a.date,
                    a.ticker,
                    a.prc as close,
                    a.ret,
                    a.retx,
                    a.shrout as shares_outstanding,
                    a.vol as volume,
                    a.cfacshr,
                    a.cfacpr,
                    b.comnam as company_name
                FROM crsp.dsf as a
                LEFT JOIN crsp.stocknames as b
                ON a.permno = b.permno
                AND b.namedt <= a.date
                AND a.date <= b.nameendt
                WHERE a.ticker = '{symbol}'
                AND a.date BETWEEN '{start_date_sql}' AND '{end_date_sql}'
                ORDER BY a.date
                """
                df = self.conn.raw_sql(query_simple)
            except Exception as e2:
                warnings.warn(f"Failed to load {symbol} from WRDS: {e2}")
                return pd.DataFrame()
        
        if df.empty:
            warnings.warn(f"No data found for {symbol} in date range")
            return pd.DataFrame()
        
        # Process dates
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Calculate adjusted close
        # prc is already adjusted by cfacpr, but we need to account for splits
        df['price'] = df['close'].abs()  # CRSP prices can be negative (special codes)
        df['price'] = df['price'] * df['cfacpr'] / df['cfacpr'].iloc[-1]  # Adjust to latest
        
        # Handle negative prices (special codes in CRSP)
        df.loc[df['close'] < 0, 'price'] = np.nan
        
        # Shares outstanding (in thousands, convert to actual shares)
        df['shares_outstanding'] = df['shares_outstanding'] * 1000
        
        # Volume
        df['volume'] = df['volume'].fillna(0)
        
        # Compute derived metrics
        df['log_price'] = np.log(df['price'])
        df['returns'] = df['ret'].fillna(0)
        df['log_returns'] = df['log_price'].diff()
        
        # Compute flow proxy
        from .utils import compute_flow_proxy, compute_flow_pct_aum, compute_adv
        df['flow'] = compute_flow_proxy(df['shares_outstanding'], df['price'])
        df['flow_pct'] = compute_flow_pct_aum(df['shares_outstanding'], df['price'])
        df['adv'] = compute_adv(df['volume'], df['price'], window=20)
        
        df['symbol'] = symbol
        
        # Store in parent class
        self.price_data[symbol] = df[['price', 'close', 'volume', 'shares_outstanding', 
                                      'log_price', 'returns', 'log_returns', 
                                      'flow', 'flow_pct', 'adv', 'symbol']]
        
        return self.price_data[symbol]
    
    def load_multiple_etfs_wrds(self,
                               symbols: List[str],
                               start_date: str = '2010-01-01',
                               end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple ETFs from WRDS.
        
        Parameters:
        -----------
        symbols : list
            List of ETF ticker symbols
        start_date : str
            Start date
        end_date : str or None
            End date
        
        Returns:
        --------
        dict : Dictionary of {symbol: dataframe}
        """
        results = {}
        for symbol in symbols:
            print(f"Loading {symbol}...")
            try:
                df = self.load_etf_price_data_wrds(symbol, start_date, end_date)
                if not df.empty:
                    results[symbol] = df
                    print(f"  ✓ Loaded {len(df)} days of data")
                else:
                    print(f"  ✗ No data for {symbol}")
            except Exception as e:
                print(f"  ✗ Error loading {symbol}: {e}")
        
        return results
    
    def get_etf_metadata_wrds(self, symbol: str) -> Dict:
        """
        Get ETF metadata from WRDS.
        
        Parameters:
        -----------
        symbol : str
            ETF ticker
        
        Returns:
        --------
        dict : Metadata dictionary
        """
        if not self.conn:
            raise ConnectionError("WRDS connection not established")
        
        query = f"""
        SELECT DISTINCT
            a.ticker,
            a.comnam as company_name,
            a.namedt as name_date,
            a.nameendt as name_end_date,
            a.shrcd as share_code,
            a.exchcd as exchange_code
        FROM crsp.stocknames as a
        WHERE a.ticker = '{symbol}'
        ORDER BY a.namedt DESC
        LIMIT 1
        """
        
        try:
            df = self.conn.raw_sql(query)
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                return {}
        except Exception as e:
            warnings.warn(f"Failed to get metadata for {symbol}: {e}")
            return {}
    
    def load_holdings_from_file(self,
                                symbol: str,
                                file_path: str) -> pd.DataFrame:
        """
        Load holdings data from file.
        
        Note: WRDS doesn't have comprehensive ETF holdings data.
        Holdings typically come from:
        - ETF issuer websites
        - Refinitiv/Lipper
        - SEC filings (N-PORT, N-CSR)
        
        This method loads from a CSV file with expected format:
        Date, Ticker, Weight
        
        Parameters:
        -----------
        symbol : str
            ETF symbol
        file_path : str
            Path to holdings CSV file
        
        Returns:
        --------
        pd.DataFrame : Holdings data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Standardize columns
            date_cols = ['Date', 'date', 'DATE', 'asofdate', 'AsOfDate']
            ticker_cols = ['Ticker', 'ticker', 'TICKER', 'holding', 'Holding']
            weight_cols = ['Weight', 'weight', 'WEIGHT', 'pct', 'Pct']
            
            date_col = next((c for c in date_cols if c in df.columns), None)
            ticker_col = next((c for c in ticker_cols if c in df.columns), None)
            weight_col = next((c for c in weight_cols if c in df.columns), None)
            
            if not all([date_col, ticker_col, weight_col]):
                raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
            
            # Process dates
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            
            # Group by date and create holdings dictionary
            holdings_dict = {}
            for date in df.index.unique():
                date_data = df.loc[date]
                if isinstance(date_data, pd.Series):
                    holdings_dict[date] = {date_data[ticker_col]: date_data[weight_col]}
                else:
                    holdings_dict[date] = dict(zip(date_data[ticker_col], date_data[weight_col]))
            
            result = pd.DataFrame(index=df.index.unique())
            result['holdings'] = result.index.map(holdings_dict.get)
            
            self.holdings_data[symbol] = result
            return result
            
        except Exception as e:
            warnings.warn(f"Failed to load holdings for {symbol}: {e}")
            return pd.DataFrame()


def load_etf_pairs_from_wrds(pairs: List[tuple],
                             wrds_username: str,
                             start_date: str = '2010-01-01',
                             end_date: Optional[str] = None,
                             data_dir: Optional[str] = None) -> Dict:
    """
    Convenience function to load multiple ETF pairs from WRDS.
    
    Parameters:
    -----------
    pairs : list
        List of (symbol_a, symbol_b, sector) tuples
    wrds_username : str
        WRDS username
    start_date : str
        Start date
    end_date : str or None
        End date
    data_dir : str or None
        Data directory for caching
    
    Returns:
    --------
    dict : Dictionary with pair datasets
    """
    loader = WRDSETFLoader(wrds_username, data_dir)
    
    # Get all unique symbols
    all_symbols = set()
    for pair in pairs:
        all_symbols.add(pair[0])
        all_symbols.add(pair[1])
    
    # Load all ETFs
    print(f"Loading {len(all_symbols)} ETFs from WRDS...")
    loader.load_multiple_etfs_wrds(list(all_symbols), start_date, end_date)
    
    # Create pair datasets
    pair_datasets = {}
    for symbol_a, symbol_b, sector in pairs:
        print(f"\nCreating pair dataset: {symbol_a} - {symbol_b} ({sector})")
        try:
            pair_data = loader.create_pair_dataset(
                symbol_a, symbol_b, start_date, end_date
            )
            if not pair_data.empty:
                pair_data['sector'] = sector
                pair_datasets[(symbol_a, symbol_b)] = pair_data
                print(f"  ✓ Created dataset with {len(pair_data)} days")
            else:
                print(f"  ✗ No overlapping data")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    loader.close()
    return pair_datasets


# Example usage function
def example_wrds_loading():
    """
    Example of how to use WRDS loader.
    """
    import config
    
    # Initialize loader
    loader = WRDSETFLoader(wrds_username='your_username')
    
    # Load single ETF
    xle_data = loader.load_etf_price_data_wrds('XLE', start_date='2010-01-01')
    
    # Load multiple ETFs
    symbols = ['XLE', 'VDE', 'XLK', 'FTEC']
    etf_data = loader.load_multiple_etfs_wrds(symbols, start_date='2010-01-01')
    
    # Create pair dataset
    pair_data = loader.create_pair_dataset('XLE', 'VDE', 
                                          start_date='2010-01-01',
                                          end_date='2024-12-31')
    
    # Load holdings from file (if available)
    # loader.load_holdings_from_file('XLE', 'data/raw/XLE_holdings.csv')
    
    # Close connection
    loader.close()
    
    return pair_data

