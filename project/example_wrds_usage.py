"""
Example script showing how to use WRDS loader for ETF pairs trading.

This script demonstrates:
1. Connecting to WRDS
2. Loading ETF price data
3. Creating pair datasets
4. Computing flows and other metrics
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

import config
from wrds_loader import WRDSETFLoader, load_etf_pairs_from_wrds

def main():
    """Main example function."""
    
    # Your WRDS username
    wrds_username = 'chalakorn'  # Replace with your username
    
    print("="*60)
    print("WRDS ETF Pairs Trading Data Loader Example")
    print("="*60)
    
    # Initialize loader
    print("\n1. Initializing WRDS loader...")
    loader = WRDSETFLoader(wrds_username=wrds_username, data_dir='data/raw')
    
    # Example 1: Load single ETF
    print("\n2. Loading single ETF (XLE)...")
    xle_data = loader.load_etf_price_data_wrds(
        'XLE',
        start_date='2010-01-01',
        end_date='2024-12-31'
    )
    print(f"   Loaded {len(xle_data)} days of data")
    print(f"   Date range: {xle_data.index.min()} to {xle_data.index.max()}")
    print(f"   Columns: {xle_data.columns.tolist()}")
    
    # Example 2: Load multiple ETFs
    print("\n3. Loading multiple ETFs...")
    symbols = ['XLE', 'VDE', 'XLK', 'FTEC']
    etf_data = loader.load_multiple_etfs_wrds(
        symbols,
        start_date='2010-01-01',
        end_date='2024-12-31'
    )
    print(f"   Successfully loaded {len(etf_data)} ETFs")
    
    # Example 3: Create pair dataset
    print("\n4. Creating pair dataset (XLE - VDE)...")
    pair_data = loader.create_pair_dataset(
        'XLE', 'VDE',
        start_date='2010-01-01',
        end_date='2024-12-31'
    )
    print(f"   Pair dataset shape: {pair_data.shape}")
    print(f"   Columns: {pair_data.columns.tolist()}")
    print(f"\n   Sample data:")
    print(pair_data.head())
    
    # Example 4: Compute holdings metrics (if holdings data available)
    print("\n5. Computing holdings metrics...")
    # Note: Holdings data typically comes from external sources
    # loader.load_holdings_from_file('XLE', 'data/raw/XLE_holdings.csv')
    # metrics = loader.compute_pair_holdings_metrics('XLE', 'VDE')
    # print(f"   Holdings overlap: {metrics['overlap']:.2f}")
    # print(f"   Weighting distance: {metrics['weighting_distance']:.2f}")
    
    # Example 5: Load all pairs from config
    print("\n6. Loading all pairs from config...")
    pair_datasets = {}
    for symbol_a, symbol_b, sector in config.ETF_PAIRS:
        print(f"   Loading {symbol_a} - {symbol_b} ({sector})...")
        try:
            loader.load_etf_price_data_wrds(symbol_a, 
                                            start_date=config.START_DATE,
                                            end_date=config.END_DATE)
            loader.load_etf_price_data_wrds(symbol_b,
                                            start_date=config.START_DATE,
                                            end_date=config.END_DATE)
            
            pair_data = loader.create_pair_dataset(symbol_a, symbol_b,
                                                  start_date=config.START_DATE,
                                                  end_date=config.END_DATE)
            
            if not pair_data.empty:
                pair_data['sector'] = sector
                pair_datasets[(symbol_a, symbol_b)] = pair_data
                print(f"     ✓ {len(pair_data)} days")
        except Exception as e:
            print(f"     ✗ Error: {e}")
    
    print(f"\n   Successfully loaded {len(pair_datasets)} pairs")
    
    # Close connection
    print("\n7. Closing WRDS connection...")
    loader.close()
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    
    return pair_datasets


if __name__ == '__main__':
    pair_datasets = main()
    
    # Save pair datasets if needed
    # import pickle
    # with open('data/processed/pair_datasets.pkl', 'wb') as f:
    #     pickle.dump(pair_datasets, f)

