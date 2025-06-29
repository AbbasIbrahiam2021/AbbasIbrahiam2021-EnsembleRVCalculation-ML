"""
Market data fetcher for SPX and VIX with specific date range handling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import time
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_fetcher.log'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, api_key, data_dir="market data"):
        self.api_key = api_key
        self.data_dir = data_dir
        self.base_url = "https://www.alphavantage.co/query"
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")

    def fetch_daily_data(self, symbol):
        """Fetch daily data from Alpha Vantage"""
        # Handle SPX specifically
        if symbol.upper() in ["SPX", "^SPX", "^GSPC"]:
            symbol = "^GSPC"  # Use ^GSPC for S&P 500 index
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error(f"Error fetching {symbol}: {data.get('Note', 'Unknown error')}")
                return None
            
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = [col.split(". ")[1] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_vix_data(self):
        """Fetch VIX data specifically"""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "^VIX",
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error(f"Error fetching VIX: {data.get('Note', 'Unknown error')}")
                return None
            
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.columns = [col.split(". ")[1] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # For VIX we mainly care about the close price
            df = df[["close"]]
            df.columns = ["VIX"]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {str(e)}")
            return None

    def get_data(self, start_date, end_date):
        """Get SPX and VIX data for specific date range"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        results = {}
        
        # Fetch SPX data
        logger.info("Fetching SPX data...")
        spx_data = self.fetch_daily_data("^GSPC")
        if spx_data is not None:
            spx_data = spx_data[spx_data.index.between(start_date, end_date)]
            spx_data['Returns'] = spx_data['Close'].pct_change()
            spx_data['RealizedVol'] = spx_data['Returns'].rolling(window=21).std() * np.sqrt(252) * 100
            results["SPX"] = spx_data
            
            # Save SPX data
            spx_file = os.path.join(self.data_dir, "SPX_data.csv")
            spx_data.to_csv(spx_file)
            logger.info(f"Saved SPX data to {spx_file}")
        
        # Wait to respect rate limits
        time.sleep(12)
        
        # Fetch VIX data
        logger.info("Fetching VIX data...")
        vix_data = self.fetch_vix_data()
        if vix_data is not None:
            vix_data = vix_data[vix_data.index.between(start_date, end_date)]
            results["VIX"] = vix_data
            
            # Save VIX data
            vix_file = os.path.join(self.data_dir, "VIX_data.csv")
            vix_data.to_csv(vix_file)
            logger.info(f"Saved VIX data to {vix_file}")
            
            # Create combined dataset
            combined_data = pd.merge(
                spx_data[['Close', 'RealizedVol']], 
                vix_data,
                left_index=True,
                right_index=True,
                how='outer'
            )
            combined_data.columns = ['SPX_Close', 'SPX_RealizedVol', 'VIX_Close']
            
            # Save combined data
            combined_file = os.path.join(self.data_dir, "SPX_VIX_Combined.csv")
            combined_data.to_csv(combined_file)
            logger.info(f"Saved combined data to {combined_file}")
        
        return results

if __name__ == "__main__":
    API_KEY = "09ATCY1HHH8M0E5O"
    fetcher = DataFetcher(api_key=API_KEY)
    
    # Get data from 2021 to June 24, 2025
    data = fetcher.get_data(
        start_date="2021-01-01",
        end_date="2025-06-24"
    )
    
    if data:
        print("\nData has been saved to the 'market data' folder:")
        print("1. SPX_data.csv - Full SPX data with realized volatility")
        print("2. VIX_data.csv - VIX data")
        print("3. SPX_VIX_Combined.csv - Combined SPX close, realized vol, and VIX")
        
        print("\nSample of combined data:")
        combined_file = os.path.join(fetcher.data_dir, "SPX_VIX_Combined.csv")
        if os.path.exists(combined_file):
            sample = pd.read_csv(combined_file, index_col=0, parse_dates=True)
            print(sample.head())