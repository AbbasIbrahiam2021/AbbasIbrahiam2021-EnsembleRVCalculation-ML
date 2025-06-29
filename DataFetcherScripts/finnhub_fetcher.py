"""
Market Data Fetcher for Volatility Analysis

A tool for fetching and analyzing SPX and VIX market data.
Requires a Finnhub API key stored in config.json.

Usage:
    from finnhub_fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher()
    data = fetcher.get_market_data()
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import time
import requests
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f).get('finnhub_api_key')
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        return None

class MarketDataFetcher:
    """Handles market data retrieval and processing for volatility analysis"""
    
    def __init__(self, config_path='config.json'):
        """Initialize the data fetcher with configuration"""
        self.api_key = load_config(config_path)
        if not self.api_key:
            raise ValueError("Missing API key in configuration")
        self.base_url = "https://finnhub.io/api/v1"
        
    def fetch_data(self, symbol, start_date, end_date):
        """
        Retrieve market data for a given symbol and date range.
        
        Parameters:
            symbol (str): Market symbol (e.g., '^GSPC' for SPX, '^VIX' for VIX)
            start_date (datetime): Start date for data retrieval
            end_date (datetime): End date for data retrieval
            
        Returns:
            pandas.DataFrame: Market data with OHLCV columns
        """
        try:
            # Convert dates to Unix timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(pd.Timestamp(end_date).timestamp())
            
            # Map SPX to its market symbol
            if symbol.upper() in ["SPX", "^SPX"]:
                symbol = "^GSPC"
            
            # Prepare API request
            params = {
                "symbol": symbol,
                "resolution": "D",
                "from": start_ts,
                "to": end_ts,
                "token": self.api_key
            }
            
            # Fetch market data
            response = requests.get(f"{self.base_url}/stock/candle", params=params)
            data = response.json()
            
            if data.get('s') != 'ok':
                logger.error(f"Failed to fetch data for {symbol}")
                return None
            
            # Process response into DataFrame
            df = pd.DataFrame({
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }, index=pd.to_datetime([datetime.fromtimestamp(x) for x in data['t']]))
            
            df.index.name = 'Date'
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Data retrieval error: {str(e)}")
            return None

    def get_market_data(self, start_date=None, end_date=None):
        """
        Retrieve and process market data for volatility analysis.
        
        Parameters:
            start_date (datetime, optional): Start date, defaults to one year ago
            end_date (datetime, optional): End date, defaults to today
            
        Returns:
            pandas.DataFrame: Combined market data with volatility metrics
        """
        # Set default date range to last year
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
            
        # Retrieve SPX data and calculate realized volatility
        logger.info("Retrieving SPX data")
        spx = self.fetch_data("^GSPC", start_date, end_date)
        if spx is not None:
            spx['Returns'] = spx['Close'].pct_change()
            spx['RealizedVol'] = spx['Returns'].rolling(21).std() * np.sqrt(252) * 100
            
        # Retrieve VIX data
        logger.info("Retrieving VIX data")
        vix = self.fetch_data("^VIX", start_date, end_date)
        if vix is not None:
            vix = vix[['Close']].rename(columns={'Close': 'VIX'})
            
        # Combine and process data
        if spx is not None and vix is not None:
            combined = pd.merge(
                spx[['Close', 'RealizedVol']],
                vix,
                left_index=True,
                right_index=True,
                how='outer'
            )
            combined.columns = ['SPX_Close', 'SPX_RealizedVol', 'VIX_Close']
            return combined
        
        return None

def main():
    """Example usage of the MarketDataFetcher"""
    try:
        fetcher = MarketDataFetcher()
        data = fetcher.get_market_data()
        
        if data is not None:
            print("\nMarket Data Summary:")
            print(data.tail())
            
            # Calculate and display volatility premium
            data['VolatilityPremium'] = data['SPX_RealizedVol'] - data['VIX_Close']
            print("\nVolatility Premium Analysis:")
            print(data['VolatilityPremium'].describe())
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Ensure config.json exists with valid API credentials")

if __name__ == "__main__":
    main() 