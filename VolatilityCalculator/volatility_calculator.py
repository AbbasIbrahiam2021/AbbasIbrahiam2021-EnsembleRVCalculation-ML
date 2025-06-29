"""
SPX Volatility Calculator

This script calculates various volatility metrics for S&P 500 (SPX) market data using the same
21-day calculation window as the VIX index for direct comparison.

Input:
-------
The script expects a CSV file with the following columns:
- Date: Trading date in MM/DD/YY format
- Open: Opening price
- High: Highest price of the day
- Low: Lowest price of the day
- Close: Closing price

The default input file path is 'market data/SPX_History.csv'

Output:
--------
The script generates a CSV file 'spx_volatility_results.csv' with the following columns:
- Date: Trading date
- Close: Closing price
- Daily_Return: Log returns
- Close_to_Close_Vol: Standard deviation of log returns (21-day window)
- Parkinson_Vol: Volatility using high-low range (21-day window)
- Garman_Klass_Vol: Volatility using OHLC prices (21-day window)
- Yang_Zhang_Vol: Volatility using open, close, and previous close (21-day window)
- Rogers_Satchell_Vol: Volatility using OHLC prices (21-day window)

All volatility metrics use:
- 21-day rolling window (exactly matching VIX methodology for direct comparison)
- Annualised by multiplying by √252 (trading days per year)
- Expressed as percentages (like VIX)
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class VolatilityParameters:
    """Parameters for volatility calculations.
    
    Attributes:
        rolling_window: Number of trading days for rolling calculations 
                      (default: 21 to exactly match VIX calculation window)
        annualisation_factor: Number of trading days in a year for annualising results
    """
    rolling_window: int = 21  # 21 trading days to exactly match VIX calculation window
    annualisation_factor: float = 252.0  # Number of trading days in a year

class VolatilityCalculator:
    """
    A class to calculate various volatility metrics for SPX market data.
    All volatility metrics use a 21-day window (matching VIX), are annualised 
    (multiplied by √252), and converted to percentage to match VIX format.
    
    The class implements five different volatility estimation methods, each using
    the same 21-day window as VIX for direct comparison:
    1. Close-to-Close: Traditional volatility using closing prices
    2. Parkinson: Using daily high-low range
    3. Garman-Klass: Using OHLC prices
    4. Yang-Zhang: Using open, close, and previous close
    5. Rogers-Satchell: Using OHLC prices with different weighting
    """
    
    def __init__(self, params: Optional[VolatilityParameters] = None):
        """
        Initialise the calculator.
        
        Parameters:
            params: VolatilityParameters object containing calculation parameters
                   If None, uses default 21-day window (matching VIX) and 
                   252 trading days/year for annualisation
        """
        self.params = params or VolatilityParameters()
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data.
        
        Parameters:
            data: DataFrame containing OHLC price data
        
        Returns:
            Validated and preprocessed DataFrame with calculated daily returns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain all required columns: {required_cols}")
        
        df = data.copy()
        # Convert Date and ensure proper format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        df = df.sort_values('Date')
        
        # Calculate returns
        df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        return df
    
    def close_to_close(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Close-to-Close volatility using standard deviation of log returns.
        Uses 21-day window to match VIX calculation methodology.
        
        Parameters:
            data: DataFrame with daily price data
            
        Returns:
            Series containing annualised volatility in percentage
        """
        return (
            data['Daily_Return']
            .rolling(window=self.params.rolling_window)  # 21-day window matching VIX
            .std() 
            * np.sqrt(self.params.annualisation_factor)
            * 100  # Convert to percentage to match VIX format
        )
    
    def parkinson(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low range.
        More efficient than close-to-close when market is continuous.
        Uses 21-day window to match VIX calculation methodology.
        
        Parameters:
            data: DataFrame with daily price data
            
        Returns:
            Series containing annualised volatility in percentage
        """
        hl_square = np.log(data['High'] / data['Low'])**2
        
        return (
            np.sqrt(1 / (4 * self.params.rolling_window * np.log(2)) 
            * hl_square.rolling(window=self.params.rolling_window).sum())  # 21-day window matching VIX
            * np.sqrt(self.params.annualisation_factor)
            * 100  # Convert to percentage to match VIX format
        )
    
    def garman_klass(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Garman-Klass volatility using OHLC prices.
        More efficient than Parkinson when opening and closing prices are considered.
        Uses 21-day window to match VIX calculation methodology.
        
        Parameters:
            data: DataFrame with daily price data
            
        Returns:
            Series containing annualised volatility in percentage
        """
        log_hl = np.log(data['High'] / data['Low'])**2
        log_co = np.log(data['Close'] / data['Open'])**2
        
        return (
            np.sqrt(
                (0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
                .rolling(window=self.params.rolling_window)  # 21-day window matching VIX
                .mean()
            ) * np.sqrt(self.params.annualisation_factor)
            * 100  # Convert to percentage to match VIX format
        )
    
    def yang_zhang(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Yang-Zhang volatility.
        Accounts for opening jumps and is independent of drift.
        Uses 21-day window to match VIX calculation methodology.
        
        Parameters:
            data: DataFrame with daily price data
            
        Returns:
            Series containing annualised volatility in percentage
        """
        # Overnight volatility
        log_co = np.log(data['Open'] / data['Close'].shift(1))
        # Open-to-Close volatility
        log_oc = np.log(data['Close'] / data['Open'])
        
        overnight_vol = log_co.rolling(window=self.params.rolling_window).var()  # 21-day window matching VIX
        open_close_vol = log_oc.rolling(window=self.params.rolling_window).var()  # 21-day window matching VIX
        
        return (
            np.sqrt(overnight_vol + open_close_vol)
            * np.sqrt(self.params.annualisation_factor)
            * 100  # Convert to percentage to match VIX format
        )
    
    def rogers_satchell(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility.
        Accounts for drift and uses all OHLC prices.
        Uses 21-day window to match VIX calculation methodology.
        
        Parameters:
            data: DataFrame with daily price data
            
        Returns:
            Series containing annualised volatility in percentage
        """
        # Calculate components
        log_ho = np.log(data['High'] / data['Open'])
        log_hc = np.log(data['High'] / data['Close'])
        log_lo = np.log(data['Low'] / data['Open'])
        log_lc = np.log(data['Low'] / data['Close'])
        
        rs = log_ho * (log_ho - log_hc) + log_lo * (log_lo - log_lc)
        
        return (
            np.sqrt(rs.rolling(window=self.params.rolling_window).mean())  # 21-day window matching VIX
            * np.sqrt(self.params.annualisation_factor)
            * 100  # Convert to percentage to match VIX format
        )
    
    def calculate_all(self, spx_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility metrics for SPX data using a 21-day window to match VIX.
        All volatility metrics are annualised and in percentage format to match VIX.
        
        Parameters:
            spx_data: DataFrame containing SPX OHLC data with columns:
                     Date, Open, High, Low, Close
        
        Returns:
            DataFrame containing:
            - Original date and price data
            - Daily returns
            - All volatility metrics in percentage format (like VIX)
            All metrics use 21-day window to match VIX methodology
        """
        # Validate and prepare data
        data = self._validate_data(spx_data)
        
        # Calculate all volatility metrics using 21-day window (matching VIX)
        results = pd.DataFrame({
            'Date': data['Date'],
            'Close': data['Close'],
            'Daily_Return': data['Daily_Return'],
            'Close_to_Close_Vol': self.close_to_close(data),
            'Parkinson_Vol': self.parkinson(data),
            'Garman_Klass_Vol': self.garman_klass(data),
            'Yang_Zhang_Vol': self.yang_zhang(data),
            'Rogers_Satchell_Vol': self.rogers_satchell(data)
        })
        
        return results

def calculate_spx_volatility(spx_path: str = "market data/SPX_History.csv",
                           params: Optional[VolatilityParameters] = None) -> pd.DataFrame:
    """
    Convenience function to calculate all volatility metrics for SPX data from a file.
    Uses 21-day window to match VIX methodology, with results annualised and in percentage format.
    
    Parameters:
        spx_path: Path to the CSV file containing SPX data
        params: Optional VolatilityParameters object for custom calculation parameters
                (default uses 21-day window to match VIX)
        
    Returns:
        DataFrame containing date, price data, and all volatility metrics
        calculated using 21-day window (matching VIX)
    """
    # Read SPX data
    spx_data = pd.read_csv(spx_path, skipinitialspace=True)
    
    # Create calculator and compute metrics using 21-day window (matching VIX)
    calculator = VolatilityCalculator(params)
    results = calculator.calculate_all(spx_data)
    
    return results

if __name__ == "__main__":
    # Calculate volatility metrics using 21-day window (matching VIX) and save to CSV
    results = calculate_spx_volatility()
    results.to_csv("spx_volatility_results.csv", index=False) 