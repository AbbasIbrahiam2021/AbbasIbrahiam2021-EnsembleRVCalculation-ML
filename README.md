# AbbasIbrahiam2021-EnsembleRVCalculation-ML

# Market Rally Prediction via Volatility Analysis

A practical approach to analysing market volatility using ensemble methods. This project helps spot potential market rallies by examining the relationship between realised and implied volatility.

## Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [Getting Started](#getting-started)
- [Under the Bonnet](#under-the-bonnet)
- [Quick Start](#quick-start)
- [Maths Behind the Magic](#maths-behind-the-magic)

## Overview

We've built this tool to combine several volatility measures into one robust indicator. By comparing our calculated realised volatility against market-implied volatility, we can spot potential market moves before they happen.

## Key Components

### Market Data Sources
- **SPX**: The S&P 500 Index (our main market benchmark)
- **VIX**: The CBOE Volatility Index (the market's "fear gauge")

We fetch this data through Alpha Vantage's API, with built-in rate limiting and local caching to keep things running smoothly.

### Volatility Calculations
We use several tried-and-tested volatility estimators:
- Garman-Klass
- Rogers-Satchell
- Parkinson
- Close-to-Close
- Yang-Zhang

### Smart Weighting
Our system uses machine learning to work out the optimal mix of these estimators:
- Linear Regression
- Ridge Regression (for more stable results)
- LASSO (when we need to be selective)

## Getting Started

Pop this in your terminal:
```bash
pip install -r requirements.txt
```

You'll need:
- An Alpha Vantage API key (set it as `ALPHA_VANTAGE_KEY` in your environment)

## Under the Bonnet

### Data Handling
1. Live market data fetching (mindful of API limits)
2. Smart local caching
3. Rolling 21-day calculations
4. Automated quality checks

### The Core Calculation

Here's how we calculate realised volatility:

$$ RV = \sum_{i=1}^{k} a_i \sigma_i $$

Where:
- $RV$ is our Realised Volatility
- $a_i$ are our clever ML-derived weights
- $\sigma_i$ are the individual volatility measures
- $k$ is how many measures we're using

### Volatility Premium

$$ Volatility\ Premium = RV - IV $$

- RV = Our calculated Realised Volatility
- IV = Market-implied Volatility (from VIX)

## Quick Start

Here's a simple example:

```python
from volatility_system import DataFetcher, VolatilityEstimator

# Set things up
fetcher = DataFetcher()
estimator = VolatilityEstimator()

# Get our market data
spx_data = fetcher.fetch_data("SPX")
vix_data = fetcher.fetch_data("VIX")

# Work out the premium
vol_premium = estimator.calculate_premium(spx_data, vix_data)
```

### Reading the Signals

1. **When RV < IV (Negative Premium)**
   - The market might be too worried
   - Often seen near market bottoms
   - Could be time to buy

2. **When RV > IV (Positive Premium)**
   - The market might be too complacent
   - Usually spotted near market tops
   - Worth being cautious

## Maths Behind the Magic

### Our Volatility Measures

1. **Garman-Klass**
   - Uses the day's full price range
   - Sharper than just looking at closes

2. **Rogers-Satchell**
   - Handles trending markets well
   - Stays accurate when prices drift

3. **Parkinson**
   - Focuses on price ranges
   - Brilliant for choppy markets

### The Optimisation Bit

We fine-tune our weights by solving:

$$ \min_{a_i} \sum_{t=1}^{T} (RV_t - \sum_{i=1}^{k} a_i \sigma_{i,t})^2 $$

With these rules:
- All weights must add up to 1
- No negative weights allowed

## Want to Help?

Found a bug? Got an idea? Open an issue or send us a pull request.

## Licence

[MIT](LICENCE)
