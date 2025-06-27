# Market Rally Prediction via Volatility Analysis

A practical approach to analysing market volatility using ensemble methods. This project helps spot potential market rallies by examining the relationship between realised and implied volatility.

// ... existing code ...

### Market Data Sources
- **SPX**: The S&P 500 Index (our main market benchmark)
- **VIX**: The CBOE Volatility Index (the market's "fear gauge")

We fetch this data through Finnhub's API, with built-in rate limiting and local caching to keep things running smoothly.

// ... existing code ...

## Getting Started

Pop this in your terminal:
```bash
pip install -r requirements.txt
```

You'll need:
- A Finnhub API key (set it as `FINNHUB_API_KEY` in your environment)

// ... existing code ...

Found a bug? Got an idea? Open an issue or send us a pull request.

## Licence

[MIT](LICENCE)
