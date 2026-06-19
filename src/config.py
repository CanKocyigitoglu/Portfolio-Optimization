"""Configuration for the portfolio-optimization pipeline.

Edit the universe, date window, or model parameters here; every other module
reads its defaults from this file.
"""

# Market proxy used as the single CAPM factor (S&P 500 index).
MARKET_TICKER = "^GSPC"

# Investable universe (10 large-cap US equities).
TICKERS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "NFLX",
    "JPM",
    "AVGO",
]

# Historical window: 5 years of daily data.
START_DATE = "2020-03-15"
END_DATE = "2025-03-15"

# Annual risk-free rate used to compute excess returns.
RISK_FREE_RATE = 0.02

# Trading days per year.
# 365.2425 (avg days/year) * 5/7 (work days/week) - 10 (holidays) ≈ 251.
TRADING_DAYS = 251
