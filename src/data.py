"""Historical price downloads via yfinance."""

from __future__ import annotations

import yfinance as yf

from .config import END_DATE, MARKET_TICKER, START_DATE, TICKERS


def download_prices(tickers=None, start=START_DATE, end=END_DATE):
    """Download daily price data for each ticker.

    Returns a dict mapping ``ticker -> DataFrame`` of OHLCV data
    (auto-adjusted for splits and dividends).
    """
    tickers = tickers or TICKERS
    return {
        ticker: yf.download(ticker, start=start, end=end, auto_adjust=True)
        for ticker in tickers
    }


def download_market(start=START_DATE, end=END_DATE):
    """Download daily price data for the market proxy (S&P 500)."""
    return yf.download(MARKET_TICKER, start=start, end=end, auto_adjust=True)
