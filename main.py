"""Run the full CAPM + quadratic-programming portfolio-optimization pipeline.

Usage:
    python main.py

Downloads historical data, estimates CAPM parameters for each asset, builds the
single-factor covariance matrix, traces the efficient frontier, and saves a plot
to ``efficient_frontier.png``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.capm import daily_returns, estimate_capm
from src.config import TICKERS
from src.data import download_market, download_prices
from src.optimization import build_covariance, efficient_frontier
from src.plotting import plot_efficient_frontier

OUTPUT_PLOT = "efficient_frontier.png"


def main():
    print("Downloading market data (^GSPC)...")
    market = download_market()
    market_returns = daily_returns(market)
    market_variance = float(market_returns.var())

    print(f"Downloading {len(TICKERS)} assets...")
    prices = download_prices()

    print("\nEstimating CAPM parameters:")
    results = {ticker: estimate_capm(prices[ticker], market_returns) for ticker in TICKERS}
    for ticker in TICKERS:
        r = results[ticker]
        print(
            f"  {ticker:6s} alpha={r.alpha:+.5f}  beta={r.beta:.3f}  "
            f"E[r]={r.expected_return:.5f}  idio={r.idiosyncratic_risk:.6f}"
        )

    betas = [results[t].beta for t in TICKERS]
    idiosyncratic = [results[t].idiosyncratic_risk for t in TICKERS]
    expected_returns = [results[t].expected_return for t in TICKERS]

    cov = build_covariance(betas, idiosyncratic, market_variance)

    print("\nTracing the efficient frontier...")
    targets, risks, weights = efficient_frontier(cov, expected_returns, n_points=10)

    frontier = pd.DataFrame({"target_return": targets, "risk": risks})
    print(frontier.to_string(index=False))

    print("\nOptimal weights per target return (%):")
    weights_df = pd.DataFrame(
        np.round(weights * 100, 1),
        index=[round(t, 5) for t in targets],
        columns=TICKERS,
    )
    print(weights_df.to_string())

    plot_efficient_frontier(risks, targets, save_path=OUTPUT_PLOT)
    print(f"\nSaved efficient-frontier plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
