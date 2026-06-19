"""Single-factor (CAPM) estimation of alpha, beta, and risk.

Each asset is regressed against the market proxy:

    r_i - r_f = alpha_i + beta_i * (r_m - r_f) + epsilon_i

from which we recover the systematic exposure (beta), the CAPM expected
return, and the idiosyncratic (residual) variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from .config import RISK_FREE_RATE, TRADING_DAYS


@dataclass
class CAPMResult:
    """Estimated CAPM parameters for a single asset (daily units)."""

    alpha: float
    beta: float
    expected_return: float
    idiosyncratic_risk: float


def daily_returns(prices):
    """Daily percentage returns from a price frame.

    Accepts either a Series or a (possibly multi-indexed) DataFrame and always
    returns a 1-D Series of close-to-close returns with the first NaN dropped.
    """
    close = prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.pct_change().dropna()


def estimate_capm(
    stock_prices,
    market_returns,
    risk_free_rate=RISK_FREE_RATE,
    trading_days=TRADING_DAYS,
):
    """Estimate CAPM parameters for one asset.

    Args:
        stock_prices: Price DataFrame for the asset (must contain ``Close``).
        market_returns: Daily returns of the market proxy (a Series).
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year used to convert the rate to daily.

    Returns:
        A :class:`CAPMResult` with daily-frequency values.
    """
    daily_rf = risk_free_rate / trading_days

    stock_returns = daily_returns(stock_prices)
    excess_stock = stock_returns - daily_rf
    excess_market = market_returns - daily_rf

    # Align on common dates so the regression is well defined.
    excess_stock, excess_market = excess_stock.align(excess_market, join="inner")

    X = sm.add_constant(excess_market)
    model = sm.OLS(excess_stock, X).fit()
    alpha, beta = model.params

    expected_return = daily_rf + beta * (market_returns.mean() - daily_rf)

    residuals = excess_stock - model.predict(X)
    idiosyncratic_risk = residuals.var()

    return CAPMResult(
        alpha=float(alpha),
        beta=float(beta),
        expected_return=float(expected_return),
        idiosyncratic_risk=float(idiosyncratic_risk),
    )
