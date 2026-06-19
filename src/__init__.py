"""CAPM-based portfolio optimization with quadratic programming.

Modules:
    config        Universe, date window, and model parameters.
    data          Historical price downloads via yfinance.
    capm          Single-factor (CAPM) alpha/beta/risk estimation.
    optimization  Covariance assembly and the quadratic-programming optimizer.
    plotting      Efficient-frontier visualization.
"""
