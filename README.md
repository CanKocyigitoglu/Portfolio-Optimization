# Portfolio Optimisation Using CAPM and Quadratic Programming

Optimise a long-only stock portfolio by combining the **Capital Asset Pricing
Model (CAPM)** with **quadratic programming**. Each asset's risk and expected
return are estimated from a single-factor regression against the market, and an
efficient frontier is traced by minimising portfolio variance for a range of
target returns.

This project was completed as part of a Machine Learning for Finance coursework.

## How it works

1. **Data** — Download 5 years of daily prices for 10 large-cap US equities plus
   the S&P 500 (`^GSPC`) as the market proxy, using [yfinance](https://github.com/ranaroussi/yfinance).
2. **CAPM estimation** — For each asset, regress excess stock returns on excess
   market returns (OLS via `statsmodels`) to recover **alpha**, **beta**, the
   CAPM **expected return**, and the **idiosyncratic (residual) variance**.
3. **Covariance** — Build the covariance matrix from the single-factor model:
   `Cov(i, j) = βᵢ · βⱼ · σ²_market`, with each asset's idiosyncratic variance
   added on the diagonal.
4. **Optimisation** — For each target return, solve the quadratic program

   ```
   minimise    wᵀ Σ w
   subject to  Σ wᵢ = 1,   μᵀw = target,   w ≥ 0
   ```

   with [CVXOPT](https://cvxopt.org/), then plot the resulting efficient frontier.

## Project structure

```
.
├── main.py                       # End-to-end pipeline entry point
├── src/
│   ├── config.py                 # Universe, date window, model parameters
│   ├── data.py                   # yfinance price downloads
│   ├── capm.py                   # CAPM alpha/beta/risk estimation
│   ├── optimization.py           # Covariance matrix + QP optimiser
│   └── plotting.py               # Efficient-frontier plot
├── Portfolio-Optimisation.ipynb  # Original exploratory analysis (demo)
├── requirements.txt
└── README.md
```

The `src/` package is the reusable, refactored version; the notebook preserves
the original step-by-step exploration.

## Installation

```bash
git clone <repository-url>
cd Portfolio-Optimization-1
pip install -r requirements.txt
```

## Usage

Run the full pipeline as a script:

```bash
python main.py
```

This prints each asset's CAPM parameters and the efficient-frontier table, then
saves the plot to `efficient_frontier.png`.

Or explore interactively in the notebook:

```bash
jupyter notebook Portfolio-Optimisation.ipynb
```

## Configuration

Tickers, the date range, the risk-free rate, and the trading-day count all live
in [`src/config.py`](src/config.py). Edit them there to change the investable
universe or analysis window.

## Technologies

Python · NumPy · pandas · Matplotlib · yfinance · statsmodels · CVXOPT
