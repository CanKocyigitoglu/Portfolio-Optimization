"""Covariance assembly and the quadratic-programming portfolio optimizer."""

from __future__ import annotations

import numpy as np
from cvxopt import matrix, solvers


def build_covariance(betas, idiosyncratic_risks, market_variance):
    """Build the single-factor (CAPM) covariance matrix.

    Off-diagonal terms come purely from shared market exposure; the diagonal
    additionally carries each asset's idiosyncratic variance:

        Cov(i, j) = beta_i * beta_j * market_variance        (i != j)
        Var(i)    = beta_i^2  * market_variance + idio_i      (i == j)
    """
    betas = np.asarray(betas, dtype=float)
    idiosyncratic_risks = np.asarray(idiosyncratic_risks, dtype=float)

    systematic = np.outer(betas, betas) * market_variance
    return systematic + np.diag(idiosyncratic_risks)


def optimize_portfolio(cov, expected_returns, target_return):
    """Minimum-variance long-only weights for a target expected return.

    Solves the quadratic program

        minimize    wᵀ Σ w
        subject to  Σ wᵢ = 1,  μᵀw = target_return,  w ≥ 0.

    Returns the optimal weight vector as a 1-D numpy array.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    n = len(expected_returns)

    P = matrix(np.asarray(cov, dtype=float))
    q = matrix(np.zeros(n))

    # Inequality constraints: -w <= 0  (i.e. w >= 0, long only).
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))

    # Equality constraints: weights sum to 1 and hit the target return.
    A = matrix(np.vstack([np.ones(n), expected_returns]))
    b = matrix(np.array([1.0, float(target_return)]))

    solvers.options["show_progress"] = False
    solution = solvers.qp(P, q, G, h, A, b)
    return np.array(solution["x"]).flatten()


def efficient_frontier(cov, expected_returns, n_points=10):
    """Trace the efficient frontier across a range of target returns.

    Returns ``(targets, risks, weights)`` where ``risks`` are portfolio
    standard deviations and ``weights`` is an ``(n_points, n_assets)`` array.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov, dtype=float)

    targets = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    risks = []
    weights = []
    for target in targets:
        w = optimize_portfolio(cov, expected_returns, target)
        risks.append(float(np.sqrt(w @ cov @ w)))
        weights.append(w)

    return targets, np.array(risks), np.array(weights)
