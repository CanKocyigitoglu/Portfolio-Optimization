"""Efficient-frontier visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_efficient_frontier(risks, returns, ax=None, save_path=None):
    """Plot expected return against portfolio risk.

    Args:
        risks: Portfolio standard deviations (x-axis).
        returns: Target expected returns (y-axis).
        ax: Optional existing Matplotlib axes to draw on.
        save_path: If given, the figure is written to this path.

    Returns:
        The Matplotlib axes the frontier was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(risks, returns, marker="o", linestyle="-")
    ax.set_xlabel("Portfolio Risk (daily std. dev.)")
    ax.set_ylabel("Expected Return (daily)")
    ax.set_title("Efficient Frontier")
    ax.grid(True)

    if save_path:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax
