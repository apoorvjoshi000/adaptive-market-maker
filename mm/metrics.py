"""Performance metrics shared across the package."""

from __future__ import annotations

import numpy as np


def sharpe(daily_pnl: np.ndarray, floor: float = 1.0) -> float:
    """Sharpe-like score: total PnL divided by the daily-PnL volatility.

    The volatility is floored so a near-flat series does not blow the ratio up.
    """
    daily_pnl = np.asarray(daily_pnl, dtype=float)
    return float(daily_pnl.sum() / max(daily_pnl.std(), floor))


def max_drawdown(daily_pnl: np.ndarray) -> float:
    """Largest peak-to-trough drop of the cumulative PnL (>= 0)."""
    cum = np.cumsum(np.asarray(daily_pnl, dtype=float))
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())
