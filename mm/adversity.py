"""Client adversity profiling and spread economics.

A trade is adverse at horizon h (from the LP's perspective) when closing the
position h seconds later loses money:

    pnl(h) = side * volume * (mid_h - trade_price) < 0.

The functions here measure how toxic each client's flow is, whether trading with
them is profitable, and the minimum half-spread that would make a client break
even if we always quoted symmetrically around the mid.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .data import HORIZONS


def adverse_mask(df: pd.DataFrame, h: int) -> pd.Series:
    return df["side"] * (df[f"mid_{h}"] - df["trade_price"]) < 0


def adversity_profile(df: pd.DataFrame, client: str,
                      horizons: List[int] = HORIZONS) -> List[float]:
    """Adversity percentage (0-100) of one client at each horizon."""
    d = df[df["client"] == client]
    return [float(100.0 * adverse_mask(d, h).mean()) for h in horizons]


def profile_table(df: pd.DataFrame, horizons: List[int] = HORIZONS) -> pd.DataFrame:
    rows = {c: adversity_profile(df, c, horizons) for c in sorted(df["client"].unique())}
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=[f"h={h}" for h in horizons]
    )


def expected_pnl(df: pd.DataFrame, client: str,
                 horizons: List[int] = HORIZONS) -> dict:
    """Expected per-trade PnL of a client at each horizon and in aggregate."""
    d = df[df["client"] == client]
    side, vol, tp = d["side"], d["volume"], d["trade_price"]
    per_h = [float((side * vol * (d[f"mid_{h}"] - tp)).mean()) for h in horizons]
    mean_mid = d[[f"mid_{h}" for h in horizons]].mean(axis=1)
    aggregate = float((side * vol * (mean_mid - tp)).mean())
    return {"per_horizon": per_h, "aggregate": aggregate}


def classify_client(df: pd.DataFrame, client: str) -> str:
    return "profitable" if expected_pnl(df, client)["aggregate"] >= 0 else "costly"


def min_half_spread(df: pd.DataFrame, client: str,
                    horizons: List[int] = HORIZONS) -> float:
    """Smallest non-negative half-spread making expected aggregate PnL >= 0.

    Quoting symmetrically at mid +/- delta gives an execution price
    trade_price = mid - side * delta, so the aggregate PnL per trade becomes
    side*vol*(mean_mid - mid) + vol*delta. Averaging, the break-even delta is
    delta* = max(0, -A / mean(vol)) with A = mean(side*vol*(mean_mid - mid)).
    """
    d = df[df["client"] == client]
    side, vol = d["side"], d["volume"]
    mean_mid = d[[f"mid_{h}" for h in horizons]].mean(axis=1)
    a = float((side * vol * (mean_mid - d["mid"])).mean())
    return max(0.0, -a / float(vol.mean()))


def profitability_table(df: pd.DataFrame,
                        horizons: List[int] = HORIZONS) -> pd.DataFrame:
    rows = []
    for c in sorted(df["client"].unique()):
        ep = expected_pnl(df, c, horizons)
        rows.append([c, *ep["per_horizon"], ep["aggregate"],
                     classify_client(df, c), min_half_spread(df, c, horizons)])
    cols = (["client"] + [f"h={h}" for h in horizons]
            + ["aggregate", "class", "min_half_spread"])
    return pd.DataFrame(rows, columns=cols)
