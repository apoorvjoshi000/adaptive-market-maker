"""Backtest of the quoting strategy on the tape used as a proxy.

For every trade we know the volatility, the time of day, and the adversity score
from the model. The quoting function sets the half-spreads; the trade fills with
probability lambda * exp(-gamma * delta / sigma); a fill earns the spread plus
the (adverse) drift to the closing mids and moves the inventory. At the end of
each day an inventory penalty phi * E[I^2] * sigma_day is charged, where
E[I^2] = E[I]^2 + Var(I) so the inventory risk that random fills create is
priced. Fills are taken fractionally (by their probability), which removes Monte
Carlo noise and makes the run reproducible.

The true (lambda, gamma, phi) are not knowable in advance and shift between
regimes, so the strategy is scored as the average Sharpe-like score across a
small set of plausible regimes rather than tuned to any single one.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import HORIZONS
from .metrics import max_drawdown, sharpe
from .model import AdversityModel, build_features
from .quoting import C_MIN, M_MAX, PARAMS, quote

# (lambda, gamma, phi): fill base rate, fill decay, inventory penalty.
ROBUST_SCENARIOS = [(0.9, 1.0, 1.0e-3), (0.7, 1.5, 4.0e-3), (1.0, 0.8, 3.0e-4)]


def prepare(df: pd.DataFrame, model: AdversityModel) -> list:
    """Per-day arrays the simulation needs (built once)."""
    df, feat, _ = build_features(df)
    probs = np.column_stack([model.predict_proba_frame(feat, h)
                             for h in model.horizons])
    df = df.copy()
    df["alpha"] = probs.mean(axis=1)
    df["sigma_px"] = feat["sigma"].to_numpy() * df["mid"].to_numpy()
    df["eta"] = feat["eta"].to_numpy()
    df["mean_future_mid"] = df[[f"mid_{h}" for h in HORIZONS]].mean(axis=1)

    days = []
    for _, d in df.groupby("day", sort=True):
        sig = d["sigma_px"].to_numpy(float)
        sig = np.where(sig > 0, sig, 1e-6)
        days.append((d["side"].to_numpy(float), d["volume"].to_numpy(float),
                     d["mid"].to_numpy(float), d["mean_future_mid"].to_numpy(float),
                     sig, d["alpha"].to_numpy(float), d["eta"].to_numpy(float)))
    return days


def simulate(days: list, params: dict, lam: float, gam: float, phi: float) -> dict:
    cb, ca = params["c_base"], params["c_adv"]
    ci, ct, iref = params["c_inv"], params["c_time"], params["i_ref"]
    daily = []
    for side, vol, mid, fut, sig, alpha, eta in days:
        inv = 0.0
        inv_var = 0.0
        pnl = 0.0
        for k in range(len(side)):
            s = sig[k]
            base = cb + ca * alpha[k]
            skew = (ci + ct * eta[k]) * math.tanh(inv / iref)
            m = (base + skew) if side[k] > 0 else (base - skew)
            m = C_MIN if m < C_MIN else (M_MAX if m > M_MAX else m)
            delta = m * s
            f = lam * math.exp(-gam * m)
            if f > 1.0:
                f = 1.0
            tp = mid[k] - side[k] * delta
            pnl += f * side[k] * vol[k] * (fut[k] - tp)
            inv += f * side[k] * vol[k]
            inv_var += f * (1.0 - f) * vol[k] * vol[k]
        pnl -= phi * (inv * inv + inv_var) * sig.mean()
        daily.append(pnl)
    arr = np.array(daily)
    return {"total_pnl": float(arr.sum()), "daily_pnl": arr,
            "score": sharpe(arr), "pnl_std": float(arr.std()),
            "max_drawdown": max_drawdown(arr)}


def fixed_spread_params(multiple: float) -> dict:
    return {"c_base": multiple, "c_adv": 0.0, "c_inv": 0.0, "c_time": 0.0,
            "i_ref": 1.0}


def robust_score(days: list, params: dict) -> float:
    return float(np.mean([simulate(days, params, *sc)["score"]
                          for sc in ROBUST_SCENARIOS]))


def tune(days: list, base: dict = PARAMS) -> "tuple[dict, float]":
    """Coordinate-descent search for the quoting gains on the proxy data."""
    grid = {
        "c_base": [0.5, 0.8, 1.1, 1.3, 1.6],
        "c_adv": [0.0, 0.8, 1.6, 2.2, 3.0],
        "c_inv": [0.0, 0.6, 1.2, 1.8],
        "c_time": [0.0, 0.8, 1.4, 1.8, 2.4],
        "i_ref": [800.0, 1500.0, 2500.0, 4000.0],
    }
    p = dict(base)
    best = robust_score(days, p)
    for _ in range(2):
        for key, values in grid.items():
            for v in values:
                trial = dict(p, **{key: v})
                s = robust_score(days, trial)
                if s > best:
                    best, p = s, trial
    return p, best


def compare(days: list, params: dict = PARAMS) -> pd.DataFrame:
    """Dynamic strategy vs fixed-spread baselines, per regime."""
    rows = []
    for lam, gam, phi in ROBUST_SCENARIOS:
        tag = f"lam={lam},gam={gam},phi={phi}"
        for mult in (0.5, 1.0, 2.0):
            r = simulate(days, fixed_spread_params(mult), lam, gam, phi)
            rows.append([tag, f"fixed m={mult}", r["total_pnl"], r["pnl_std"],
                         r["score"], r["max_drawdown"]])
        r = simulate(days, params, lam, gam, phi)
        rows.append([tag, "dynamic", r["total_pnl"], r["pnl_std"], r["score"],
                     r["max_drawdown"]])
    return pd.DataFrame(rows, columns=["regime", "strategy", "total_pnl",
                                       "pnl_std", "score", "max_drawdown"])
