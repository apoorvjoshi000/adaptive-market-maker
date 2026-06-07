"""Optimal externalization policy.

Using the adverse-selection model, a trade is externalized (its PnL netted to
zero) when its predicted adverse probability exceeds a cutoff theta, and kept
otherwise. The kept PnL at horizon h is

    sum over trades with p <= theta of  side * volume * (mid_h - trade_price).

theta is chosen on the validation split to maximize PnL, then the realized PnL is
reported on the held-out test split. Thresholds can be global or per client.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import AdversityModel, _day_split, build_features

THETA_GRID = np.linspace(0.0, 1.0, 201)


class Externalizer:
    def __init__(self, df: pd.DataFrame, model: AdversityModel):
        self.df, feat, _ = build_features(df)
        self.model = model
        self.masks = _day_split(self.df)
        self.prob = {h: model.predict_proba_frame(feat, h) for h in model.horizons}
        self.pnl = {h: (self.df["side"] * self.df["volume"]
                        * (self.df[f"mid_{h}"] - self.df["trade_price"])).to_numpy()
                    for h in model.horizons}
        self._client = self.df["client"].to_numpy()

    def _kept_pnl(self, sel: np.ndarray, h: int, theta: float) -> float:
        keep = sel & (self.prob[h] <= theta)
        return float(self.pnl[h][keep].sum())

    def _best_theta(self, sel: np.ndarray, h: int):
        pnls = [self._kept_pnl(sel, h, t) for t in THETA_GRID]
        i = int(np.argmax(pnls))
        return float(THETA_GRID[i]), float(pnls[i])

    def optimal_threshold(self, horizon: int, client: str = None) -> dict:
        val, test = self.masks["validation"], self.masks["test"]
        if client is not None:
            sv = val & (self._client == client)
            st = test & (self._client == client)
            theta, vpnl = self._best_theta(sv, horizon)
            return {"theta": theta, "validation_pnl": vpnl,
                    "test_pnl": self._kept_pnl(st, horizon, theta)}
        theta, vpnl = self._best_theta(val, horizon)
        return {"theta": theta, "validation_pnl": vpnl,
                "test_pnl": self._kept_pnl(test, horizon, theta)}

    def pnl_curve(self, horizon: int, split: str = "validation") -> np.ndarray:
        sel = self.masks[split]
        return np.array([self._kept_pnl(sel, horizon, t) for t in THETA_GRID])

    def summary_table(self) -> pd.DataFrame:
        """Per (client, horizon) optimal theta and held-out test PnL."""
        rows = []
        for c in sorted(self.df["client"].unique()):
            for h in self.model.horizons:
                r = self.optimal_threshold(h, client=c)
                rows.append([c, h, r["theta"], r["test_pnl"]])
        return pd.DataFrame(rows, columns=["client", "horizon", "theta", "test_pnl"])

    def uplift(self, horizon: int) -> dict:
        """Test PnL with the global optimal theta vs no externalization."""
        test = self.masks["test"]
        g = self.optimal_threshold(horizon)
        none = self._kept_pnl(test, horizon, 1.0)
        return {"theta": g["theta"], "with_externalization": g["test_pnl"],
                "without": none}
