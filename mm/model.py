"""Adverse-selection prediction model.

One gradient-boosted classifier per horizon predicts the probability that a
trade is adverse. Only information available at the moment of the trade is used;
the forward mids define the labels and never enter the features. The data is
split by day into train/validation/test so the evaluation is strictly out of
time, as it must be for anything that drives trading decisions.

Feature vector (index -> feature):
    0 side                LP side (+1 buy, -1 sell)
    1 volume              trade volume
    2 spread              quoted spread at the trade
    3 signed_edge         side * (mid - trade_price), the half-spread captured
    4 sigma               realised volatility of the previous 20 mid returns
    5 eta                 elapsed fraction of the trading day
    6 mom5                side * recent 5-trade mid return (signed momentum)
    7 mom20               side * recent 20-trade mid return (signed momentum)
    8 signed_volume       side * volume
    9.. client one-hot    one column per client, alphabetical
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from .adversity import adverse_mask
from .data import HORIZONS

EPS = 1e-12


def _realised_vol(mid: pd.Series, n: int = 20) -> pd.Series:
    r = mid.pct_change()
    return np.sqrt(r.pow(2).shift(1).rolling(n, min_periods=1).mean())


def build_features(df: pd.DataFrame) -> "tuple[pd.DataFrame, list[str]]":
    """Return the engineered feature frame and the client list (column order)."""
    df = df.sort_values(["day", "second"]).reset_index(drop=True)
    clients = sorted(df["client"].unique())
    session = df["second"].max()

    feat = pd.DataFrame(index=df.index)
    feat["side"] = df["side"].astype(float)
    feat["volume"] = df["volume"].astype(float)
    feat["spread"] = df["spread"].astype(float)
    feat["signed_edge"] = df["side"] * (df["mid"] - df["trade_price"])
    feat["sigma"] = (df.groupby("day", group_keys=False)["mid"]
                     .apply(_realised_vol).fillna(0.0))
    feat["eta"] = (df["second"] / session).clip(0.0, 1.0)
    m5 = df.groupby("day", group_keys=False)["mid"].apply(lambda m: m / m.shift(5) - 1.0)
    m20 = df.groupby("day", group_keys=False)["mid"].apply(lambda m: m / m.shift(20) - 1.0)
    feat["mom5"] = (df["side"] * m5).fillna(0.0)
    feat["mom20"] = (df["side"] * m20).fillna(0.0)
    feat["signed_volume"] = df["side"] * df["volume"]
    for c in clients:
        feat[f"client_{c}"] = (df["client"] == c).astype(float)
    return df, feat, clients


def _day_split(df: pd.DataFrame):
    days = sorted(df["day"].unique())
    n = len(days)
    tr = set(days[: int(0.6 * n)])
    va = set(days[int(0.6 * n): int(0.8 * n)])
    te = set(days[int(0.8 * n):])
    return {"train": df["day"].isin(tr).to_numpy(),
            "validation": df["day"].isin(va).to_numpy(),
            "test": df["day"].isin(te).to_numpy()}


class AdversityModel:
    """Per-horizon classifier ensemble for adverse-trade probability."""

    def __init__(self, horizons: List[int] = HORIZONS):
        self.horizons = horizons
        self.models: dict = {}
        self.clients: List[str] = []

    def fit(self, df: pd.DataFrame) -> "AdversityModel":
        df, feat, clients = build_features(df)
        self.clients = clients
        self._columns = list(feat.columns)
        X = feat.to_numpy()
        masks = _day_split(df)
        for h in self.horizons:
            y = adverse_mask(df, h).to_numpy().astype(int)
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6,
                l2_regularization=1.0, random_state=0)
            clf.fit(X[masks["train"]], y[masks["train"]])
            self.models[h] = clf
        return self

    def predict_proba_frame(self, feat: pd.DataFrame, horizon: int) -> np.ndarray:
        return self.models[horizon].predict_proba(feat.to_numpy())[:, 1]

    def predict(self, *, client: str, side: float, volume: float, mid: float,
                spread: float, trade_price: float, horizon: int,
                sigma: float = 0.0, eta: float = 0.5,
                mom5: float = 0.0, mom20: float = 0.0) -> float:
        """Adverse probability for a single trade (history features optional)."""
        row = {"side": side, "volume": volume, "spread": spread,
               "signed_edge": side * (mid - trade_price), "sigma": sigma,
               "eta": eta, "mom5": mom5, "mom20": mom20,
               "signed_volume": side * volume}
        for c in self.clients:
            row[f"client_{c}"] = 1.0 if client == c else 0.0
        x = np.array([[row[col] for col in self._columns]], dtype=float)
        return float(self.models[horizon].predict_proba(x)[0, 1])

    def metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """accuracy / precision / recall / auc / log_loss averaged over horizons."""
        df, feat, _ = build_features(df)
        X = feat
        masks = _day_split(df)
        out = {}
        for split, mask in masks.items():
            acc, prec, rec, auc, ll = [], [], [], [], []
            for h in self.horizons:
                y = adverse_mask(df, h).to_numpy().astype(int)[mask]
                p = self.predict_proba_frame(X[mask], h)
                pred = (p >= 0.5).astype(int)
                acc.append(accuracy_score(y, pred))
                prec.append(precision_score(y, pred, zero_division=0))
                rec.append(recall_score(y, pred, zero_division=0))
                auc.append(roc_auc_score(y, p))
                ll.append(log_loss(y, p, labels=[0, 1]))
            out[split] = [np.mean(acc), np.mean(prec), np.mean(rec),
                          np.mean(auc), np.mean(ll)]
        return pd.DataFrame.from_dict(
            out, orient="index",
            columns=["accuracy", "precision", "recall", "auc", "log_loss"]
        ).loc[["train", "validation", "test"]]
