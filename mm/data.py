"""Synthetic market and client-flow generator.

The data is built from a simple but honest microstructure model so that the
downstream tasks have real, learnable structure rather than noise:

* The mid price follows a random walk whose volatility depends on the regime.
* Each trade leaves a permanent price impact in the client's favour (adverse to
  the liquidity provider). The impact size scales with the client's
  informativeness, so some clients are systematically toxic and others benign.
* Informed clients trade with directional persistence, so recent signed order
  flow (momentum) carries predictive information about the next move.
* At regime-shift points the volatility and the clients' informativeness change
  discontinuously, with no marker the models are allowed to use.

The recorded schema mirrors a real market-making tape: for every trade we store
the side (from the LP's perspective), volume, execution price, the mid at the
trade, the quoted spread, and the mid at several future horizons. None of the
numbers come from any external dataset; everything here is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

HORIZONS = [5, 10, 15, 20, 25, 30]  # forward horizons in seconds


@dataclass
class ClientSpec:
    name: str
    weight: float          # share of arriving flow
    informativeness: float # permanent-impact scale (toxicity), in price units
    persistence: float     # autocorrelation of trade direction in [0, 1)


@dataclass
class Regime:
    start_day: int
    vol_multiplier: float
    tox_multiplier: float


@dataclass
class MarketConfig:
    n_days: int = 40
    trades_per_day: int = 3000
    session_seconds: int = 23400          # 6.5 hours
    start_price: float = 100.0
    base_vol_bps_per_sqrt_s: float = 1.2  # diffusion scale
    base_half_spread_bps: float = 1.3     # LP quoted half-spread for the tape
    seed: int = 7
    clients: List[ClientSpec] = field(default_factory=lambda: [
        ClientSpec("A", 0.24, 0.2, 0.05),
        ClientSpec("B", 0.20, 0.5, 0.10),
        ClientSpec("C", 0.18, 0.9, 0.20),
        ClientSpec("D", 0.16, 1.5, 0.35),
        ClientSpec("E", 0.12, 2.2, 0.45),
        ClientSpec("F", 0.10, 3.2, 0.55),
    ])
    regimes: List[Regime] = field(default_factory=lambda: [
        Regime(0, 1.0, 1.0),
        Regime(14, 1.8, 1.4),   # stress: higher vol, more toxic flow
        Regime(27, 0.7, 0.7),   # calm: lower vol, benign flow
    ])


def _regime_for_day(cfg: MarketConfig, day: int) -> Regime:
    active = cfg.regimes[0]
    for r in cfg.regimes:
        if day >= r.start_day:
            active = r
    return active


def generate(cfg: MarketConfig = MarketConfig()) -> pd.DataFrame:
    """Generate the full trade tape as a DataFrame."""
    rng = np.random.default_rng(cfg.seed)
    clients = cfg.clients
    weights = np.array([c.weight for c in clients])
    weights = weights / weights.sum()

    rows = []
    for day in range(cfg.n_days):
        regime = _regime_for_day(cfg, day)
        vol = cfg.base_vol_bps_per_sqrt_s * 1e-4 * regime.vol_multiplier
        half_spread = cfg.base_half_spread_bps * 1e-4 * cfg.start_price

        # Diffusion path of the mid at one-second resolution, plus a cumulative
        # impact level that trades push around. The observed mid is their sum.
        n = cfg.session_seconds + max(HORIZONS) + 1
        steps = rng.normal(0.0, 1.0, n)
        diffusion = np.empty(n)
        diffusion[0] = cfg.start_price
        for s in range(1, n):
            diffusion[s] = diffusion[s - 1] * np.exp(vol * steps[s])
        impact_level = np.zeros(n)  # added to the path from the trade time on

        # Trade arrival times within the session.
        times = np.sort(rng.choice(np.arange(1, cfg.session_seconds),
                                   size=cfg.trades_per_day, replace=False))
        last_dir = {c.name: 0 for c in clients}

        for t in times:
            # Execution happens at the pre-trade mid; the permanent impact moves
            # the mid only afterwards, so the trade's own adverse drift shows up
            # in the forward mids and not in M0.
            mid0 = diffusion[t] + impact_level[t]

            ci = rng.choice(len(clients), p=weights)
            c = clients[ci]
            # Direction with client-specific persistence (informed flow trends).
            if rng.random() < c.persistence and last_dir[c.name] != 0:
                client_dir = last_dir[c.name]
            else:
                client_dir = 1 if rng.random() < 0.5 else -1
            last_dir[c.name] = client_dir
            # LP side is the opposite of the client's direction: the client buys
            # (LP sells, side = -1) when they expect the price to rise.
            side = -client_dir

            volume = int(np.clip(rng.lognormal(mean=4.0, sigma=0.6), 5, 5000))
            trade_price = mid0 - side * half_spread

            # Permanent impact in the client's favour (adverse to the LP), scaled
            # by toxicity and the regime, realized just after the trade.
            tox = c.informativeness * regime.tox_multiplier * 1e-4 * cfg.start_price
            impact = client_dir * tox * abs(rng.normal(1.0, 0.3))
            impact_level[t + 1:] += impact

            fwd = [diffusion[t + h] + impact_level[t + h] for h in HORIZONS]

            rows.append((day, int(t), c.name, side, volume, trade_price, mid0,
                         2 * half_spread, regime.vol_multiplier, *fwd))

    cols = (["day", "second", "client", "side", "volume", "trade_price", "mid",
             "spread", "regime_vol"] + [f"mid_{h}" for h in HORIZONS])
    df = pd.DataFrame(rows, columns=cols)
    return df


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\n{len(df):,} trades over {df['day'].nunique()} days, "
          f"{df['client'].nunique()} clients")
