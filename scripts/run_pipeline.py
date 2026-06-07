"""End-to-end pipeline: data -> adversity -> model -> externalization -> quoting.

Runs every stage, saves figures to figures/, and prints a results summary.
Run from the project root:

    python scripts/run_pipeline.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mm
from mm import backtest as bt
from mm.data import HORIZONS
from mm.quoting import quote

FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})
PALETTE = plt.cm.viridis(np.linspace(0.1, 0.9, 6))


def fig_adversity(df):
    table = mm.profile_table(df)
    plt.figure(figsize=(7, 4.5))
    for i, c in enumerate(table.index):
        plt.plot(HORIZONS, table.loc[c].values, marker="o", color=PALETTE[i], label=c)
    plt.xlabel("horizon (seconds)"); plt.ylabel("adverse trades (%)")
    plt.title("Client adversity profiles")
    plt.legend(title="client", ncol=2)
    plt.tight_layout(); plt.savefig(f"{FIG}/01_adversity_profiles.png"); plt.close()
    return table


def fig_profitability(df):
    tbl = mm.profitability_table(df)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in tbl["aggregate"]]
    ax[0].bar(tbl["client"], tbl["aggregate"], color=colors)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("Aggregate PnL per trade")
    ax[0].set_xlabel("client"); ax[0].set_ylabel("PnL per trade")
    ax[1].bar(tbl["client"], tbl["min_half_spread"], color="#1f77b4")
    ax[1].set_title("Minimum break-even half-spread")
    ax[1].set_xlabel("client"); ax[1].set_ylabel("half-spread (price)")
    plt.tight_layout(); plt.savefig(f"{FIG}/02_client_profitability.png"); plt.close()
    return tbl


def fig_model(df, model):
    df2, feat, _ = mm.model.build_features(df)
    masks = mm.model._day_split(df2)
    test = masks["test"]
    from sklearn.metrics import roc_auc_score
    aucs = []
    for h in HORIZONS:
        y = mm.adversity.adverse_mask(df2, h).to_numpy()[test]
        p = model.predict_proba_frame(feat[test], h)
        aucs.append(roc_auc_score(y, p))
    plt.figure(figsize=(7, 4.5))
    plt.bar([str(h) for h in HORIZONS], aucs, color="#6a3d9a")
    plt.axhline(0.5, color="k", ls="--", lw=0.8, label="no skill")
    plt.ylim(0.5, max(aucs) + 0.03)
    plt.xlabel("horizon (seconds)"); plt.ylabel("test ROC AUC")
    plt.title("Adverse-selection model: out-of-time discrimination")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/03_model_auc.png"); plt.close()


def fig_externalization(ext):
    plt.figure(figsize=(7.5, 4.5))
    for i, h in enumerate(HORIZONS):
        curve = ext.pnl_curve(h, "validation")
        plt.plot(ext.THETA_GRID if hasattr(ext, "THETA_GRID") else
                 np.linspace(0, 1, len(curve)), curve, color=PALETTE[i], label=f"h={h}")
    plt.xlabel("externalization cutoff θ"); plt.ylabel("validation PnL (kept flow)")
    plt.title("PnL vs externalization threshold")
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(f"{FIG}/04_pnl_vs_theta.png"); plt.close()


def fig_quote_surface():
    invs = np.linspace(-3000, 3000, 60)
    alphas = np.linspace(0.0, 1.0, 60)
    sigma = 0.03
    bid = np.zeros((len(alphas), len(invs)))
    skew = np.zeros_like(bid)
    for i, a in enumerate(alphas):
        for j, q in enumerate(invs):
            db, da = quote(q, sigma, a, eta=0.5)
            bid[i, j] = db
            skew[i, j] = db - da
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = ax[0].pcolormesh(invs, alphas, bid, shading="auto", cmap="magma")
    ax[0].set_title("Bid half-spread δ_bid"); ax[0].set_xlabel("inventory")
    ax[0].set_ylabel("adversity α"); fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].pcolormesh(invs, alphas, skew, shading="auto", cmap="coolwarm")
    ax[1].set_title("Quote skew δ_bid - δ_ask"); ax[1].set_xlabel("inventory")
    ax[1].set_ylabel("adversity α"); fig.colorbar(im1, ax=ax[1])
    plt.tight_layout(); plt.savefig(f"{FIG}/05_quote_surface.png"); plt.close()


def fig_backtest(days, params):
    cmp = bt.compare(days, params)
    regimes = cmp["regime"].unique()
    strategies = ["fixed m=0.5", "fixed m=1.0", "fixed m=2.0", "dynamic"]
    width = 0.2
    plt.figure(figsize=(9, 4.6))
    x = np.arange(len(regimes))
    for i, strat in enumerate(strategies):
        scores = [cmp[(cmp.regime == r) & (cmp.strategy == strat)]["score"].values[0]
                  for r in regimes]
        color = "#d62728" if strat == "dynamic" else None
        plt.bar(x + i * width, scores, width, label=strat, color=color)
    plt.xticks(x + 1.5 * width, [f"regime {i+1}" for i in range(len(regimes))])
    plt.ylabel("Sharpe-like score"); plt.title("Risk-adjusted score by strategy and regime")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/06_backtest_scores.png"); plt.close()
    return cmp


def main():
    print("Generating synthetic market and flow ...")
    df = mm.generate()
    print(f"  {len(df):,} trades, {df['day'].nunique()} days, "
          f"{df['client'].nunique()} clients\n")

    print("Task 1/2 - adversity and profitability")
    prof = fig_adversity(df)
    pt = fig_profitability(df)
    print(pt.round(4).to_string(index=False), "\n")

    print("Task 3 - adverse-selection model")
    model = mm.AdversityModel().fit(df)
    print(model.metrics(df).round(4).to_string())
    fig_model(df, model)
    print()

    print("Task 4 - optimal externalization")
    ext = mm.Externalizer(df, model)
    ext.THETA_GRID = mm.externalization.THETA_GRID
    fig_externalization(ext)
    for h in (HORIZONS[0], HORIZONS[-1]):
        u = ext.uplift(h)
        print(f"  h={h:2d}: theta*={u['theta']:.3f}  "
              f"test PnL with={u['with_externalization']:.0f}  "
              f"without={u['without']:.0f}")
    print()

    print("Task 5 - dynamic quoting and backtest")
    days = bt.prepare(df, model)
    fig_quote_surface()
    cmp = fig_backtest(days, mm.PARAMS)
    print(cmp.round(2).to_string(index=False))

    print(f"\nFigures written to {FIG}/")


if __name__ == "__main__":
    main()
