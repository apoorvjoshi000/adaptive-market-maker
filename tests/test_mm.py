"""Test suite for the adverse-flow market-making engine."""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

import mm
from mm import backtest as bt
from mm.data import HORIZONS, MarketConfig
from mm.quoting import C_MIN, quote


@pytest.fixture(scope="module")
def small():
    df = mm.generate(MarketConfig(n_days=12, trades_per_day=800, seed=1))
    model = mm.AdversityModel().fit(df)
    return df, model


# ------------------------------------------------------------------ data
def test_schema_and_size(small):
    df, _ = small
    for col in ["client", "side", "volume", "trade_price", "mid", "spread"]:
        assert col in df.columns
    for h in HORIZONS:
        assert f"mid_{h}" in df.columns
    assert set(df["side"].unique()) <= {-1, 1}
    assert len(df) > 0


def test_toxicity_ordering(small):
    df, _ = small
    # Client F is built to be far more toxic than client A.
    a = mm.expected_pnl(df, "A")["aggregate"]
    f = mm.expected_pnl(df, "F")["aggregate"]
    assert a > f
    assert mm.classify_client(df, "A") == "profitable"
    assert mm.classify_client(df, "F") == "costly"


# ------------------------------------------------------------- adversity
def test_aggregate_is_mean_of_horizons(small):
    df, _ = small
    ep = mm.expected_pnl(df, "C")
    assert abs(ep["aggregate"] - np.mean(ep["per_horizon"])) < 1e-9


def test_profitability_consistent_with_spread(small):
    df, _ = small
    realized = float((df["side"] * (df["mid"] - df["trade_price"])).mean())
    for c in sorted(df["client"].unique()):
        profitable = mm.classify_client(df, c) == "profitable"
        assert profitable == (mm.min_half_spread(df, c) <= realized + 1e-9)


# ----------------------------------------------------------------- model
def test_model_has_skill(small):
    df, model = small
    m = model.metrics(df)
    assert list(m.columns) == ["accuracy", "precision", "recall", "auc", "log_loss"]
    assert m.loc["test", "auc"] > 0.52  # better than chance out of time


def test_predict_returns_probability(small):
    _, model = small
    p = model.predict(client="F", side=1, volume=100, mid=100.0, spread=0.03,
                      trade_price=99.99, horizon=HORIZONS[-1])
    assert 0.0 <= p <= 1.0


# -------------------------------------------------------- externalization
def test_externalization_threshold(small):
    df, model = small
    ext = mm.Externalizer(df, model)
    r = ext.optimal_threshold(HORIZONS[-1])
    assert 0.0 <= r["theta"] <= 1.0
    # At the validation optimum, externalizing is at least as good as keeping all.
    keep_all = ext._kept_pnl(ext.masks["validation"], HORIZONS[-1], 1.0)
    assert r["validation_pnl"] >= keep_all - 1e-6


# ----------------------------------------------------------------- quoting
def test_quote_respects_lower_bound(small):
    sigma = 0.02
    for inv in (-2000, 0, 2000):
        for a in (0.0, 0.5, 1.0):
            for eta in (0.0, 1.0):
                db, da = quote(inv, sigma, a, eta)
                assert db >= C_MIN * sigma - 1e-12
                assert da >= C_MIN * sigma - 1e-12


def test_quote_skews_to_flatten_inventory():
    sigma = 0.02
    db_long, da_long = quote(2000, sigma, 0.3, 0.5)
    assert db_long > da_long          # long: widen bid, tighten ask
    db_short, da_short = quote(-2000, sigma, 0.3, 0.5)
    assert da_short > db_short        # short: widen ask, tighten bid


# ---------------------------------------------------------------- backtest
def test_dynamic_beats_tight_fixed_in_stress(small):
    df, model = small
    days = bt.prepare(df, model)
    lam, gam, phi = bt.ROBUST_SCENARIOS[1]  # the stress regime
    dyn = bt.simulate(days, mm.PARAMS, lam, gam, phi)
    tight = bt.simulate(days, bt.fixed_spread_params(0.5), lam, gam, phi)
    assert dyn["score"] > tight["score"]
    assert dyn["max_drawdown"] <= tight["max_drawdown"] + 1e-9
