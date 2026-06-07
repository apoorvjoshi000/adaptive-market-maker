# Methodology

All notation is from the liquidity provider's (LP's) perspective. `side` is `+1`
when the LP buys and `-1` when the LP sells.

## 1. Synthetic market

The tape is generated from a microstructure model so the downstream tasks have
real structure to find:

- The mid price is a random walk whose volatility depends on the regime.
- Each trade leaves a permanent price impact in the client's favour (adverse to
  the LP). The impact scales with the client's informativeness, so some clients
  are systematically toxic and others benign.
- Informed clients trade with directional persistence, so recent signed order
  flow predicts the next move.
- At regime-shift points the volatility and the clients' informativeness jump,
  with no marker the models may use.

For each trade the tape stores `side`, `volume`, `trade_price`, the mid at the
trade, the quoted spread, and the mid at horizons `5, 10, ..., 30` seconds.

## 2. Adversity and profitability

A trade is adverse at horizon `h` when `side * volume * (mid_h - trade_price) < 0`.
The adversity profile of a client is the share of its trades that are adverse at
each horizon. The expected per-trade PnL and its horizon average measure whether
the flow is worth internalising.

If we instead quoted symmetrically at `mid +/- delta`, the execution price would
be `trade_price = mid - side * delta`, so the aggregate PnL per trade becomes
`side * volume * (mean_mid - mid) + volume * delta`. The break-even half-spread is

```
delta* = max(0, -A / mean(volume)),   A = mean( side * volume * (mean_mid - mid) ).
```

A client is profitable exactly when the realised half-spread already exceeds its
`delta*`, which ties the two measures together.

## 3. Adverse-selection model

One gradient-boosted classifier per horizon predicts `P(adverse at h)` from
features known at the trade: side, volume, spread, the captured half-spread,
realised volatility, time of day, signed momentum, signed volume, and client
identity. The forward mids define the labels only. The data is split by day into
train / validation / test so the evaluation is strictly out of time.

## 4. Externalization

A trade is externalized (PnL netted to zero) when its predicted adverse
probability exceeds a cutoff `theta`. The cutoff is chosen on the validation
split to maximise kept PnL, then the realised PnL is reported on the test split.
Thresholds can be global or per client; toxic clients warrant a lower cutoff.

## 5. Dynamic quoting

Half-spreads are quoted as multiples of volatility, `delta = m * sigma`, because
the fill model depends only on `delta / sigma`, making `m` the control variable
and removing any dependence on the hidden fill parameters or the units of sigma.

```
base = c_base + c_adv * alpha
skew = (c_inv + c_time * eta) * tanh(inventory / i_ref)
m_bid = base + skew,   m_ask = base - skew,   clipped to [c_min, m_max].
```

The base widens with the adversity score `alpha`; the skew pushes inventory back
to zero and strengthens into the close, where the end-of-day inventory penalty
bites.

## 6. Backtest and scoring

A fill occurs with probability `lambda * exp(-gamma * m)`. A fill earns the
spread plus the adverse drift and moves the inventory. At day end an inventory
penalty `phi * E[I^2] * sigma_day` is charged, with `E[I^2] = E[I]^2 + Var(I)`
so the inventory risk that random fills create is priced. The strategy is scored
by a Sharpe-like ratio (total PnL over daily-PnL volatility), with maximum
drawdown as the tiebreaker. Because the true `(lambda, gamma, phi)` are unknown
and shift between regimes, the gains are tuned to maximise the average score
across several plausible regimes rather than any single one.
