"""Dynamic inventory-aware quoting.

The bid and ask half-spreads are quoted as multiples of the current volatility:

    delta_bid = m_bid * sigma,   delta_ask = m_ask * sigma.

Quoting in multiples of sigma is deliberate. The fill probability depends only
on the ratio delta / sigma, so the multiple m is the real control variable and
it is independent of the hidden fill parameters and of the units of sigma. It
also satisfies the lower bound c_min * sigma <= delta automatically once
m >= c_min.

The multiples combine three observable effects:

    base = c_base + c_adv * alpha
        a floor that widens with the model's adversity score, charging more for
        toxic flow and filling it less often;

    skew = (c_inv + c_time * eta) * tanh(inventory / i_ref)
        a tilt that pushes inventory back to zero (when long we widen the bid and
        tighten the ask) and strengthens as the day closes, where the end-of-day
        inventory penalty bites;

    m_bid = base + skew,   m_ask = base - skew,  clipped to [c_min, m_max].
"""

from __future__ import annotations

import math

C_MIN = 0.5
M_MAX = 16.0

# Tuned on the synthetic data used as a proxy (see backtest.tune): a moderately
# wide, volatility-scaled base that widens on toxic flow, plus a firm inventory
# skew that ramps into the close.
PARAMS = {
    "c_base": 1.6,
    "c_adv": 3.0,
    "c_inv": 1.8,
    "c_time": 2.4,
    "i_ref": 800.0,
}


def quote(inventory: float, sigma: float, alpha: float, eta: float,
          params: dict = None) -> "tuple[float, float]":
    """Return (delta_bid, delta_ask) for the current state."""
    p = params or PARAMS
    base = p["c_base"] + p["c_adv"] * alpha
    skew = (p["c_inv"] + p["c_time"] * eta) * math.tanh(inventory / p["i_ref"])
    m_bid = min(max(base + skew, C_MIN), M_MAX)
    m_ask = min(max(base - skew, C_MIN), M_MAX)
    return float(m_bid * sigma), float(m_ask * sigma)
