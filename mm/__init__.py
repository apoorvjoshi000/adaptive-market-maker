"""Adverse-flow market-making engine.

A research toolkit for liquidity provision: profile client adverse selection,
predict it, externalize the toxic flow, and quote dynamically under inventory
and regime pressure.
"""

from .data import MarketConfig, generate, HORIZONS
from .adversity import (adversity_profile, profile_table, expected_pnl,
                        classify_client, min_half_spread, profitability_table)
from .model import AdversityModel
from .externalization import Externalizer
from .quoting import quote, PARAMS
from . import backtest

__all__ = [
    "MarketConfig", "generate", "HORIZONS",
    "adversity_profile", "profile_table", "expected_pnl", "classify_client",
    "min_half_spread", "profitability_table",
    "AdversityModel", "Externalizer", "quote", "PARAMS", "backtest",
]

__version__ = "0.1.0"
