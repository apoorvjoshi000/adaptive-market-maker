"""Generate the synthetic trade tape and save it to data/trades.csv."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm.data import MarketConfig, generate


def main():
    df = generate(MarketConfig())
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "trades.csv")
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} trades to {out}")


if __name__ == "__main__":
    main()
