"""The funding rate of the BTC and ETH perpetual futures on Binance, and what it paid to the cash-and-carry trade.

A perpetual future has no expiry; to keep its price near the spot price, the side that is crowded pays the other
every 8 hours. When the rate is positive, longs pay shorts. Holding the coin and shorting the same amount of the
future is market-neutral and collects the rate: the cash-and-carry trade. Its yield by month and by year goes to
data/funding.json.

    uv run --with pandas --with pyarrow python articles/crypto_trading/funding.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from market import OUT

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
COINS = ["BTCUSDT", "ETHUSDT"]
ENTRY_EXIT_BPS = 2 * (10 + 5)  # once in and once out: taker on spot and on the future, each way


def main():
    result = {"coins": {}}
    for symbol in COINS:
        rate = pd.read_parquet(DATA_DIR / f"{symbol}-funding.parquet")["funding_rate"]
        monthly = rate.groupby(rate.index.strftime("%Y-%m"))
        yearly = rate.groupby(rate.index.year)
        entry = {
            "payments": len(rate), "first": str(rate.index[0]), "last": str(rate.index[-1]),
            "mean_rate": float(rate.mean()), "share_negative": float((rate < 0).mean()),
            # the sum of a month's payments, scaled to a year: the yield of the carry trade in that month
            "monthly_annualized": {k: float(v.sum() * 12) for k, v in monthly},
            "monthly_share_negative": {k: float((v < 0).mean()) for k, v in monthly},
            "yearly": {str(k): {"carry": float(v.sum()), "payments": len(v), "share_negative": float((v < 0).mean()),
                                "annualized": float(v.mean() * 3 * 365)} for k, v in yearly},
            "cumulative": {"dates": [d.strftime("%Y-%m-%d") for d in rate.resample("1D").sum().index],
                           "carry": np.cumsum(rate.resample("1D").sum().values).round(5).tolist()},
            "entry_exit_cost": ENTRY_EXIT_BPS / 1e4,
        }
        result["coins"][symbol] = entry
        print(symbol, entry["payments"], "payments; share negative", round(entry["share_negative"], 3),
              {k: round(v["annualized"], 4) for k, v in entry["yearly"].items()}, flush=True)
    (OUT / "funding.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
