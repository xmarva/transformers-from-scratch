"""Loading the candles of prepare.py, and the facts about returns that every other experiment leans on.

Run as a script, it measures what a model sees in BTC, ETH and SOL returns and writes data/market.json:

- how fat the tails are: the share of hourly moves beyond 4 and 6 standard deviations, against a normal law;
- autocorrelation of returns and of absolute returns, hourly, up to a week of lags;
- by year, how often an hour moves in the same direction as the hour before, and the rank correlation of the two;
- the average absolute move over horizons from 5 minutes to a day, and the hit rate a strategy that trades on
  every bar needs to pay for its costs at each horizon.

    uv run --with pandas --with pyarrow --with scipy python articles/crypto_trading/market.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
OUT = Path(__file__).parent / "data"
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START = "2021-01-01"  # SOL trades from August 2020; every coin is measured over the same period
HORIZONS = {"5 min": "5min", "15 min": "15min", "1 hour": "1h", "4 hours": "4h", "1 day": "1D"}
COSTS_BPS = [2, 4, 10, 20]  # round trip: maker at a high-volume tier, maker and taker on futures, taker on spot
MAX_LAG = 168


def candles(symbol, interval="5m"):
    frame = pd.read_parquet(DATA_DIR / f"{symbol}-{interval}.parquet")
    return frame[frame.index >= START]


def closes(symbol, rule):
    """Close prices of bars of length rule (pandas offset), from the 5-minute candles."""
    five = candles(symbol)["close"]
    if rule == "5min":
        return five
    return five.resample(rule, label="left", closed="left").last().dropna()


def log_returns(prices):
    return np.log(prices).diff().dropna()


def autocorrelation(x, max_lag):
    x = np.asarray(x) - np.mean(x)
    variance = np.dot(x, x)
    return [float(np.dot(x[:-lag], x[lag:]) / variance) for lag in range(1, max_lag + 1)]


def tails(r):
    z = (r - r.mean()) / r.std()
    from scipy.stats import norm

    out = {"kurtosis": float(((z**4).mean()) - 3)}
    for k in (4, 6, 8):
        out[f"beyond_{k}"] = float((np.abs(z) > k).mean())
        out[f"normal_beyond_{k}"] = float(2 * norm.sf(k))
    return out


def main():
    result = {"start": START, "coins": {}}
    for symbol in COINS:
        hourly = log_returns(closes(symbol, "1h"))
        entry = {"hours": len(hourly), "end": str(hourly.index[-1]), "tails": tails(hourly),
                 "acf_returns": autocorrelation(hourly, MAX_LAG),
                 "acf_absolute": autocorrelation(np.abs(hourly), MAX_LAG)}
        # the hourly histogram in units of standard deviation, for the chart
        z = (hourly - hourly.mean()) / hourly.std()
        edges = np.linspace(-20, 20, 161)
        counts, _ = np.histogram(z, bins=edges)
        entry["histogram"] = {"edges": edges.tolist(), "counts": counts.tolist(), "largest": float(np.abs(z).max())}
        moves = {}
        for name, rule in HORIZONS.items():
            r = log_returns(closes(symbol, rule))
            mean_abs_bps = float(1e4 * np.abs(r).mean())
            breakeven = {str(c): 0.5 + c / (2 * mean_abs_bps) for c in COSTS_BPS}
            moves[name] = {"mean_abs_bps": mean_abs_bps, "breakeven_hit_rate": breakeven, "bars": len(r)}
        entry["moves"] = moves
        by_year = {}
        previous = hourly.shift(1)
        for year in sorted(set(hourly.index.year)):
            now, before = hourly[hourly.index.year == year], previous[hourly.index.year == year]
            moved = (now != 0) & (before != 0) & before.notna()
            by_year[str(year)] = {"same_sign": float((np.sign(now[moved]) == np.sign(before[moved])).mean()),
                                  "rank_correlation": float(now[moved].corr(before[moved], method="spearman")),
                                  "hours": int(moved.sum())}
        entry["reversal_by_year"] = by_year
        result["coins"][symbol] = entry
        print(symbol, json.dumps({"tails": entry["tails"], "acf1": entry["acf_returns"][0],
                                  "acf1_abs": entry["acf_absolute"][0],
                                  "same_sign": {y: round(v["same_sign"], 4) for y, v in by_year.items()},
                                  "moves": {k: round(v["mean_abs_bps"], 1) for k, v in moves.items()}}))
    OUT.mkdir(exist_ok=True)
    (OUT / "market.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
