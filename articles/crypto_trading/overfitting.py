"""How a backtest flatters the best of many strategies.

Part one: a grid of the textbook rules (moving-average crossovers, channel breakouts, RSI and Bollinger-band mean
reversion), each with a range of parameters, run on the hourly candles of twelve large coins. Every rule is scored
by its Sharpe ratio after a fee of 5 basis points per side, on 2021-2023 (in sample) and on 2024 to the end of
the data (out of sample).

Part two: strategies that are pure noise. Each holds a random long or short position that flips on average once a
day, on Bitcoin's hourly returns of 2021-2023. The best Sharpe ratio among N of them grows with N, and it is
compared with the expected maximum of Bailey and Lopez de Prado (2014),

    E[max SR] = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e))),

where V is the variance of the trials' Sharpe ratios and gamma is the Euler-Mascheroni constant.
Results go to data/overfitting.json.

    uv run --with pandas --with pyarrow --with scipy python articles/crypto_trading/overfitting.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from market import OUT
from scipy.stats import kurtosis, norm, skew

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
         "LTCUSDT", "TRXUSDT", "DOTUSDT"]
IN_SAMPLE = ("2021-01-01", "2024-01-01")
OUT_OF_SAMPLE = ("2024-01-01", "2026-10-30")
FEE_BPS = 5
HOURS_PER_YEAR = 24 * 365
NOISE_TRIALS = 20000
NOISE_FLIP = 1 / 24  # chance per hour that a random strategy flips its position
EULER_GAMMA = 0.5772156649
SEED = 0


def hourly_close(symbol):
    frame = pd.read_parquet(DATA_DIR / f"{symbol}-1h.parquet")
    return frame["close"]


def crossover(close, fast, slow):
    return np.sign(close.rolling(fast).mean() - close.rolling(slow).mean())


def breakout(close, window):
    upper = close.rolling(window).max().shift(1)
    lower = close.rolling(window).min().shift(1)
    position = pd.Series(np.nan, index=close.index)
    position[close > upper] = 1.0
    position[close < lower] = -1.0
    return position.ffill().fillna(0.0)


def rsi(close, period):
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_reversion(close, period, low):
    value = rsi(close, period)
    position = pd.Series(np.nan, index=close.index)
    position[value < low] = 1.0
    position[value > 100 - low] = -1.0
    position[(value > 45) & (value < 55)] = 0.0
    return position.ffill().fillna(0.0)


def band_reversion(close, window, width):
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std
    position = pd.Series(np.nan, index=close.index)
    position[z < -width] = 1.0
    position[z > width] = -1.0
    position[z.abs() < 0.25] = 0.0
    return position.ffill().fillna(0.0)


def rules():
    out = []
    for fast in (4, 8, 12, 24, 48, 72):
        for slow in (24, 48, 96, 168, 336, 504, 720):
            if fast < slow:
                out.append((f"MA {fast}/{slow}", lambda c, f=fast, s=slow: crossover(c, f, s)))
    for window in (24, 48, 72, 120, 168, 336, 504, 720):
        out.append((f"breakout {window}", lambda c, w=window: breakout(c, w)))
    for period in (6, 14, 24, 48):
        for low in (20, 25, 30):
            out.append((f"RSI {period} {low}/{100 - low}", lambda c, p=period, lo=low: rsi_reversion(c, p, lo)))
    for window in (24, 72, 168):
        for width in (1.5, 2.0, 2.5):
            out.append((f"bands {window} x{width}", lambda c, w=window, k=width: band_reversion(c, w, k)))
    return out


def sharpe(pnl):
    std = pnl.std()
    return float(pnl.mean() / std * np.sqrt(HOURS_PER_YEAR)) if std > 0 else 0.0


def net_pnl(position, close):
    held = position.shift(1).fillna(0.0)  # a position decided at the close of an hour earns the next hour
    r = np.log(close).diff().fillna(0.0)
    return held * r - held.diff().abs().fillna(0.0) * FEE_BPS / 1e4


def deflated_sharpe(pnl, best_sr_per_period, trial_srs_per_period):
    """Probability that the best strategy's true Sharpe ratio is above zero, after N trials (per-period units)."""
    n = len(trial_srs_per_period)
    v = np.var(trial_srs_per_period)
    expected_max = np.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                 + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e)))
    t = len(pnl)
    g3, g4 = skew(pnl), kurtosis(pnl, fisher=False)
    z = (best_sr_per_period - expected_max) * np.sqrt(t - 1)
    z /= np.sqrt(1 - g3 * best_sr_per_period + (g4 - 1) / 4 * best_sr_per_period**2)
    return float(norm.cdf(z)), float(expected_max)


def grid():
    rows = []
    for symbol in COINS:
        close = hourly_close(symbol)
        for name, rule in rules():
            pnl = net_pnl(rule(close), close)
            ins = pnl[(pnl.index >= IN_SAMPLE[0]) & (pnl.index < IN_SAMPLE[1])]
            oos = pnl[(pnl.index >= OUT_OF_SAMPLE[0]) & (pnl.index < OUT_OF_SAMPLE[1])]
            rows.append({"coin": symbol, "rule": name, "in_sample": sharpe(ins), "out_of_sample": sharpe(oos),
                         "pnl_in_sample": ins})
        print(symbol, "done", flush=True)
    return rows


def noise():
    close = hourly_close("BTCUSDT")
    r = np.log(close).diff().dropna()
    r = r[(r.index >= IN_SAMPLE[0]) & (r.index < IN_SAMPLE[1])].values
    rng = np.random.default_rng(SEED)
    srs = np.empty(NOISE_TRIALS)
    for i in range(NOISE_TRIALS):
        flips = rng.random(len(r)) < NOISE_FLIP
        position = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0) * (1 if rng.random() < 0.5 else -1)
        held = np.concatenate([[0.0], position[:-1]])
        pnl = held * r - np.abs(np.diff(np.concatenate([[0.0], held]))) * FEE_BPS / 1e4
        srs[i] = pnl.mean() / pnl.std() * np.sqrt(HOURS_PER_YEAR)
    counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    # the empirical best of N: average of the maximum over disjoint groups of N trials
    best = []
    for n in counts:
        groups = srs[: (NOISE_TRIALS // n) * n].reshape(-1, n)
        best.append(float(groups.max(axis=1).mean()))
    spread = float(np.std(srs))
    mean = float(np.mean(srs))
    theory = []
    for n in counts:
        if n == 1:
            theory.append(mean)
            continue
        theory.append(mean + spread * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                       + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))
    return {"trials": NOISE_TRIALS, "hours": len(r), "mean_sharpe": mean, "std_sharpe": spread,
            "counts": counts, "best_empirical": best, "best_theory": theory,
            "histogram": np.histogram(srs, bins=60)[0].tolist(),
            "edges": np.histogram(srs, bins=60)[1].round(4).tolist()}


def main():
    rows = grid()
    ins = np.array([r["in_sample"] for r in rows])
    oos = np.array([r["out_of_sample"] for r in rows])
    order = np.argsort(-ins)
    best = rows[order[0]]
    hours = len(best["pnl_in_sample"])
    per_period = ins / np.sqrt(HOURS_PER_YEAR)
    probability, expected_max = deflated_sharpe(best["pnl_in_sample"].values, per_period[order[0]], per_period)
    result = {
        "in_sample": IN_SAMPLE, "out_of_sample": OUT_OF_SAMPLE, "fee_bps": FEE_BPS, "rules": len(rules()),
        "coins": COINS, "strategies": len(rows),
        "points": [{"coin": r["coin"], "rule": r["rule"], "in_sample": round(r["in_sample"], 4),
                    "out_of_sample": round(r["out_of_sample"], 4)} for r in rows],
        "rank_correlation": float(pd.Series(ins).corr(pd.Series(oos), method="spearman")),
        "share_positive_in_sample": float((ins > 0).mean()), "share_positive_out_of_sample": float((oos > 0).mean()),
        "top": [{"coin": rows[i]["coin"], "rule": rows[i]["rule"], "in_sample": rows[i]["in_sample"],
                 "out_of_sample": rows[i]["out_of_sample"]} for i in order[:20]],
        "top_decile_out_of_sample_mean": float(oos[order[: len(rows) // 10]].mean()),
        "bottom_decile_out_of_sample_mean": float(oos[order[-(len(rows) // 10):]].mean()),
        "best_deflated": {"hours": hours, "probability_true_sharpe_positive": probability,
                          "expected_max_sharpe_annual": expected_max * np.sqrt(HOURS_PER_YEAR)},
        "noise": noise(),
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("points", "noise", "coins")}, indent=1))
    print("noise", {k: result["noise"][k] for k in ("mean_sharpe", "std_sharpe", "best_empirical", "best_theory")})
    (OUT / "overfitting.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
