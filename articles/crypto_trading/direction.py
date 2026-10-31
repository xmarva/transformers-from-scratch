"""Predicting the direction of the next hour of BTC, ETH and SOL with gradient boosting, then trading on it.

Every hour the model sees what a trader's screen shows at the close of that hour: returns over the last 1 to 168
hours, the coin's and Bitcoin's; volatility from 5-minute returns; volume against its weekly norm; the share of
volume bought by takers (traders who cross the spread); the hour and the weekday. It predicts the probability that
the next hour closes higher.

The test is walk-forward: for every month from January 2022 on, a LightGBM model is trained on all hours before
that month (less a day kept out as a gap) and predicts the month. One model serves all three coins. A strategy
then goes long (+1) when the probability rises above 0.5 + margin, short (-1) when it falls below 0.5 - margin,
and in between keeps the position it has: the wider the margin, the less it trades. It pays a fee on every change
of position. Results go to data/direction.json.

    uv run --with pandas --with pyarrow --with lightgbm --with scikit-learn \
        python articles/crypto_trading/direction.py
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from market import COINS, OUT, candles
from sklearn.metrics import roc_auc_score

FIRST_TEST, LAST_TEST = "2022-01", "2026-10"
GAP = pd.Timedelta(hours=24)  # hours between the end of the training data and the month it predicts
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
FEES_BPS = [0, 1, 2, 5, 10]  # per side: none, a large trader's maker fee, maker and taker on futures, taker on spot
LOOKBACKS = [1, 2, 4, 8, 24, 72, 168]
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 500,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1}
ROUNDS = 300
HOURS_PER_YEAR = 24 * 365


def hourly_bars(symbol):
    five = candles(symbol)
    bars = five.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum",
         "taker_buy_quote_volume": "sum"})
    squared = np.log(five["close"]).diff().pow(2)
    bars["variance_5m"] = squared.resample("1h", label="left", closed="left").sum()
    return bars.dropna()


def features(bars, btc):
    f = pd.DataFrame(index=bars.index)
    log_close = np.log(bars["close"])
    for k in LOOKBACKS:
        f[f"return_{k}h"] = log_close - log_close.shift(k)
    btc_close = np.log(btc["close"]).reindex(bars.index)
    for k in (1, 4, 24):
        f[f"btc_return_{k}h"] = btc_close - btc_close.shift(k)
    f["volatility_24h"] = np.sqrt(bars["variance_5m"].rolling(24).sum())
    f["volatility_168h"] = np.sqrt(bars["variance_5m"].rolling(168).sum() / 7)
    f["volatility_ratio"] = f["volatility_24h"] / f["volatility_168h"]
    log_volume = np.log(bars["quote_volume"].where(bars["quote_volume"] > 0))  # no trades: exchange outages
    f["volume_vs_week"] = log_volume - log_volume.rolling(168).mean()
    taker_share = bars["taker_buy_quote_volume"] / bars["quote_volume"]
    f["taker_buy_share_1h"] = taker_share
    f["taker_buy_share_24h"] = taker_share.rolling(24).mean()
    f["range_1h"] = np.log(bars["high"] / bars["low"])
    f["hour"] = bars.index.hour
    f["weekday"] = bars.index.day_of_week
    f["next_return"] = log_close.shift(-1) - log_close
    return f.dropna()


def positions(p, margin):
    out = np.empty(len(p))
    held = 0.0
    for i in range(len(p)):
        if p[i] > 0.5 + margin:
            held = 1.0
        elif p[i] < 0.5 - margin:
            held = -1.0
        out[i] = held
    return out


def backtest(position, next_return, fee_bps):
    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    gross = position * next_return
    net = gross - turnover * fee_bps / 1e4
    return gross, net, turnover


def summary(pnl):
    mean, std = pnl.mean(), pnl.std()
    return {"annual_return": float(mean * HOURS_PER_YEAR),
            "sharpe": float(mean / std * np.sqrt(HOURS_PER_YEAR)) if std > 0 else 0.0}


def main():
    btc = hourly_bars("BTCUSDT")
    frames = []
    for code, symbol in enumerate(COINS):
        bars = btc if symbol == "BTCUSDT" else hourly_bars(symbol)
        f = features(bars, btc)
        f["coin"] = code
        frames.append(f)
    data = pd.concat(frames).sort_index()
    columns = [c for c in data.columns if c != "next_return"]
    print("hours:", len(data), "features:", len(columns), flush=True)

    predictions = []
    for month in pd.period_range(FIRST_TEST, LAST_TEST, freq="M"):
        start = month.start_time.tz_localize("UTC")
        end = (month + 1).start_time.tz_localize("UTC")
        train = data[data.index < start - GAP]
        test = data[(data.index >= start) & (data.index < end)]
        if len(test) == 0:
            continue
        model = lgb.train(PARAMS, lgb.Dataset(train[columns], (train["next_return"] > 0).astype(int),
                                              categorical_feature=["coin"]), ROUNDS)
        p = model.predict(test[columns])
        predictions.append(pd.DataFrame({"coin": test["coin"], "p_up": p, "next_return": test["next_return"]},
                                        index=test.index))
        hit = ((p > 0.5) == (test["next_return"] > 0)).mean()
        print(month, "train", len(train), "test", len(test), f"hit {hit:.4f}", flush=True)
    importance = dict(zip(columns, model.feature_importance("gain").tolist(), strict=True))
    predicted = pd.concat(predictions)

    result = {"first_test": FIRST_TEST, "last_test": LAST_TEST, "end": str(predicted.index[-1]),
              "rounds": ROUNDS, "params": PARAMS, "features": columns, "importance_last_model": importance,
              "coins": {}}
    for code, symbol in enumerate(COINS):
        part = predicted[predicted["coin"] == code]
        up = (part["next_return"] > 0).values
        p = part["p_up"].values
        r = part["next_return"].values
        entry = {"hours": len(part), "share_up": float(up.mean()), "hit_rate": float(((p > 0.5) == up).mean()),
                 "auc": float(roc_auc_score(up, p)),
                 "monthly_hit_rate": {str(k): float(v) for k, v in
                                      (((part["p_up"] > 0.5) == (part["next_return"] > 0))
                                       .groupby(part.index.strftime("%Y-%m")).mean().items())},
                 "buy_and_hold": summary(r), "strategies": {}}
        for margin in MARGINS:
            position = positions(p, margin)
            traded = position != 0
            row = {"share_in_market": float(traded.mean()),
                   "hit_rate_in_market": float(((position > 0) == up)[traded].mean()),
                   "mean_abs_return_in_market_bps": float(1e4 * np.abs(r[traded]).mean()),
                   "gross_bps_per_hour_in_market": float(1e4 * (position * r)[traded].mean()),
                   "gross_bps_per_hour_when_right": float(1e4 * np.abs(r[traded & ((position > 0) == up)]).mean()),
                   "gross_bps_per_hour_when_wrong": float(1e4 * np.abs(r[traded & ((position > 0) != up)]).mean()),
                   "fees": {}}
            for fee in FEES_BPS:
                gross, net, turnover = backtest(position, r, fee)
                row["position_changes_per_day"] = float(turnover.sum() / (len(r) / 24))
                row["fees"][str(fee)] = summary(net)
            entry["strategies"][str(margin)] = row
        # equity curves of the plain strategy (margin 0), daily, gross and after fees
        position = np.where(p > 0.5, 1.0, -1.0)
        curves = {"buy_and_hold": np.cumsum(r)}
        for fee in (0, 2, 5):
            _, net, _ = backtest(position, r, fee)
            curves[f"fee_{fee}"] = np.cumsum(net)
        daily = pd.DataFrame(curves, index=part.index).resample("1D").last().dropna()
        entry["curves"] = {"dates": [d.strftime("%Y-%m-%d") for d in daily.index],
                           **{k: daily[k].round(5).tolist() for k in curves}}
        result["coins"][symbol] = entry
        print(symbol, f"hit {entry['hit_rate']:.4f} auc {entry['auc']:.4f}",
              {m: (round(s["fees"]["0"]["annual_return"], 3), round(s["fees"]["5"]["annual_return"], 3),
                   round(s["position_changes_per_day"], 2)) for m, s in entry["strategies"].items()}, flush=True)
    (OUT / "direction.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
