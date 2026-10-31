"""Forecasting tomorrow's volatility of BTC, ETH and SOL, and tomorrow's return with the same tools, for contrast.

Daily realized variance is the sum of the squared 5-minute log returns of a UTC day. The HAR model of Corsi (2009)
predicts its logarithm from the averages over the last day, week and month:

    log RV[t+1] = b0 + b1 log RV[t] + b2 log RVweek[t] + b3 log RVmonth[t].

The same linear regression on the last day's, week's and month's returns predicts tomorrow's return. Both are
refitted on all earlier days at the start of every month from January 2022 and scored on that month, by the
out-of-sample R^2 against the simplest forecast (the training mean for volatility, zero for returns). An EWMA
forecast (RiskMetrics, lambda 0.94) is a second baseline for volatility.

The forecast is then put to use: a position in the coin scaled to a constant expected volatility, against holding
the coin. Results go to data/volatility.json.

    uv run --with pandas --with pyarrow python articles/crypto_trading/volatility.py
"""

import json

import numpy as np
import pandas as pd
from market import COINS, OUT, candles

FIRST_TEST, LAST_TEST = "2022-01", "2026-10"
WEEK, MONTH = 7, 30  # crypto trades every day
EWMA_LAMBDA = 0.94
TARGET_VOLATILITY = 0.5  # annual, for the volatility-targeted position
MAX_LEVERAGE = 2.0
DAYS_PER_YEAR = 365


def daily(symbol):
    five = candles(symbol)["close"]
    r = np.log(five).diff().dropna()
    variance = r.pow(2).resample("1D").sum()
    count = r.resample("1D").count()
    day_return = r.resample("1D").sum()
    frame = pd.DataFrame({"variance": variance, "return": day_return})
    return frame[count >= 280]  # skip days with missing candles (exchange outages)


def design(d):
    x = pd.DataFrame(index=d.index)
    x["log_rv_day"] = np.log(d["variance"])
    x["log_rv_week"] = np.log(d["variance"].rolling(WEEK).mean())
    x["log_rv_month"] = np.log(d["variance"].rolling(MONTH).mean())
    x["return_day"] = d["return"]
    x["return_week"] = d["return"].rolling(WEEK).sum()
    x["return_month"] = d["return"].rolling(MONTH).sum()
    x["next_log_rv"] = x["log_rv_day"].shift(-1)
    x["next_return"] = d["return"].shift(-1)
    x["next_variance"] = d["variance"].shift(-1)
    return x.dropna()


def fit(x, y):
    a = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coefficients


def predict(coefficients, x):
    return coefficients[0] + x @ coefficients[1:]


def r2(y, forecast, baseline):
    return float(1 - np.sum((y - forecast) ** 2) / np.sum((y - baseline) ** 2))


def ewma_variance(variance):
    out = np.empty(len(variance))
    out[0] = variance[0]
    for t in range(1, len(variance)):
        out[t] = EWMA_LAMBDA * out[t - 1] + (1 - EWMA_LAMBDA) * variance[t]
    return out


def annual_percent(log_variance):
    """Annualized volatility in percent from the log of a daily variance, for the charts."""
    return (100 * np.sqrt(np.exp(log_variance) * DAYS_PER_YEAR)).round(2).tolist()


def drawdown(log_wealth):
    peak = np.maximum.accumulate(log_wealth)
    return float(np.max(1 - np.exp(log_wealth - peak)))


def main():
    vol_features = ["log_rv_day", "log_rv_week", "log_rv_month"]
    return_features = ["return_day", "return_week", "return_month"]
    result = {"first_test": FIRST_TEST, "last_test": LAST_TEST, "coins": {}}
    for symbol in COINS:
        d = daily(symbol)
        x = design(d)
        x["ewma_log_rv"] = np.log(ewma_variance(np.exp(x["log_rv_day"].values)))
        rows = []
        for month in pd.period_range(FIRST_TEST, LAST_TEST, freq="M"):
            start = month.start_time.tz_localize("UTC")
            end = (month + 1).start_time.tz_localize("UTC")
            train = x[x.index < start - pd.Timedelta(days=1)]
            test = x[(x.index >= start) & (x.index < end)]
            if len(test) == 0:
                continue
            vol = fit(train[vol_features].values, train["next_log_rv"].values)
            # the forecast is of the log; the variance itself is larger on average by exp(s^2 / 2)
            residual = train["next_log_rv"].values - predict(vol, train[vol_features].values)
            ret = fit(train[return_features].values, train["next_return"].values)
            rows.append(pd.DataFrame({
                "log_rv": test["next_log_rv"], "har": predict(vol, test[vol_features].values),
                "ewma": test["ewma_log_rv"], "mean_log_rv": train["next_log_rv"].mean(),
                "variance_forecast": np.exp(predict(vol, test[vol_features].values) + residual.var() / 2),
                "ret": test["next_return"], "ret_forecast": predict(ret, test[return_features].values),
                "next_variance": test["next_variance"]}, index=test.index))
        t = pd.concat(rows)
        hit = float((np.sign(t["ret_forecast"]) == np.sign(t["ret"])).mean())
        entry = {"days": len(t), "end": str(t.index[-1].date()),
                 "r2_har": r2(t["log_rv"], t["har"], t["mean_log_rv"]),
                 "r2_ewma": r2(t["log_rv"], t["ewma"], t["mean_log_rv"]),
                 "r2_return": r2(t["ret"], t["ret_forecast"], 0.0),
                 "correlation_vol": float(np.corrcoef(t["log_rv"], t["har"])[0, 1]),
                 "correlation_return": float(np.corrcoef(t["ret"], t["ret_forecast"])[0, 1]),
                 "hit_rate_return": hit}
        entry["scatter"] = {"realized": annual_percent(t["log_rv"]), "har": annual_percent(t["har"]),
                            "ret": (100 * t["ret"]).round(3).tolist(),
                            "ret_forecast": (100 * t["ret_forecast"]).round(4).tolist(),
                            "dates": [d.strftime("%Y-%m-%d") for d in t.index]}
        # a position sized to a constant expected volatility, from the forecast made the day before
        forecast_vol = np.sqrt(t["variance_forecast"] * DAYS_PER_YEAR)
        weight = np.minimum(TARGET_VOLATILITY / forecast_vol, MAX_LEVERAGE)
        hold = t["ret"].values
        targeted = weight.values * hold
        realized_vol_hold = np.std(hold) * np.sqrt(DAYS_PER_YEAR)
        realized_vol_targeted = np.std(targeted) * np.sqrt(DAYS_PER_YEAR)
        entry["targeting"] = {
            "target": TARGET_VOLATILITY, "max_leverage": MAX_LEVERAGE,
            "hold": {"annual_return": float(hold.mean() * DAYS_PER_YEAR), "volatility": float(realized_vol_hold),
                     "sharpe": float(hold.mean() / hold.std() * np.sqrt(DAYS_PER_YEAR)),
                     "max_drawdown": drawdown(np.cumsum(hold))},
            "targeted": {"annual_return": float(targeted.mean() * DAYS_PER_YEAR),
                         "volatility": float(realized_vol_targeted),
                         "sharpe": float(targeted.mean() / targeted.std() * np.sqrt(DAYS_PER_YEAR)),
                         "max_drawdown": drawdown(np.cumsum(targeted))},
            "rolling_dates": [d.strftime("%Y-%m-%d") for d in t.index[29:]],
            "rolling_vol_hold": (100 * pd.Series(hold, index=t.index).rolling(30).std() * np.sqrt(DAYS_PER_YEAR))
            .dropna().round(2).tolist(),
            "rolling_vol_targeted": (100 * pd.Series(targeted, index=t.index).rolling(30).std()
                                     * np.sqrt(DAYS_PER_YEAR)).dropna().round(2).tolist(),
        }
        result["coins"][symbol] = entry
        print(symbol, {k: round(v, 4) for k, v in entry.items() if isinstance(v, float)},
              {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in entry["targeting"].items()
               if k in ("hold", "targeted")}, flush=True)
    (OUT / "volatility.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
