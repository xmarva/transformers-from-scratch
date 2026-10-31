"""Time-series foundation models on crypto: Chronos-2 and Kronos, zero-shot, on the hours and days of 2026.

Chronos-2 (Amazon, October 2025) is a general forecaster pretrained on many kinds of series; Kronos-small
(August 2025) was pretrained on candles from 45 exchanges and reads open, high, low, close and volume. Neither
is fine-tuned here. Both forecast the next hour of BTC, ETH and SOL for every hour of 2026 up to 29 October
(Kronos for every fourth hour: it samples its forecast token by token and is slower), from the 512 hours before:
Chronos-2 from the hourly log returns, Kronos from the candles themselves. Chronos-2 also forecasts tomorrow's log
realized variance from the 512 days before, the task of the HAR model in volatility.py.

Three steps: `inputs` writes the series the models need from the candles of prepare.py (locally), `run` runs the
models (on a GPU; kaggle.py runs it on Kaggle) and writes data/foundation_raw.json, and `summarize` scores the
forecasts against the realized values and writes data/foundation.json.

    uv run --with pandas --with pyarrow python articles/crypto_trading/foundation.py inputs
    python articles/crypto_trading/foundation.py run
    uv run --with pandas --with pyarrow python articles/crypto_trading/foundation.py summarize
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
OUT = Path(os.environ.get("OUT", HERE / "data"))
INPUTS = Path(os.environ.get("INPUTS", DATA_DIR / "foundation_inputs.json"))
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TEST_START, TEST_END = "2026-01-01", "2026-10-29"
CONTEXT = 512
KRONOS_EVERY = 4
KRONOS_SAMPLES = 4
SMOKE = bool(os.environ.get("SMOKE"))  # a few forecasts on the CPU, to check the code on a laptop


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_inputs():
    import numpy as np
    import pandas as pd

    out = {"coins": {}}
    for symbol in COINS:
        five = pd.read_parquet(DATA_DIR / f"{symbol}-5m.parquet")
        bars = five.resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
             "quote_volume": "sum"}).dropna()
        bars = bars[bars.index >= pd.Timestamp(TEST_START, tz="UTC") - pd.Timedelta(hours=CONTEXT + 24)]
        r = np.log(five["close"]).diff().dropna()
        variance = r.pow(2).resample("1D").sum()
        counts = r.resample("1D").count()
        log_rv = np.log(variance[counts >= 280])
        out["coins"][symbol] = {
            "hours": [t.isoformat() for t in bars.index],
            **{k: bars[k].round(8).tolist() for k in ("open", "high", "low", "close", "volume", "quote_volume")},
            "days": [t.strftime("%Y-%m-%d") for t in log_rv.index], "log_rv": log_rv.round(6).tolist(),
        }
        print(symbol, len(bars), "hours,", len(log_rv), "days")
    INPUTS.write_text(json.dumps(out))
    print(INPUTS, INPUTS.stat().st_size // 1000, "kB")


def chronos_forecasts(pipeline, contexts, batch=256):
    """Median one-step forecasts for a list of 1-D series."""
    import torch

    out = []
    for i in range(0, len(contexts), batch):
        chunk = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + batch]]
        quantiles, _ = pipeline.predict_quantiles(chunk, prediction_length=1, quantile_levels=[0.5])
        for q in quantiles:
            out.append(float(torch.as_tensor(q).flatten()[0]))
    return out


def run_chronos(inputs, result):
    import numpy as np
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device())
    entry = {"model": "amazon/chronos-2", "coins": {}}
    for symbol, series in inputs["coins"].items():
        hours = series["hours"]
        log_close = np.log(np.array(series["close"]))
        returns = 100 * np.diff(log_close)  # returns[i] is the move from hour i to hour i + 1, in percent
        targets = [i for i in range(len(hours) - 1) if TEST_START <= hours[i + 1][:10] <= TEST_END
                   and i >= CONTEXT]
        if SMOKE:
            targets = targets[:8]
        contexts = [returns[i - CONTEXT:i] for i in targets]
        started = time.time()
        forecast = chronos_forecasts(pipeline, contexts)
        days = series["days"]
        log_rv = np.array(series["log_rv"])
        day_targets = [i for i in range(CONTEXT, len(days)) if TEST_START <= days[i] <= TEST_END]
        if SMOKE:
            day_targets = day_targets[:4]
        vol_forecast = chronos_forecasts(pipeline, [log_rv[i - CONTEXT:i] for i in day_targets])
        entry["coins"][symbol] = {
            "hours": [hours[i + 1] for i in targets], "forecast": forecast,
            "realized": [float(returns[i]) for i in targets],
            "days": [days[i] for i in day_targets], "log_rv_forecast": vol_forecast,
            "log_rv": [float(log_rv[i]) for i in day_targets],
            "log_rv_context_mean": [float(log_rv[i - CONTEXT:i].mean()) for i in day_targets],
            "seconds": time.time() - started}
        print("chronos-2", symbol, len(targets), "hours,", len(day_targets), "days,",
              f"{time.time() - started:.0f} s", flush=True)
    result["chronos"] = entry


def run_kronos(inputs, result):
    import numpy as np
    import pandas as pd
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0" if device() == "cuda" else "cpu",
                                max_context=CONTEXT)
    entry = {"model": "NeoQuasar/Kronos-small", "samples": KRONOS_SAMPLES, "coins": {}}
    for symbol, series in inputs["coins"].items():
        frame = pd.DataFrame({k: series[k] for k in ("open", "high", "low", "close", "volume")})
        frame["amount"] = series["quote_volume"]
        stamps = pd.Series(pd.to_datetime(series["hours"]))
        hours = series["hours"]
        targets = [i for i in range(CONTEXT, len(hours) - 1) if TEST_START <= hours[i + 1][:10] <= TEST_END]
        targets = targets[::KRONOS_EVERY][:4] if SMOKE else targets[::KRONOS_EVERY]
        started, forecast = time.time(), []
        for start in range(0, len(targets), 64):
            chunk = targets[start:start + 64]
            frames = [frame.iloc[i - CONTEXT + 1:i + 1].reset_index(drop=True) for i in chunk]
            x_stamps = [stamps.iloc[i - CONTEXT + 1:i + 1].reset_index(drop=True) for i in chunk]
            y_stamps = [stamps.iloc[i + 1:i + 2].reset_index(drop=True) for i in chunk]
            predicted = predictor.predict_batch(df_list=frames, x_timestamp_list=x_stamps, y_timestamp_list=y_stamps,
                                                pred_len=1, T=1.0, top_p=0.9, sample_count=KRONOS_SAMPLES,
                                                verbose=False)
            for i, p in zip(chunk, predicted, strict=True):
                forecast.append(float(100 * np.log(p["close"].iloc[0] / series["close"][i])))
        entry["coins"][symbol] = {
            "hours": [hours[i + 1] for i in targets], "forecast": forecast,
            "realized": [float(100 * np.log(series["close"][i + 1] / series["close"][i])) for i in targets],
            "seconds": time.time() - started}
        print("kronos-small", symbol, len(targets), "hours,", f"{time.time() - started:.0f} s", flush=True)
    result["kronos"] = entry


def run():
    inputs = json.loads(INPUTS.read_text())
    path = OUT / ("foundation_smoke.json" if SMOKE else "foundation_raw.json")
    result = json.loads(path.read_text()) if path.exists() else {}
    only = os.environ.get("ONLY")
    for name, step in (("chronos", run_chronos), ("kronos", run_kronos)):
        if name in result or (only and name != only):
            continue
        try:
            step(inputs, result)
        except Exception as error:  # one model failing should not cost the other
            import traceback

            traceback.print_exc()
            result[name] = {"error": repr(error)[:600]}
        path.write_text(json.dumps(result))


def summarize():
    import numpy as np

    raw = json.loads((OUT / "foundation_raw.json").read_text())
    volatility = json.loads((OUT / "volatility.json").read_text())
    out = {"test": [TEST_START, TEST_END], "context": CONTEXT, "models": {}}
    for name, entry in raw.items():
        if "error" in entry:
            out["models"][name] = {"error": entry["error"]}
            continue
        coins = {}
        for symbol, c in entry["coins"].items():
            f, y = np.array(c["forecast"]), np.array(c["realized"])
            row = {"hours": len(y), "hit_rate": float((np.sign(f) == np.sign(y)).mean()),
                   "share_forecast_up": float((f > 0).mean()),
                   "r2_vs_zero": float(1 - np.sum((y - f) ** 2) / np.sum(y**2)),
                   "correlation": float(np.corrcoef(f, y)[0, 1]),
                   "mean_abs_forecast": float(np.abs(f).mean()), "mean_abs_realized": float(np.abs(y).mean())}
            if "log_rv_forecast" in c:
                # the same days forecast by the HAR model of volatility.py (stored as annualized percent)
                har = volatility["coins"][symbol]["scatter"]
                har_by_day = dict(zip(har["dates"], har["har"], strict=True))
                kept = [i for i, day in enumerate(c["days"]) if day in har_by_day]
                v = np.array(c["log_rv_forecast"])[kept]
                actual = np.array(c["log_rv"])[kept]
                baseline = np.array(c["log_rv_context_mean"])[kept]
                har_log = np.log((np.array([har_by_day[c["days"][i]] for i in kept]) / 100) ** 2 / 365)
                row["volatility_days"] = len(kept)
                row["r2_log_rv"] = float(1 - np.sum((actual - v) ** 2) / np.sum((actual - baseline) ** 2))
                row["r2_log_rv_har"] = float(1 - np.sum((actual - har_log) ** 2) / np.sum((actual - baseline) ** 2))
            coins[symbol] = row
        out["models"][name] = {"model": entry["model"], "coins": coins}
        print(name, {s: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)} for s, r in coins.items()})
    (OUT / "foundation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"inputs": build_inputs, "run": run, "summarize": summarize}[sys.argv[1]]()
