"""Market data of the experiments: Binance candles and funding rates, from Binance's public data archive.

- 5-minute spot candles of BTC, ETH and SOL against USDT, from January 2020 (SOL from its listing in August 2020);
- hourly spot candles of the twelve largest coins that have traded on Binance since 2021, for the backtest grid;
- the funding rates of the BTC and ETH perpetual futures (USDT-margined), every 8 hours, from January 2020;
- daily spot candles of BTC from September 2017, its first full month on Binance, for the questions to LLMs.

The archive has a file per month and, for the current month, a file per day. Everything ends on END and goes to
DATA_DIR (default ~/.cache/tfs-crypto), outside the repository, as one parquet file per series:

    uv run --with pandas --with pyarrow python articles/crypto_trading/prepare.py
"""

import io
import json
import os
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
ARCHIVE = "https://data.binance.vision/data"
END = date(2026, 10, 29)  # the last day of data
FIVE_MINUTE = {"BTCUSDT": "2020-01", "ETHUSDT": "2020-01", "SOLUSDT": "2020-08"}
HOURLY = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
          "LTCUSDT", "TRXUSDT", "DOTUSDT"]
HOURLY_START = "2021-01"
FUNDING = {"BTCUSDT": "2020-01", "ETHUSDT": "2020-01"}
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start):
    year, month = map(int, start.split("-"))
    out = []
    while (year, month) < (END.year, END.month):
        out.append(f"{year}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def days_of_last_month():
    day = date(END.year, END.month, 1)
    out = []
    while day <= END:
        out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def fetch_csv(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open(archive.namelist()[0]) as f:
        return f.read().decode()


def read_klines(text):
    first = text.split("\n", 1)[0]
    has_header = not first[0].isdigit()
    frame = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
    frame.columns = KLINE_COLUMNS
    # from 2025 the spot archive gives times in microseconds, before that in milliseconds
    microseconds = frame["open_time"].where(frame["open_time"] > 10**14, frame["open_time"] * 1000)
    frame.index = pd.DatetimeIndex(pd.to_datetime(microseconds, unit="us", utc=True), name="time")
    return frame[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_quote_volume"]]


def klines(symbol, interval, start):
    path = DATA_DIR / f"{symbol}-{interval}.parquet"
    if path.exists():
        return
    parts = []
    for month in months(start):
        parts.append(read_klines(fetch_csv(f"{ARCHIVE}/spot/monthly/klines/{symbol}/{interval}/"
                                           f"{symbol}-{interval}-{month}.zip")))
    for day in days_of_last_month():
        parts.append(read_klines(fetch_csv(f"{ARCHIVE}/spot/daily/klines/{symbol}/{interval}/"
                                           f"{symbol}-{interval}-{day}.zip")))
    frame = pd.concat(parts).sort_index()
    frame = frame[~frame.index.duplicated()]
    frame.to_parquet(path)
    print(symbol, interval, len(frame), "candles,", frame.index[0], "to", frame.index[-1], flush=True)


def funding(symbol, start):
    path = DATA_DIR / f"{symbol}-funding.parquet"
    if path.exists():
        return
    parts = []
    for month in months(start):
        text = fetch_csv(f"{ARCHIVE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip")
        frame = pd.read_csv(io.StringIO(text))
        parts.append(pd.Series(frame["last_funding_rate"].values,
                               index=pd.to_datetime(frame["calc_time"], unit="ms", utc=True)))
    # the archive has no daily funding files, so the current month comes from the public API
    first = pd.Timestamp(date(END.year, END.month, 1), tz="UTC")
    last = pd.Timestamp(END + timedelta(days=1), tz="UTC")
    url = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={int(first.timestamp() * 1000)}"
           f"&endTime={int(last.timestamp() * 1000) - 1}&limit=1000")
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.load(response)
    parts.append(pd.Series([float(r["fundingRate"]) for r in rows],
                           index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True)))
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated()]
    # funding is settled at 00:00, 08:00 and 16:00 UTC; the recorded times are a few milliseconds off
    series.index = series.index.round("min")
    series.rename("funding_rate").to_frame().to_parquet(path)
    print(symbol, "funding", len(series), "payments,", series.index[0], "to", series.index[-1], flush=True)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, start in FIVE_MINUTE.items():
        klines(symbol, "5m", start)
    for symbol in HOURLY:
        klines(symbol, "1h", HOURLY_START)
    for symbol, start in FUNDING.items():
        funding(symbol, start)
    klines("BTCUSDT", "1d", "2017-09")


if __name__ == "__main__":
    main()
