"""Download resolved Polymarket markets and their price histories from Polymarket's public APIs.

Two steps, both resumable:

    uv run python articles/polymarket/download.py events   # every closed event with its markets and tags
    uv run python articles/polymarket/download.py events-by-month   # the same from 2026 on, a month per thread
    uv run python articles/polymarket/download.py prices   # price histories of the markets markets.py selects
    uv run python articles/polymarket/download.py flow     # dollars traded in the week before each forecast
    uv run python articles/polymarket/download.py book     # best bid and ask of every open market, now

`events` pages through the Gamma API (gamma-api.polymarket.com/events/keyset) and keeps each closed event whose
volume is at least MIN_VOLUME dollars, with its tags and the fields of its markets that the analysis needs.

`prices` asks the CLOB API (clob.polymarket.com/prices-history) for the price of each selected market's first
outcome token: the whole life of the market at a 12-hour step, and the last 14 days before it closed at an hourly
step (the API returns finer steps only for windows of up to about two weeks). One file per market goes to
DATA_DIR/prices/. With SAMPLE=topic it takes markets.by_topic(), the main sample, instead.

`book` pages through the markets open right now and keeps their best bid and ask, to measure what crossing the spread
costs at each price; it writes DATA_DIR/book.jsonl.gz with the time it was taken.
"""

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-polymarket"))
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MIN_VOLUME = 1000
MARKET_FIELDS = ["id", "question", "slug", "conditionId", "outcomes", "outcomePrices", "clobTokenIds", "volumeNum",
                 "liquidityNum", "createdAt", "startDate", "endDate", "closedTime", "umaResolutionStatus",
                 "category", "negRisk", "groupItemTitle", "fee", "makerBaseFee", "takerBaseFee"]
EVENT_FIELDS = ["id", "slug", "title", "startDate", "endDate", "closedTime", "createdAt", "volume", "negRisk",
                "category", "seriesSlug"]


def get(url, tries=6):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tfs-polymarket-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            status = getattr(error, "code", None)
            if status is not None and status < 500 and status != 429:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url}")


def events(window=None):
    """Every closed event of at least MIN_VOLUME dollars, appended page by page to events.jsonl.gz; with a window
    (first, last end date), only the events ending in it, to events-<first>.jsonl.gz, so that windows can be
    downloaded side by side."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = "events" if window is None else f"events-{window[0]}"
    out = DATA_DIR / f"{name}.jsonl.gz"
    cursor_file = DATA_DIR / f"{name}.cursor"
    cursor = cursor_file.read_text().strip() if cursor_file.exists() else ""
    if cursor == "done":
        return
    pages = 0
    while True:
        url = f"{GAMMA}/events/keyset?closed=true&limit=100&volume_min={MIN_VOLUME}"
        if window is not None:
            url += f"&end_date_min={window[0]}T00:00:00Z&end_date_max={window[1]}T00:00:00Z"
        if cursor:
            url += f"&after_cursor={cursor}"
        page = get(url)
        rows = []
        for event in page.get("events", []):
            row = {k: event.get(k) for k in EVENT_FIELDS}
            row["tags"] = [t.get("label") for t in event.get("tags") or []]
            row["markets"] = [{k: m.get(k) for k in MARKET_FIELDS} for m in event.get("markets") or []]
            rows.append(row)
        with gzip.open(out, "at", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        cursor = page.get("next_cursor") or ""
        pages += 1
        if pages % 50 == 0:
            print(name, pages, "pages", flush=True)
        if not cursor or not rows:
            cursor_file.write_text("done")
            break
        cursor_file.write_text(cursor)
    print(name, "done:", pages, "pages", flush=True)


def events_by_month():
    """The events that end from January 2026 on, one window per month, downloaded side by side."""
    starts = [f"2026-{month:02d}-01" for month in range(1, 13)] + ["2027-01-01", "2032-01-01"]
    windows = list(zip(starts[:-1], starts[1:], strict=True))
    with ThreadPoolExecutor(max_workers=len(windows)) as pool:
        list(pool.map(events, windows))


def history(token, **query):
    params = "&".join(f"{k}={v}" for k, v in query.items())
    points = get(f"{CLOB}/prices-history?market={token}&{params}").get("history", [])
    return [(point["t"], point["p"]) for point in points]


def one(market):
    path = DATA_DIR / "prices" / f"{market['id']}.json.gz"
    if path.exists():
        return
    closed = market["closed_ts"]
    whole = history(market["token"], interval="max", fidelity=720)
    last = history(market["token"], startTs=closed - 14 * 86400, endTs=closed + 3600, fidelity=60)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"id": market["id"], "whole": whole, "last": last}, f)


def prices():
    from markets import by_topic, selected

    (DATA_DIR / "prices").mkdir(parents=True, exist_ok=True)
    chosen = by_topic() if os.environ.get("SAMPLE") == "topic" else selected()
    todo = [m for m in chosen if not (DATA_DIR / "prices" / f"{m['id']}.json.gz").exists()]
    print(len(chosen), "markets selected,", len(todo), "to download", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "6"))) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % 1000 == 0:
                print(done, "downloaded", flush=True)
    print("done", flush=True)


DATA_API = "https://data-api.polymarket.com"
FLOW_HORIZONS = {"1 day": 86400, "1 week": 7 * 86400, "1 month": 30 * 86400, "3 months": 91 * 86400,
                 "6 months": 182 * 86400}


def flow_one(market):
    """Dollars traded in the week before each horizon: a measure of how live the market was when the forecast was
    made, known at the time, unlike the total volume. One page of the latest 100 trades; `capped` marks a page that
    was full, where the true figure is higher (a market with a hundred trades in a week is live in any case)."""
    path = DATA_DIR / "flow" / f"{market['id']}.json"
    if not market.get("condition"):
        return
    out = json.loads(path.read_text()) if path.exists() else {}
    age = market["closed_ts"] - market["created_ts"]
    missing = [name for name, h in FLOW_HORIZONS.items() if name not in out and age >= h
               and (name != "1 day" or age >= FLOW_HORIZONS["1 month"])]  # a day before: only to compare horizons
    if path.exists() and not missing:
        return
    for name in missing:
        end = market["closed_ts"] - FLOW_HORIZONS[name]
        trades = get(f"{DATA_API}/trades?market={market['condition']}&start={end - 7 * 86400}&end={end}&limit=100")
        if not isinstance(trades, list):
            continue
        dollars = sum(float(t.get("size") or 0) * float(t.get("price") or 0) for t in trades)
        out[name] = {"dollars": dollars, "trades": len(trades), "capped": len(trades) >= 100}
    path.write_text(json.dumps(out))


def flow():
    from markets import by_topic, union

    (DATA_DIR / "flow").mkdir(parents=True, exist_ok=True)
    sample = union() if os.environ.get("SAMPLE") == "union" else by_topic()
    # markets open at least a week before the close, the longest-lived first
    chosen = [m for m in sample if m["created_ts"] <= m["closed_ts"] - FLOW_HORIZONS["1 week"]]
    chosen.sort(key=lambda m: m["created_ts"] - m["closed_ts"])
    todo = chosen
    print(len(chosen), "markets to ask (those already asked are skipped quickly)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "12"))) as pool:
        for _ in pool.map(flow_one, todo):
            done += 1
            if done % 2000 == 0:
                print(done, "done", flush=True)
    print("done", flush=True)


def book():
    from datetime import datetime, timezone

    taken = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, cursor = [], ""
    while True:
        url = f"{GAMMA}/markets/keyset?active=true&closed=false&limit=100"
        if cursor:
            url += f"&after_cursor={cursor}"
        page = get(url)
        for m in page.get("markets", []):
            if m.get("bestBid") is None or m.get("bestAsk") is None or not m.get("enableOrderBook"):
                continue
            rows.append({"id": m["id"], "bid": float(m["bestBid"]), "ask": float(m["bestAsk"]),
                         "last": m.get("lastTradePrice"), "volume": m.get("volumeNum"),
                         "liquidity": m.get("liquidityNum"), "tick": m.get("orderPriceMinTickSize"),
                         "end": m.get("endDate"), "question": m.get("question")})
        cursor = page.get("next_cursor") or ""
        if not cursor or not page.get("markets"):
            break
    with gzip.open(DATA_DIR / "book.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"taken": taken}) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(len(rows), "open markets with a book at", taken, flush=True)


if __name__ == "__main__":
    {"events": events, "events-by-month": events_by_month, "prices": prices, "flow": flow, "book": book}[sys.argv[1]]()
