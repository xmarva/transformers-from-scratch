"""Alpha Arena, Season 1: six LLMs trading $10,000 each on Hyperliquid, 18 October to 3 November 2025.

Nof1's public API (nof1.ai/api/account-totals) gave every model's equity hour by hour, and that of a Bitcoin
buy-and-hold account Nof1 ran alongside them. The API is gone; the Internet Archive kept a copy made after the
season ended. This script reads that copy, keeps the hourly equity up to the official close (5 pm New York time on
3 November, 22:00 UTC) and writes data/arena.json: the curves, each model's return, its largest drawdown, the
volatility of its hourly returns and their correlations.

    python articles/crypto_trading/arena.py
"""

import gzip
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).parent / "data"
SOURCE = "https://web.archive.org/web/20251112224724id_/https://nof1.ai/api/account-totals"
CLOSE = datetime(2025, 11, 3, 22, 0, tzinfo=timezone.utc).timestamp()
NAMES = {"qwen3-max": "Qwen3-Max", "deepseek-chat-v3.1": "DeepSeek V3.1", "claude-sonnet-4-5": "Claude Sonnet 4.5",
         "grok-4": "Grok 4", "gemini-2.5-pro": "Gemini 2.5 Pro", "gpt-5": "GPT-5",
         "buynhold_btc": "Bitcoin, bought and held"}
START_EQUITY = 10_000


def download():
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        raw = response.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)["accountTotals"]


def hourly(rows, model):
    """The last record of every hour of the season, as (hour marker, time, equity)."""
    last = {}
    for row in rows:
        if row["model_id"] != model or row["timestamp"] > CLOSE:
            continue
        marker = row["since_inception_hourly_marker"]
        if marker not in last or row["timestamp"] > last[marker]["timestamp"]:
            last[marker] = row
    return [(m, last[m]["timestamp"], last[m]["dollar_equity"]) for m in sorted(last)]


def max_drawdown(equity):
    peak, worst = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, 1 - value / peak)
    return worst


def returns(equity):
    return [math.log(b / a) for a, b in zip(equity, equity[1:], strict=False)]


def correlation(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def main():
    rows = download()
    series = {model: hourly(rows, model) for model in NAMES}
    # a common grid: the hours every account has a record for
    common = set.intersection(*(set(m for m, _, _ in s) for s in series.values()))
    result = {"source": SOURCE, "close_utc": "2025-11-03T22:00:00Z", "hours": len(common), "models": {}}
    hourly_returns = {}
    for model, s in series.items():
        kept = [(m, t, e) for m, t, e in s if m in common]
        equity = [e for _, _, e in kept]
        hourly_returns[model] = returns(equity)
        result["models"][model] = {
            "name": NAMES[model],
            "first": datetime.fromtimestamp(kept[0][1], timezone.utc).isoformat(timespec="minutes"),
            "last": datetime.fromtimestamp(kept[-1][1], timezone.utc).isoformat(timespec="minutes"),
            # the result is the last record before the close, which may fall outside the common grid
            "final_equity": s[-1][2], "final_time": datetime.fromtimestamp(s[-1][1], timezone.utc)
            .isoformat(timespec="minutes"), "return": s[-1][2] / START_EQUITY - 1, "peak": max(equity),
            "max_drawdown": max_drawdown(equity),
            "hourly_volatility": float(sum(r * r for r in hourly_returns[model]) / len(hourly_returns[model])) ** 0.5,
            "times": [datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="minutes") for _, t, _ in kept],
            "equity": [round(e, 2) for e in equity],
        }
    models = list(NAMES)
    result["correlations"] = {a: {b: correlation(hourly_returns[a], hourly_returns[b]) for b in models}
                              for a in models}
    for model in models:
        m = result["models"][model]
        print(f"{m['name']:26s} {m['final_equity']:10.2f} {100 * m['return']:+7.1f}%  drawdown "
              f"{100 * m['max_drawdown']:5.1f}%  hourly sd {100 * m['hourly_volatility']:.2f}%  {m['first']} "
              f"{m['final_time']}")
    OUT.mkdir(exist_ok=True)
    (OUT / "arena.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
