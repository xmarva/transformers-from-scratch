"""How well Polymarket's prices forecast: calibration, Brier scores, returns by price, and how they depend on time,
volume and topic. Reads the markets chosen by markets.py and the price histories from download.py; writes
data/analysis.json.

    uv run --with numpy --with pandas --with pyarrow python articles/polymarket/analysis.py

The forecast of a market at horizon h is the price of its first outcome token h before the market closed (the last
price at or before that moment), taken from the hourly series for horizons of up to 14 days and from the 12-hour
series beyond. A market enters a horizon only if it was already open then and at least LIVE_DOLLARS had been traded
in it in the week before (download.py flow): a market nobody trades has a price, the middle of an empty book, but no
forecast. Both conditions are known at the time, unlike the market's total volume, which also counts the trading
the outcome itself set off.

The main sample is markets.union(): the markets drawn by topic (markets.by_topic) and those drawn by volume
(markets.selected). Pooled statistics weight each market by 1 / its chance of being in either draw, so they
describe all resolved markets, and their uncertainty comes from a bootstrap over events, since the markets of one
event (the candidates of one election, say) are not independent. What grouping markets by their total volume does to
the results is shown on the draw by topic alone (trap).
"""

import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
from markets import DATA_DIR, TOPIC_NAMES, by_topic, selected, union

OUT = Path(__file__).parent / "data"
HOUR, DAY = 3600, 86400
HORIZONS = {"1 day": DAY, "1 week": 7 * DAY, "1 month": 30 * DAY, "3 months": 91 * DAY, "6 months": 182 * DAY}
EDGES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0000001])
FINE = np.linspace(0, 1.0000001, 21)
BOOT = int(os.environ.get("BOOT", "400"))
LIVE_DOLLARS = float(os.environ.get("LIVE_DOLLARS", "1000"))  # traded in the week before a forecast
RNG = np.random.default_rng(20261031)


def price_at(series, moment, stale):
    """The last price at or before `moment`, if it is no older than `stale` seconds."""
    best = None
    for t, p in series:
        if t > moment:
            break
        best = (t, p)
    if best is None or moment - best[0] > stale:
        return None
    return best[1]


def load(markets):
    rows = []
    for m in markets:
        path = DATA_DIR / "prices" / f"{m['id']}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            series = json.load(f)
        flow_path = DATA_DIR / "flow" / f"{m['id']}.json"
        flow = json.loads(flow_path.read_text()) if flow_path.exists() else {}
        prices, raw, traded = {}, {}, {}
        for name, h in HORIZONS.items():
            moment = m["closed_ts"] - h
            if m["created_ts"] > moment:
                continue
            if h <= 7 * DAY:
                p = price_at(series["last"], moment, stale=6 * HOUR)
            else:
                p = price_at(series["whole"], moment, stale=2 * DAY)
            if p is None or name not in flow:
                continue
            raw[name] = p
            traded[name] = flow[name]["dollars"]
            if traded[name] >= LIVE_DOLLARS:
                prices[name] = p
        daily = {}  # the price 0.5, 1, 1.5, ... 60 days before the close, from the 12-hour series
        for half_days in range(1, 121):
            moment = m["closed_ts"] - half_days * DAY // 2
            if m["created_ts"] > moment:
                break
            p = price_at(series["whole"], moment, stale=DAY)
            if p is not None:
                daily[half_days] = p
        rows.append({**m, "prices": prices, "raw": raw, "traded": traded, "daily": daily})
    return rows


def arrays(rows, horizon, keep=lambda r: True):
    chosen = [r for r in rows if horizon in r["prices"] and keep(r)]
    p = np.array([r["prices"][horizon] for r in chosen])
    y = np.array([r["y"] for r in chosen], dtype=float)
    w = np.array([1 / r["chance"] for r in chosen])
    events = {e: i for i, e in enumerate(sorted({r["event"] for r in chosen}))}
    g = np.array([events[r["event"]] for r in chosen], dtype=int)
    return p, y, w, g, len(events)


def boot_weights(w, g, n_events):
    """Poisson bootstrap over events: every event is drawn a Poisson(1) number of times."""
    for _ in range(BOOT):
        yield w * RNG.poisson(1.0, n_events)[g]


def calibration(p, y, w, g, n_events, edges):
    which = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        inside = which == b
        if inside.sum() == 0:
            continue
        wb = w[inside]
        mean_p = float(np.average(p[inside], weights=wb))
        freq = float(np.average(y[inside], weights=wb))
        draws = []
        for bw in boot_weights(w, g, n_events):
            total = bw[inside].sum()
            if total > 0:
                draws.append((bw[inside] * y[inside]).sum() / total)
        low, high = np.percentile(draws, [2.5, 97.5]) if draws else (freq, freq)
        out.append({"low_edge": float(edges[b]), "high_edge": float(min(edges[b + 1], 1)), "mean_price": mean_p,
                    "frequency": freq, "low": float(low), "high": float(high), "markets": int(inside.sum()),
                    "events": int(len(set(g[inside])))})
    return out


def scores(p, y, w, g, n_events):
    """Brier score with Murphy's decomposition over the fine bins, accuracy, and the trivial baselines."""
    brier = float(np.average((p - y) ** 2, weights=w))
    base = float(np.average(y, weights=w))
    which = np.clip(np.digitize(p, FINE) - 1, 0, len(FINE) - 2)
    reliability = resolution = 0.0
    for b in range(len(FINE) - 1):
        inside = which == b
        if inside.sum() == 0:
            continue
        share = w[inside].sum() / w.sum()
        f = np.average(p[inside], weights=w[inside])
        o = np.average(y[inside], weights=w[inside])
        reliability += share * (f - o) ** 2
        resolution += share * (o - base) ** 2
    accuracy = float(np.average((p > 0.5) == (y == 1), weights=w))
    boot = []
    for bw in boot_weights(w, g, n_events):
        boot.append(np.average((p - y) ** 2, weights=bw))
    return {"markets": int(len(p)), "events": int(len(set(g.tolist()))), "brier": brier,
            "brier_low": float(np.percentile(boot, 2.5)), "brier_high": float(np.percentile(boot, 97.5)),
            "base_rate": base, "uncertainty": base * (1 - base), "reliability": float(reliability),
            "resolution": float(resolution), "skill": 1 - brier / (base * (1 - base)), "accuracy": accuracy,
            "always_no_accuracy": 1 - base, "decided": float(np.average((p <= 0.01) | (p >= 0.99), weights=w)),
            "ece": ece(p, y, w)}


def ece(p, y, w):
    """Expected calibration error: the weighted average gap between forecast and frequency over the fine bins."""
    which = np.clip(np.digitize(p, FINE) - 1, 0, len(FINE) - 2)
    total = 0.0
    for b in range(len(FINE) - 1):
        inside = which == b
        if inside.sum() == 0:
            continue
        gap = abs(np.average(p[inside], weights=w[inside]) - np.average(y[inside], weights=w[inside]))
        total += gap * w[inside].sum()
    return float(total / w.sum())


def contracts(p, y, w):
    """Both sides of every market as contracts: the first token at p, the second at 1 - p."""
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]), np.concatenate([w, w])


def returns_by_price(p, y, w, g, n_events):
    """Average return of buying a contract at its price, by price: y / price - 1."""
    cp, cy, cw = contracts(p, y, w)
    cg = np.concatenate([g, g])
    edges = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995])
    which = np.digitize(cp, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        inside = which == b
        if inside.sum() < 30:
            continue
        r = cy[inside] / cp[inside] - 1
        mean = float(np.average(r, weights=cw[inside]))
        draws = []
        for _ in range(BOOT):
            bw = cw[inside] * RNG.poisson(1.0, n_events)[cg[inside]]
            if bw.sum() > 0:
                draws.append(np.average(r, weights=bw))
        out.append({"low_edge": float(edges[b]), "high_edge": float(edges[b + 1]),
                    "mean_price": float(np.average(cp[inside], weights=cw[inside])),
                    "win_rate": float(np.average(cy[inside], weights=cw[inside])), "return": mean,
                    "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5)),
                    "contracts": int(inside.sum())})
    return out


def sharpness(p, w):
    counts, _ = np.histogram(p, bins=FINE, weights=w)
    return (counts / counts.sum()).tolist()


def by_group(rows, horizon, key, groups):
    out = {}
    for name in groups:
        p, y, w, g, n = arrays(rows, horizon, lambda r, name=name: key(r) == name)
        if len(p) >= 200:
            out[name] = scores(p, y, w, g, n)
            out[name]["calibration"] = calibration(p, y, w, g, n, EDGES)
    return out


def convergence(rows):
    """Brier score of all markets that lived at least 60 days and were traded a month before the close, by days
    before the close."""
    old = [r for r in rows if len(r["daily"]) == 120 and "1 month" in r["prices"]]
    y = np.array([r["y"] for r in old], dtype=float)
    w = np.array([1 / r["chance"] for r in old])
    days, brier, decided = [], [], []
    for half in range(1, 121):
        p = np.array([r["daily"][half] for r in old])
        days.append(half / 2)
        brier.append(float(np.average((p - y) ** 2, weights=w)))
        decided.append(float(np.average((p <= 0.01) | (p >= 0.99), weights=w)))
    return {"markets": len(old), "days": days, "brier": brier, "decided": decided,
            "uncertainty": float(np.average(y, weights=w) * (1 - np.average(y, weights=w)))}


def long_dated(rows):
    """Favourites months before the close: how often they won, and what buying them paid per year."""
    out = {}
    for horizon, days in (("1 month", 30), ("3 months", 91), ("6 months", 182)):
        p, y, w, g, n = arrays(rows, horizon)
        cp, cy, cw = contracts(p, y, w)
        for low, high in ((0.9, 0.95), (0.95, 0.98), (0.98, 0.995)):
            inside = (cp >= low) & (cp < high)
            if inside.sum() < 30:
                continue
            gross = float(np.average(cy[inside] / cp[inside], weights=cw[inside]))
            out[f"{horizon} {low:.3f}-{high:.3f}"] = {
                "horizon": horizon, "days": days, "low_edge": low, "high_edge": high,
                "contracts": int(inside.sum()), "mean_price": float(np.average(cp[inside], weights=cw[inside])),
                "win_rate": float(np.average(cy[inside], weights=cw[inside])), "gross": gross,
                "annualized": gross ** (365 / days) - 1}
    return out


def logit(p):
    q = np.clip(p, 0.001, 0.999)
    return np.log(q / (1 - q))


def fit_recalibration(p, y, w):
    """Weighted logistic regression of the outcome on the logit of the price: p' = sigmoid(a * logit(p) + b).
    a > 1 means the prices should have been more extreme, a < 1 less; b shifts them toward Yes or No."""
    x = logit(p)
    a, b = 1.0, 0.0
    for _ in range(50):  # Newton's method on two parameters
        q = 1 / (1 + np.exp(-(a * x + b)))
        grad = np.array([np.sum(w * (q - y) * x), np.sum(w * (q - y))])
        h = w * q * (1 - q)
        hess = np.array([[np.sum(h * x * x), np.sum(h * x)], [np.sum(h * x), np.sum(h)]])
        step = np.linalg.solve(hess, grad)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def recalibration(rows, horizon, split):
    """Fit the correction on markets that closed before `split`, apply it to those that closed after, and compare
    Brier scores on the later markets, with a bootstrap over events for the difference."""
    split_ts = int(np.datetime64(split, "s").astype(int))
    p, y, w, g, n = arrays(rows, horizon, lambda r: r["closed_ts"] < split_ts)
    a, b = fit_recalibration(p, y, w)
    tp, ty, tw, tg, tn = arrays(rows, horizon, lambda r: r["closed_ts"] >= split_ts)
    fixed = 1 / (1 + np.exp(-(a * logit(tp) + b)))
    before = float(np.average((tp - ty) ** 2, weights=tw))
    after = float(np.average((fixed - ty) ** 2, weights=tw))
    diffs = []
    for bw in boot_weights(tw, tg, tn):
        diffs.append(np.average((tp - ty) ** 2, weights=bw) - np.average((fixed - ty) ** 2, weights=bw))
    curve_x = np.linspace(0.01, 0.99, 99)
    return {"horizon": horizon, "split": split, "train_markets": int(len(p)), "test_markets": int(len(tp)),
            "a": a, "b": b, "brier_market": before, "brier_corrected": after,
            "gain_low": float(np.percentile(diffs, 2.5)), "gain_high": float(np.percentile(diffs, 97.5)),
            "curve": {"price": curve_x.tolist(), "corrected": (1 / (1 + np.exp(-(a * logit(curve_x) + b)))).tolist()}}


def book_snapshot():
    """Half the bid-ask spread of every open market, as a share of the price paid, by price: what a buyer pays
    above the middle of the book. Both sides of a market are contracts: Yes at the ask, No at 1 - the bid."""
    path = DATA_DIR / "book.jsonl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        taken = json.loads(f.readline())["taken"]
        books = [json.loads(line) for line in f]
    prices, costs, volumes = [], [], []
    for b in books:
        bid, ask = b["bid"], b["ask"]
        if not (0 < bid < ask < 1) or ask - bid > 0.2:
            continue
        mid, half = (bid + ask) / 2, (ask - bid) / 2
        for price in (mid, 1 - mid):
            prices.append(price)
            costs.append(half / price)
            volumes.append(float(b["volume"] or 0))
    prices, costs, volumes = np.array(prices), np.array(costs), np.array(volumes)
    edges = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995])
    out = {"taken": taken, "markets": len(books), "with_two_sided_book": int(len(prices) // 2), "by_price": []}
    which = np.digitize(prices, edges) - 1
    for k in range(len(edges) - 1):
        inside = which == k
        if inside.sum() < 30:
            continue
        out["by_price"].append({"low_edge": float(edges[k]), "high_edge": float(edges[k + 1]),
                                "contracts": int(inside.sum()), "median_cost": float(np.median(costs[inside])),
                                "q25": float(np.percentile(costs[inside], 25)),
                                "q75": float(np.percentile(costs[inside], 75))})
    out["by_volume"] = {}
    for low, high, label in ((0, 1e4, "under $10K"), (1e4, 1e5, "$10K–100K"), (1e5, 1e6, "$100K–1M"),
                             (1e6, 1e12, "$1M+")):
        inside = (volumes >= low) & (volumes < high) & (prices >= 0.2) & (prices <= 0.8)
        if inside.sum() >= 30:
            out["by_volume"][label] = {"contracts": int(inside.sum()), "median_cost": float(np.median(costs[inside]))}
    return out


def net_returns(entry, book):
    """The returns by price of `entry`, with the median spread cost of the same price band from the book snapshot:
    a buyer pays the ask, so a payout of 1 per contract costs price * (1 + cost)."""
    if book is None:
        return None
    cost = {(b["low_edge"], b["high_edge"]): b["median_cost"] for b in book["by_price"]}
    out = []
    for r in entry["returns"]:
        c = cost.get((r["low_edge"], r["high_edge"]))
        if c is None:
            continue
        out.append({**r, "cost": c, "net": (1 + r["return"]) / (1 + c) - 1})
    return out


def forecasters(rows, horizon="1 week"):
    """The same markets forecast three ways, to show what calibration and the Brier score reward: the prices
    themselves, a bolder copy (log-odds doubled), a timid one (log-odds halved), and a forecaster who always says
    the base rate."""
    p, y, w, g, n = arrays(rows, horizon)
    base = float(np.average(y, weights=w))
    out = {"base_rate": base, "brier_base_rate": base * (1 - base)}
    for name, k in (("market", 1.0), ("bold", 2.0), ("timid", 0.5)):
        q = 1 / (1 + np.exp(-k * logit(p))) if k != 1.0 else p
        out[name] = {"brier": float(np.average((q - y) ** 2, weights=w)),
                     "calibration": calibration(q, y, w, g, n, EDGES)}
    return out


def population():
    """Every resolved binary market with at least $1,000 traded, by the year it closed and its topic."""
    from datetime import datetime, timezone

    from markets import resolved

    counts, volume, yes = {}, {}, [0, 0]
    for m in resolved():
        year = str(datetime.fromtimestamp(m["closed_ts"], timezone.utc).year)
        counts.setdefault(year, {}).setdefault(m["topic"], 0)
        counts[year][m["topic"]] += 1
        volume[year] = volume.get(year, 0) + m["volume"]
        yes[0] += m["y"]
        yes[1] += 1
    return {"counts": counts, "volume": volume, "share_yes": yes[0] / yes[1], "markets": yes[1]}


BULLISH = re.compile(r"\b(above|reach|reaches|hit|hits|higher|up|exceed|ath|all.time high)\b", re.I)
BEARISH = re.compile(r"\b(below|dip|dips|fall|falls|drop|drops|lower|down|crash)\b", re.I)


def crypto_months(rows, horizon="1 week", low=0.3, high=0.7):
    """Crypto price markets that were close calls a week out, by the month they closed: how often the side that bet
    on rising prices won, next to Bitcoin's return that month (daily closes of Binance's BTCUSDT from the crypto
    trading post's data, if it is there). A market whose Yes pays when prices rise ("above", "reach") is bullish,
    one whose Yes pays when they fall ("dip", "below") bearish; questions without a dollar level, or with both
    kinds of words or neither, are left out.
    Markets on one underlying price do not resolve independently."""
    from datetime import datetime, timezone

    months = {}
    for r in rows:
        if r["topic"] != "crypto" or horizon not in r["prices"] or not low <= r["prices"][horizon] <= high:
            continue
        up, down = bool(BULLISH.search(r["question"])), bool(BEARISH.search(r["question"]))
        if up == down or "$" not in r["question"]:  # a price level, and one direction
            continue
        bull_won = r["y"] if up else 1 - r["y"]
        bull_price = r["prices"][horizon] if up else 1 - r["prices"][horizon]
        month = datetime.fromtimestamp(r["closed_ts"], timezone.utc).strftime("%Y-%m")
        entry = months.setdefault(month, {"markets": 0, "bull_won": 0, "bull_price": 0.0})
        entry["markets"] += 1
        entry["bull_won"] += bull_won
        entry["bull_price"] += bull_price
    for entry in months.values():
        entry["bull_price"] /= entry["markets"]
    btc = {}
    daily = Path.home() / ".cache" / "tfs-crypto" / "BTCUSDT-1d.parquet"
    if daily.exists():
        import pandas as pd

        close = pd.read_parquet(daily)["close"]
        monthly = close.resample("ME").last()
        btc = {d.strftime("%Y-%m"): float(r) for d, r in monthly.pct_change().dropna().items()}
    return {"months": months, "btc": btc, "low": low, "high": high}


def volume_band(r):
    for low, label in ((10_000_000, "$10M+"), (1_000_000, "$1M–10M"), (100_000, "$100K–1M"), (10_000, "$10K–100K"),
                       (1_000, "$1K–10K")):
        if r["volume"] >= low:
            return label
    return "under $1K"


VOLUME_BANDS = ["under $1K", "$1K–10K", "$10K–100K", "$100K–1M", "$1M–10M", "$10M+"]


def win_ratio(rows, horizon, low, high, key, field="prices"):
    """For contracts priced in [low, high) at the horizon, both sides of every market: how often they won divided by
    their average price, by group, with a bootstrap over events. Above 1, they won more often than priced."""
    groups = {}
    for r in rows:
        if horizon not in r[field]:
            continue
        name = key(r)
        if name is None:
            continue
        p = r[field][horizon]
        for price, won in ((p, r["y"]), (1 - p, 1 - r["y"])):
            if low <= price < high:
                groups.setdefault(name, []).append((price, won, 1 / r["chance"], r["event"]))
    out = {}
    for name, items in groups.items():
        if len(items) < 50:
            continue
        price = np.array([i[0] for i in items])
        won = np.array([i[1] for i in items], dtype=float)
        w = np.array([i[2] for i in items])
        events = {e: k for k, e in enumerate(sorted({i[3] for i in items}))}
        g = np.array([events[i[3]] for i in items])
        ratio = float((w * won).sum() / (w * price).sum())
        draws = []
        for _ in range(BOOT):
            bw = w * RNG.poisson(1.0, len(events))[g]
            if (bw * price).sum() > 0:
                draws.append((bw * won).sum() / (bw * price).sum())
        out[name] = {"ratio": ratio, "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5)),
                     "contracts": len(items), "mean_price": float(np.average(price, weights=w))}
    return out


TRADED_BANDS = [(0, 100, "under $100"), (100, 1_000, "$100–1K"), (1_000, 10_000, "$1K–10K"),
                (10_000, 100_000, "$10K–100K"), (100_000, 1e18, "$100K+")]


def trap(rows, horizon="1 week", field="raw"):
    """Longshots (2-20 cents) and favourites (80-98 cents) of every market with a price, grouped once by the market's
    total volume, which is only known after the outcome, and once by the dollars traded in the week before the
    forecast, which are known at the time."""
    def traded_band(r):
        dollars = r["traded"].get(horizon)
        for low, high, label in TRADED_BANDS:
            if dollars is not None and low <= dollars < high:
                return label
        return None

    out = {}
    for name, (low, high) in (("longshots", (0.02, 0.2)), ("favourites", (0.8, 0.98))):
        out[name] = {"by_volume": win_ratio(rows, horizon, low, high, volume_band, field),
                     "by_traded": win_ratio(rows, horizon, low, high, traded_band, field)}
    return out


def main():
    rows = load(union())
    print(len(rows), "markets with prices", flush=True)
    result = {"markets": len(rows), "events": len({r["event"] for r in rows}), "horizons": {}}
    first = min(r["closed_ts"] for r in rows)
    last = max(r["closed_ts"] for r in rows)
    result["closed_between"] = [first, last]
    for horizon in HORIZONS:
        p, y, w, g, n = arrays(rows, horizon)
        if len(p) < 200:
            continue
        entry = scores(p, y, w, g, n)
        entry["calibration"] = calibration(p, y, w, g, n, EDGES)
        entry["calibration_fine"] = calibration(p, y, w, g, n, FINE)
        entry["sharpness"] = sharpness(p, w)
        entry["returns"] = returns_by_price(p, y, w, g, n)
        doubt = (p >= 0.02) & (p <= 0.98)
        entry["in_doubt"] = scores(p[doubt], y[doubt], w[doubt], g[doubt], n)
        result["horizons"][horizon] = entry
        print(horizon, {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()
                        if not isinstance(v, list)}, flush=True)
    # the same markets at every horizon from a month to a day, so that the horizons can be compared
    common = [r for r in rows if all(h in r["prices"] for h in ("1 day", "1 week", "1 month"))]
    result["common"] = {}
    for horizon in ("1 month", "1 week", "1 day"):
        p, y, w, g, n = arrays(common, horizon)
        entry = scores(p, y, w, g, n)
        doubt = (p >= 0.02) & (p <= 0.98)
        entry["in_doubt"] = scores(p[doubt], y[doubt], w[doubt], g[doubt], n)
        result["common"][horizon] = entry
    result["topics"] = by_group(rows, "1 week", lambda r: r["topic"], list(TOPIC_NAMES))
    result["volume"] = by_group(rows, "1 week", volume_band, VOLUME_BANDS)
    result["trap"] = {"main": trap(load(by_topic())), "live_dollars": LIVE_DOLLARS}
    result["convergence"] = convergence(rows)
    result["long_dated"] = long_dated(rows)
    result["forecasters"] = forecasters(rows)
    result["crypto_months"] = crypto_months(rows)
    result["population"] = population()
    result["recalibration"] = {h: recalibration(rows, h, "2025-07-01") for h in ("1 week", "1 month")}
    book = book_snapshot()
    result["book"] = book
    result["horizons"]["1 week"]["net_returns"] = net_returns(result["horizons"]["1 week"], book)
    print("recalibration", {h: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
                            for h, r in result["recalibration"].items()}, flush=True)
    # what the first sample, chosen by volume and with every market that has a price, showed
    first = [{**r, "prices": r["raw"]} for r in load(selected())]
    p, y, w, g, n = arrays(first, "1 week")
    result["first_sample"] = {"markets": int(len(p)), "returns": returns_by_price(p, y, w, g, n)}
    OUT.mkdir(exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
