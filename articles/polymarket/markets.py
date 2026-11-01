"""Resolved Polymarket markets, as the analysis sees them: one row per binary market with a clean outcome.

Every market on Polymarket is binary: a multi-outcome event ("Who will win the election?") is a set of Yes/No
markets, one per candidate. A market counts as resolved when its final prices are 1 and 0, to within a cent (the
early markets record 0.99999998); markets settled at 50-50 (an ambiguous outcome) and markets without a closing time
are left out. The forecast studied is the price of the first outcome token ("Yes" in a Yes/No market), and the
outcome is whether that token paid out.

The topic of a market comes from its event's tags, with a few title rules for tags that are missing or too broad.
Crypto "Up or Down" markets (will Bitcoin be higher at the end of these 15 minutes, this hour or this day) are a
topic of their own: there are hundreds of thousands of them.
"""

import gzip
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-polymarket"))
MIN_VOLUME = 10_000  # dollars traded over the market's life, for the price analysis
# Which markets get a price history: (lowest volume, share) by topic, the first band a market reaches wins. There are
# far too many sports and weather markets to take them all, so smaller ones are sampled more thinly.
SHARES = {"sports": [(1_000_000, 1.0), (10_000, 0.1)], "crypto": [(100_000, 1.0), (10_000, 0.25)],
          "science": [(100_000, 1.0), (10_000, 0.2)]}
DEFAULT_SHARES = [(100_000, 1.0), (10_000, 0.5)]
UP_DOWN_SAMPLE = 4000  # how many crypto "Up or Down" markets get a price history
# The main sample: a share of every resolved market by topic, whatever its volume. The topic is known when a market
# opens, so sampling by it, unlike sampling by volume, does not depend on how the market ended.
TOPIC_SHARES = {"sports": 0.017, "crypto-updown": 0.01, "science": 0.035, "crypto": 0.055, "politics": 0.11,
                "economy": 0.16, "culture": 0.2, "other": 0.5}

TOPICS = [  # first match wins
    ("crypto-updown", None, re.compile(r"\bup or down\b", re.I)),
    ("sports", {"sports", "nba", "nfl", "soccer", "mlb", "nhl", "tennis", "ufc", "mma", "f1", "formula 1", "golf",
                "cricket", "esports", "boxing", "ncaa", "cbb", "cfb", "epl", "la liga", "champions league", "olympics",
                "chess", "basketball", "baseball", "hockey", "football"}, None),
    ("crypto", {"crypto", "bitcoin", "ethereum", "solana", "crypto prices", "xrp", "memecoins", "airdrops", "nfts",
                "stablecoins", "defi", "token launch", "pre-market"}, None),
    ("politics", {"politics", "elections", "us election", "global elections", "world elections", "trump", "congress",
                  "us politics", "geopolitics", "world", "ukraine", "israel", "middle east", "china", "russia",
                  "midterms", "primaries", "courts", "trump presidency", "foreign policy", "gaza", "iran"}, None),
    ("economy", {"economy", "business", "finance", "fed", "fed rates", "inflation", "stocks", "economic policy",
                 "tariffs", "earnings", "commodities", "ipos", "equities", "jobs report", "gdp", "interest rates"},
     None),
    ("culture", {"pop culture", "culture", "entertainment", "music", "movies", "awards", "oscars", "grammys", "tv",
                 "celebrities", "tweet markets", "mentions", "youtube", "box office", "eurovision", "gaming"}, None),
    ("science", {"science", "tech", "ai", "space", "weather", "climate", "health", "pandemics", "openai", "spacex",
                 "global temp", "hurricanes"}, None),
]
TOPIC_NAMES = {"politics": "Politics and world", "sports": "Sports", "crypto": "Crypto prices and events",
               "crypto-updown": "Crypto up or down", "economy": "Economy and markets", "culture": "Culture and media",
               "science": "Science, tech and weather", "other": "Other"}


def timestamp(text):
    if not text:
        return None
    text = text.replace(" ", "T").replace("+00", "+00:00") if text.endswith("+00") else text.replace(" ", "T")
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def topic(event):
    tags = {t.lower() for t in event.get("tags") or [] if t}
    title = event.get("title") or ""
    for name, labels, pattern in TOPICS:
        if pattern is not None and pattern.search(title):
            return name
        if labels is not None and tags & labels:
            return name
    category = (event.get("category") or "").lower()  # old markets have a category instead of tags
    for name, labels, _ in TOPICS:
        if labels is not None and any(label in category for label in labels):
            return name
    return "other"


def events():
    """Every downloaded event once: the files of download.py's windows overlap at their edges."""
    seen = set()
    for path in sorted(DATA_DIR.glob("events*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    if event["id"] not in seen:
                        seen.add(event["id"])
                        yield event
        except (EOFError, OSError, json.JSONDecodeError):  # a file still being written ends mid-line
            continue


def resolved():
    """Every binary market with a clean outcome, once (an event can list a market that another event lists too)."""
    seen = set()
    for event in events():
        name = topic(event)
        siblings = len(event.get("markets") or [])
        for m in event.get("markets") or []:
            if m["id"] in seen or not m.get("outcomePrices") or not m.get("clobTokenIds"):
                continue
            seen.add(m["id"])
            try:
                final = [float(p) for p in json.loads(m["outcomePrices"])]
                tokens = json.loads(m["clobTokenIds"])
                outcomes = json.loads(m["outcomes"])
            except (TypeError, ValueError):
                continue
            # final prices of the early markets, run by an automated market maker, read 0.99999998 and 0.00000002
            if len(final) != 2 or len(tokens) != 2 or min(final) > 0.01 or max(final) < 0.99:
                continue
            closed = timestamp(m.get("closedTime") or event.get("closedTime"))
            created = timestamp(m.get("createdAt") or m.get("startDate"))
            if closed is None or created is None or closed <= created:
                continue
            yield {"id": m["id"], "condition": m.get("conditionId"), "event": event["id"], "topic": name,
                   "question": m["question"],
                   "outcome": outcomes[0], "y": int(final[0] > 0.5), "token": tokens[0],
                   "volume": float(m.get("volumeNum") or 0), "created_ts": created, "closed_ts": closed,
                   "end_ts": timestamp(m.get("endDate")), "neg_risk": bool(m.get("negRisk") or event.get("negRisk")),
                   "siblings": siblings}


def sampled(market_id, share, salt=""):
    """A fixed pseudo-random choice: the same markets are picked on every run; a different salt gives an
    independent draw."""
    digest = hashlib.sha256(f"{salt}{market_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def chance(m):
    """The probability that a market other than Up or Down is chosen for the price analysis."""
    for low, share in SHARES.get(m["topic"], DEFAULT_SHARES):
        if m["volume"] >= low:
            return share
    return 0.0


def selected():
    """The markets that get a price history, each with the probability it had of being chosen (`chance`), so that
    pooled statistics can weight it by 1 / chance and describe every market with at least MIN_VOLUME dollars
    traded: shares by topic and volume as in SHARES, and a fixed random sample of about UP_DOWN_SAMPLE crypto Up
    or Down markets."""
    chosen, up_down = [], []
    for m in resolved():
        if m["volume"] < MIN_VOLUME:
            continue
        if m["topic"] == "crypto-updown":
            up_down.append(m)
            continue
        share = chance(m)
        if share > 0 and sampled(m["id"], share):
            chosen.append({**m, "chance": share})
    share = min(1.0, UP_DOWN_SAMPLE / max(1, len(up_down)))
    chosen += [{**m, "chance": share} for m in up_down if sampled(m["id"], share)]
    return chosen


def by_topic():
    """The main sample: each resolved market is drawn with its topic's share in TOPIC_SHARES (`chance`), whatever
    its volume. A market's volume is counted over its whole life, after the forecast too, and a surprise draws
    trading: choosing markets by volume favours those where the unlikely happened. Drawing by topic does not."""
    chosen = []
    for m in resolved():
        share = TOPIC_SHARES[m["topic"]]
        if sampled(m["id"], share, salt="uniform"):
            chosen.append({**m, "chance": share})
    return chosen


def union():
    """The main sample of the analysis: every market drawn by topic (by_topic) or by volume (selected), each with its
    chance of being in either, 1 - (1 - by topic) * (1 - by volume); the two draws are independent. Weighting each
    market by 1 / that chance describes all resolved markets (Horvitz and Thompson's estimator), since every market
    has a known chance above zero. What biases results is leaving markets out or grouping them by volume, not
    drawing big ones more often."""
    rows = list(resolved())
    up_down = [r for r in rows if r["topic"] == "crypto-updown" and r["volume"] >= MIN_VOLUME]
    up_down_share = min(1.0, UP_DOWN_SAMPLE / max(1, len(up_down)))
    chosen = []
    for m in rows:
        by_topic_share = TOPIC_SHARES[m["topic"]]
        in_topic = sampled(m["id"], by_topic_share, salt="uniform")
        if m["volume"] < MIN_VOLUME:
            by_volume_share = 0.0
        elif m["topic"] == "crypto-updown":
            by_volume_share = up_down_share
        else:
            by_volume_share = chance(m)
        in_volume = by_volume_share > 0 and sampled(m["id"], by_volume_share)
        if in_topic or in_volume:
            chosen.append({**m, "chance": 1 - (1 - by_topic_share) * (1 - by_volume_share)})
    return chosen


if __name__ == "__main__":
    from collections import Counter

    rows = list(resolved())
    print(len(rows), "resolved binary markets")
    print("by topic:", Counter(r["topic"] for r in rows).most_common())
    big = [r for r in rows if r["volume"] >= MIN_VOLUME]
    print(len(big), f"with volume >= {MIN_VOLUME}:", Counter(r["topic"] for r in big).most_common())
    print("share resolving to the first outcome:", sum(r["y"] for r in rows) / len(rows))
    print(len(selected()), "selected for price histories")
