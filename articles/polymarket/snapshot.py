"""Prices of a few open Polymarket markets at one moment, for the opening of the post: the US midterm elections three
days before the vote, and an exchange insolvency market two days after the Binance hack. Writes data/snapshot.json.

    uv run python articles/polymarket/snapshot.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from download import GAMMA, get

OUT = Path(__file__).parent / "data"
EVENTS = {"house": 32225, "senate": 32224, "cex": 199777}


def main():
    taken = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = {"taken": taken, "events": {}}
    for name, event_id in EVENTS.items():
        event = get(f"{GAMMA}/events/{event_id}")
        markets = []
        for m in event["markets"]:
            if not m.get("outcomePrices"):
                continue
            markets.append({"question": m["question"], "outcome": m.get("groupItemTitle") or m["question"],
                            "price": float(json.loads(m["outcomePrices"])[0]), "bestBid": m.get("bestBid"),
                            "bestAsk": m.get("bestAsk"), "volume": m.get("volumeNum")})
        result["events"][name] = {"title": event["title"], "volume": event.get("volume"), "endDate": event["endDate"],
                                  "markets": markets}
    OUT.mkdir(exist_ok=True)
    (OUT / "snapshot.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1)[:1500])


if __name__ == "__main__":
    main()
