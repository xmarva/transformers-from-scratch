"""Price histories of a few single markets the post discusses: the 2024 US presidential election and the 2025 Nobel
Peace Prize. Writes data/cases.json.

    uv run python articles/polymarket/cases.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from download import GAMMA, history
from download import get as fetch

OUT = Path(__file__).parent / "data"


def ts(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def token(market_id):
    return json.loads(fetch(f"{GAMMA}/markets/{market_id}")["clobTokenIds"])[0]


def market(market_id):
    m = fetch(f"{GAMMA}/markets/{market_id}")
    return {"id": market_id, "question": m["question"], "outcome": m.get("groupItemTitle") or m["question"],
            "volume": m.get("volumeNum"), "final": json.loads(m["outcomePrices"]), "closed": m.get("closedTime")}


def main():
    cases = {}
    trump, harris = "253591", "253597"
    cases["election"] = {
        "markets": [market(trump), market(harris)],
        # the whole year at a 12-hour step, then election night at a 10-minute step
        "whole": {m: history(token(m), interval="max", fidelity=720) for m in (trump, harris)},
        "night": {m: history(token(m), startTs=ts("2024-11-05T12:00:00"), endTs=ts("2024-11-06T12:00:00"),
                             fidelity=10) for m in (trump, harris)},
    }
    nobel_event = fetch(f"{GAMMA}/events/31406")
    leaders = sorted(nobel_event["markets"], key=lambda m: -(m.get("volumeNum") or 0))[:6]
    cases["nobel"] = {
        "title": nobel_event["title"],
        "markets": [market(m["id"]) for m in leaders],
        "prices": {m["id"]: history(token(m["id"]), startTs=ts("2025-10-07T00:00:00"),
                                    endTs=ts("2025-10-10T12:00:00"), fidelity=10) for m in leaders},
        "announced": "2025-10-10T09:00:00Z",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(cases))
    for name, case in cases.items():
        print(name, [(m["outcome"], m["final"]) for m in case["markets"]])


if __name__ == "__main__":
    main()
