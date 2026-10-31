"""What a market order costs: the order books of Binance's USDT-margined perpetual futures.

The contracts are ranked by their 24-hour turnover, and the books of those at a few ranks (1st, 2nd, 5th, ... 200th)
are read several times a few minutes apart. For an order buying a given amount in dollars, the cost is the
distance between its average fill price and the mid price (half the spread plus the climb up the book), in basis
points. The median over the snapshots goes to data/depth.json.

    uv run python articles/crypto_trading/depth.py
"""

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).parent / "data"
API = "https://fapi.binance.com/fapi/v1"
RANKS = [1, 2, 5, 10, 25, 50, 100, 200]
SIZES = [1_000, 10_000, 100_000, 1_000_000]
SNAPSHOTS = 6
PAUSE = 300  # seconds between snapshots


def get(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=30) as response:
        return json.load(response)


def ranked_contracts():
    tickers = get("ticker/24hr")
    perpetual = set()
    for s in get("exchangeInfo")["symbols"]:
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
            perpetual.add(s["symbol"])
    rows = [(float(t["quoteVolume"]), t["symbol"]) for t in tickers if t["symbol"] in perpetual]
    rows.sort(reverse=True)
    return rows


def buy_cost_bps(asks, mid, dollars):
    """Average fill price of a market buy of `dollars`, against the mid price, in basis points."""
    spent, bought = 0.0, 0.0
    for price, quantity in asks:
        take = min(quantity, (dollars - spent) / price)
        spent += take * price
        bought += take
        if spent >= dollars - 1e-9:
            return 1e4 * (spent / bought / mid - 1)
    return None  # the book we were given is not deep enough


def main():
    ranked = ranked_contracts()
    chosen = {rank: ranked[rank - 1] for rank in RANKS}
    taken = []
    for snapshot in range(SNAPSHOTS):
        moment = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "books": {}}
        for _rank, (_volume, symbol) in chosen.items():
            book = get(f"depth?symbol={symbol}&limit=1000")
            bids = [(float(p), float(q)) for p, q in book["bids"]]
            asks = [(float(p), float(q)) for p, q in book["asks"]]
            mid = (bids[0][0] + asks[0][0]) / 2
            moment["books"][symbol] = {"spread_bps": 1e4 * (asks[0][0] - bids[0][0]) / mid,
                                       "costs_bps": {str(s): buy_cost_bps(asks, mid, s) for s in SIZES}}
        taken.append(moment)
        print(moment["time"], {s: round(b["costs_bps"]["100000"] or -1, 2) for s, b in moment["books"].items()},
              flush=True)
        if snapshot < SNAPSHOTS - 1:
            time.sleep(PAUSE)
    contracts = []
    for rank, (volume, symbol) in chosen.items():
        costs = {}
        for size in SIZES:
            values = sorted(m["books"][symbol]["costs_bps"][str(size)] for m in taken
                            if m["books"][symbol]["costs_bps"][str(size)] is not None)
            costs[str(size)] = values[len(values) // 2] if values else None
        spreads = sorted(m["books"][symbol]["spread_bps"] for m in taken)
        contracts.append({"rank": rank, "symbol": symbol, "volume_24h": volume,
                          "spread_bps": spreads[len(spreads) // 2], "costs_bps": costs})
    result = {"venue": "Binance USD-M perpetual futures", "snapshots": [m["time"] for m in taken],
              "contracts_listed": len(ranked), "sizes": SIZES, "contracts": contracts}
    OUT.mkdir(exist_ok=True)
    (OUT / "depth.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
