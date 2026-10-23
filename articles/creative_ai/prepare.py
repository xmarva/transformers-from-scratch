"""Data for the figures of the post "Can AI be creative?".

    data/move37.json  the first 37 moves of AlphaGo vs Lee Sedol, game 2 (Seoul, 10 March 2016), from the game record
                      in Andries Brouwer's archive at CWI
    data/aut.json     per participant or chatbot session, the mean and the best rating of ideas in Koivisto & Grassini's
                      Alternate Uses Task study (Scientific Reports, 2023), from their open data on OSF

The OSF data is a spreadsheet behind a view-only link; download it from
https://osf.io/qdz3n/?view_only=d2da32d06f0f4f7ca5bd4bdfc8cf4346 and pass its path.

    uv run --with openpyxl python articles/creative_ai/prepare.py path/to/data.xlsx
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

GAME = "https://homepages.cwi.nl/~aeb/go/games/games/AlphaGo/LeeSedol/2.sgf"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

sgf = urllib.request.urlopen(GAME).read().decode("utf-8", errors="replace")
moves = re.findall(r";([BW])\[([a-s]{2})\]", sgf)
assert moves[36] == ("B", "oj"), "move 37 should be Black at oj (P10)"
(OUT / "move37.json").write_text(json.dumps({
    "source": GAME,
    "black": "AlphaGo",
    "white": "Lee Sedol",
    "moves": [[color, point] for color, point in moves[:37]],
}, indent=1), encoding="utf-8")

table = pd.read_excel(sys.argv[1])
rows = []
for _, r in table.iterrows():
    rows.append({
        "who": r["Group2"],
        "mean_rating": round(float(r["AUTs"]), 4),
        "best_rating": round(float(r["AUTmaxs"]), 4),
    })
(OUT / "aut.json").write_text(json.dumps({
    "source": "Koivisto & Grassini 2023, https://doi.org/10.1038/s41598-023-40858-3; data https://osf.io/qdz3n/",
    "rows": rows,
}, indent=1), encoding="utf-8")
print(len(moves), "moves read;", len(rows), "AUT rows")
