"""Public data for the post "LLMs for Those Who Missed Out", as it stood on the post's date, 24 April 2024.

    data/arena.json     the Chatbot Arena leaderboard (text, overall) published on 22 April 2024, from LMArena's
                        leaderboard dataset on Hugging Face (CC BY 4.0)
    data/compute.json   training compute of language models published up to 24 April 2024, from Epoch AI's dataset
                        of notable AI models (CC BY 4.0); Epoch's estimates as of the download date

    uv run python articles/llms_intro/prepare.py
"""

import io
import json
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ARENA = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/text/full-00000-of-00001.parquet"
EPOCH = "https://epoch.ai/data/notable_ai_models.csv"
CUTOFF = "2024-04-24"
OUT = Path(__file__).parent / "data"


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(request).read()


arena = pd.read_parquet(io.BytesIO(download(ARENA)))
board = arena[(arena["category"] == "overall") & (arena["leaderboard_publish_date"] == "2024-04-22")]
board = board.sort_values("rating", ascending=False)
(OUT / "arena.json").write_text(json.dumps({
    "source": ARENA,
    "date": "2024-04-22",
    "rows": [{"model": r.model_name, "organization": r.organization, "license": r.license,
              "rating": round(r.rating, 1), "low": round(r.rating_lower, 1), "high": round(r.rating_upper, 1),
              "votes": int(r.vote_count)} for r in board.itertuples()],
}, indent=1), encoding="utf-8")

epoch = pd.read_csv(io.BytesIO(download(EPOCH)))
language = epoch[epoch["Domain"].fillna("").str.contains("Language")
                 & epoch["Training compute (FLOP)"].notna()
                 & (epoch["Publication date"] <= CUTOFF)]
(OUT / "compute.json").write_text(json.dumps({
    "source": EPOCH,
    "downloaded": date.today().isoformat(),
    "rows": [{"model": r["Model"], "organization": r["Organization"], "date": r["Publication date"],
              "flop": float(r["Training compute (FLOP)"]), "confidence": r["Confidence"]}
             for _, r in language.sort_values("Publication date").iterrows()],
}, indent=1), encoding="utf-8")
print(len(board), "arena rows;", len(language), "language models with compute")
