"""What open LLMs remember about Bitcoin's past, and why that ruins a backtest of an LLM trader.

Seven open models released between May 2024 and February 2026 answer three kinds of questions about dates from
2018 to October 2026:

- the closing price of Bitcoin on the 1st and the 15th of every month (greedy decoding, the number parsed);
- whether Bitcoin rose or fell over a calendar month;
- whether it rose or fell over a week, for every week, Monday to Monday.

For the up-or-down questions the model does not generate: the answer is read from the probabilities of the first
tokens of "up" and "down" in its reply, so that a model that would rather explain itself still has to choose.
A model that only reasons should be right half the time; one that remembers is right more often for the dates it
was trained on. The weekly answers are also traded: long Bitcoin for the week when the model says up, short when it
says down.

Two steps: `items` builds the questions and the true answers from the daily candles of prepare.py (locally),
`run` asks the models (on a GPU; kaggle.py runs it on Kaggle) and writes data/memory.json.

    uv run --with pandas --with pyarrow python articles/crypto_trading/memory.py items
    python articles/crypto_trading/memory.py run
"""

import json
import os
import re
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / ".cache" / "tfs-crypto"))
OUT = Path(os.environ.get("OUT", Path(__file__).parent / "data"))
ITEMS = Path(__file__).parent / "data" / "memory_items.json"
FIRST, LAST = date(2018, 1, 1), date(2026, 10, 29)
MODELS = [  # Hugging Face repository, release date (the repository's creation date on the Hub)
    ("mistralai/Mistral-7B-Instruct-v0.3", "2024-05-22"),
    ("Qwen/Qwen2.5-7B-Instruct", "2024-09-16"),
    ("allenai/OLMo-2-1124-7B-Instruct", "2024-12-18"),
    ("microsoft/Phi-4-mini-instruct", "2025-02-19"),
    ("Qwen/Qwen3-8B", "2025-04-27"),
    ("allenai/Olmo-3-7B-Instruct", "2025-11-19"),
    ("Qwen/Qwen3.5-9B", "2026-02-27"),
]
SYSTEM = "You answer questions about historical market data. Answer briefly, without explanations."


def nice(d):
    return f"{d:%B} {d.day}, {d.year}"


def build_items():
    import pandas as pd

    candles = pd.read_parquet(DATA_DIR / "BTCUSDT-1d.parquet")
    close = {t.date(): float(c) for t, c in candles["close"].items()}
    prices, months, weeks = [], [], []
    d = FIRST
    while d <= LAST:
        for day in (1, 15):
            when = date(d.year, d.month, day)
            if when <= LAST and when in close:
                prices.append({"date": when.isoformat(), "close": close[when],
                               "question": f"What was the closing price of Bitcoin in US dollars on {nice(when)}? "
                                           "Reply with the number only."})
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    d = FIRST
    while True:
        following = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        last_day, before = following - timedelta(days=1), d - timedelta(days=1)
        if last_day > LAST:
            break
        if last_day in close and before in close:
            months.append({"month": f"{d:%Y-%m}", "up": close[last_day] > close[before],
                           "return": close[last_day] / close[before] - 1,
                           "question": f"Did the price of Bitcoin go up or down during {d:%B %Y}? "
                                       "Answer with one word: up or down."})
        d = following
    d = FIRST  # a Monday
    while d + timedelta(days=7) <= LAST:
        end = d + timedelta(days=7)
        if d in close and end in close:
            weeks.append({"start": d.isoformat(), "up": close[end] > close[d], "return": close[end] / close[d] - 1,
                          "question": f"Did the price of Bitcoin go up or down between {nice(d)} and {nice(end)}? "
                                      "Answer with one word: up or down."})
        d = end
    ITEMS.write_text(json.dumps({"prices": prices, "months": months, "weeks": weeks}, indent=1))
    print(len(prices), "prices,", len(months), "months,", len(weeks), "weeks")


def parse_price(text):
    """The number an answer starts with; an answer that starts with words ("I don't have access...") has none, even
    if a year appears later in it."""
    text = text.replace(",", "").replace("$", "").strip()
    match = re.match(r"(\d+(?:\.\d+)?)\s*([kK]\b)?", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * 1000 if match.group(2) else value


def reparse():
    """Parse the stored answers again with parse_price, e.g. after a change to it."""
    path = OUT / "memory.json"
    result = json.loads(path.read_text())
    changed = 0
    for entry in result["models"].values():
        for answer in entry.get("prices", []):
            value = parse_price(answer["answer"])
            changed += value != answer["value"]
            answer["value"] = value
    path.write_text(json.dumps(result, indent=1))
    print(changed, "answers parsed differently")


def load(repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # T4 GPUs have no bfloat16; on a machine without CUDA (a smoke test) float32 is the safe choice
    kwargs = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32, "device_map": "auto"}
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    except ValueError:  # Qwen3.5 is published as an image-and-text model; we only give it text
        model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    model.eval()
    return tokenizer, model


def prompt(tokenizer, question):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except Exception:  # templates without a system role: fold it into the question
        messages = [{"role": "user", "content": SYSTEM + "\n\n" + question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first_tokens(tokenizer, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            pieces = tokenizer.encode(variant, add_special_tokens=False)
            if pieces:
                ids.add(pieces[0])
    return sorted(ids)


def ask_up_down(tokenizer, model, questions, batch=16):
    import torch

    up_ids = first_tokens(tokenizer, ["up", "Up", "UP"])
    down_ids = first_tokens(tokenizer, ["down", "Down", "DOWN"])
    out = []
    for i in range(0, len(questions), batch):
        texts = [prompt(tokenizer, q) for q in questions[i:i + batch]]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        probabilities = torch.softmax(logits, dim=-1)
        p_up = probabilities[:, up_ids].sum(dim=-1)
        p_down = probabilities[:, down_ids].sum(dim=-1)
        for u, d in zip(p_up.tolist(), p_down.tolist(), strict=True):
            out.append({"p_up": u / (u + d) if u + d > 0 else 0.5, "mass": u + d})
    return out


def ask_prices(tokenizer, model, questions, batch=16):
    import torch

    out = []
    for i in range(0, len(questions), batch):
        texts = [prompt(tokenizer, q) for q in questions[i:i + batch]]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=20, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        for row in generated[:, encoded["input_ids"].shape[1]:]:
            out.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return out


def paths():
    """The questions and where the answers go; a smoke test uses a tiny model and a few questions."""
    items = json.loads(ITEMS.read_text())
    if os.environ.get("SMOKE"):
        items = {kind: questions[:: len(questions) // 4] for kind, questions in items.items()}
        return items, OUT / "memory-smoke.json"
    return items, OUT / "memory.json"


def run_one(repo, released):
    """Ask one model every question and add its answers to the results file."""
    import torch

    items, path = paths()
    started = time.time()
    tokenizer, model = load(repo)
    answers = ask_prices(tokenizer, model, [p["question"] for p in items["prices"]])
    months = ask_up_down(tokenizer, model, [m["question"] for m in items["months"]])
    weeks = ask_up_down(tokenizer, model, [w["question"] for w in items["weeks"]])
    result = json.loads(path.read_text()) if path.exists() else {"models": {}}
    result["models"][repo] = {
        "released": released, "seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "prices": [{"date": p["date"], "answer": a, "value": parse_price(a)} for p, a in
                   zip(items["prices"], answers, strict=True)],
        "months": [{"month": m["month"], **a} for m, a in zip(items["months"], months, strict=True)],
        "weeks": [{"start": w["start"], **a} for w, a in zip(items["weeks"], weeks, strict=True)],
    }
    path.write_text(json.dumps(result, indent=1))
    print(repo, f"{time.time() - started:.0f} s; first answers:", answers[:3], flush=True)


def run():
    """Every model in a process of its own: a model's GPU memory is only reliably freed when its process ends."""
    import subprocess

    models = MODELS
    if os.environ.get("ONLY"):  # a comma-separated subset of MODELS, e.g. to rerun the ones that failed
        wanted = os.environ["ONLY"].split(",")
        models = [(repo, released) for repo, released in MODELS if repo in wanted]
    if os.environ.get("SMOKE"):
        models = [("Qwen/Qwen2.5-0.5B-Instruct", "2024-09-19")]
    _, path = paths()
    for repo, released in models:
        result = json.loads(path.read_text()) if path.exists() else {"models": {}}
        if repo in result["models"] and "error" not in result["models"][repo]:
            continue
        done = subprocess.run([sys.executable, __file__, "one", repo, released]).returncode
        if done != 0:  # one model failing should not cost the others
            result = json.loads(path.read_text()) if path.exists() else {"models": {}}
            result["models"][repo] = {"released": released, "error": f"exit code {done}"}
            path.write_text(json.dumps(result, indent=1))
            print(repo, "failed with exit code", done, flush=True)
        # seven models of 8 to 19 GB do not fit on a Kaggle disk together: drop each one's files when it is done
        cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
        shutil.rmtree(cache, ignore_errors=True)


def merge():
    """Add the models of other result files (named on the command line) to data/memory.json, replacing failed or
    missing ones."""
    path = OUT / "memory.json"
    result = json.loads(path.read_text()) if path.exists() else {"models": {}}
    for name in sys.argv[2:]:
        for repo, entry in json.loads((OUT / name).read_text())["models"].items():
            if "error" not in entry or repo not in result["models"]:
                result["models"][repo] = entry
    path.write_text(json.dumps(result, indent=1))
    print({repo: "error" if "error" in e else "ok" for repo, e in result["models"].items()})


if __name__ == "__main__":
    if sys.argv[1] == "one":
        run_one(sys.argv[2], sys.argv[3])
    else:
        {"items": build_items, "run": run, "merge": merge, "reparse": reparse}[sys.argv[1]]()
