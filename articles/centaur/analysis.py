"""Turn the Kaggle lab's raw outputs (lab.py) into the small numbers the figures and the post use.

    python articles/centaur/analysis.py articles/centaur/data/raw <prompts dir>

data/raw holds the lab's outputs, gzipped: openloop.json from its first run, likelihood.json and logic.json from the
second (the first ran out of memory scoring whole sessions at once). The prompts directory is the generalization/
folder of github.com/marcelbinz/Llama-3.1-Centaur-70B, for the human two-step transcripts.
"""

import gzip
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402


def read(path):
    """A JSON file, or its gzipped copy next to it."""
    if path.exists():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else None


def mean_nll(per):
    """The paper's metric: NLL averaged over a participant's responses, then over participants."""
    return [sum(p["nll"]) / len(p["nll"]) for p in per]


def stay_probs(table):
    return {k: (s / n if n else None) for k, (s, n) in table.items()}


def indices(table):
    p = stay_probs(table)
    model_free = (p["1c"] + p["1r"]) / 2 - (p["0c"] + p["0r"]) / 2  # stay more after a reward
    model_based = (p["1c"] - p["1r"]) - (p["0c"] - p["0r"])  # ...after a reward that came through the common flight
    return model_free, model_based


def bootstrap(values, n=10000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def main(raw, prompts):
    raw, prompts = Path(raw), Path(prompts)
    out = {}
    rows = [json.loads(line) for line in open(prompts / "feher2020humans" / "prompts.jsonl")]
    humans = [lab.stay_table(lab.human_days(r["text"])) for r in rows]
    openloop = read(raw / "openloop.json")
    stays = {"people": humans} | {name: [p["stays"] for p in per] for name, per in openloop.items()}
    out["stays"] = {}
    for name, tables in stays.items():
        pooled = {}
        for t in tables:
            for k, (s, n) in t.items():
                pooled[k] = [pooled.get(k, [0, 0])[0] + s, pooled.get(k, [0, 0])[1] + n]
        mf, mb = zip(*(indices(t) for t in tables), strict=True)
        out["stays"][name] = {
            "pooled": {k: s / n for k, (s, n) in pooled.items()},
            "per participant": [stay_probs(t) for t in tables],
            "model-free index": st.mean(mf), "model-free CI": bootstrap(list(mf)),
            "model-based index": st.mean(mb), "model-based CI": bootstrap(list(mb)),
            "per participant model-based": list(mb),
        }
    coins = {name: sum(p["coins"] for p in per) / sum(len(p["days"]) for p in per) for name, per in openloop.items()}
    days = [d for r in rows for d in lab.human_days(r["text"]) if d[2] is not None]
    coins["people"] = sum(d[2] for d in days) / len(days)
    out["coins per day"] = coins
    out["valid mass"] = {name: st.mean(p["valid_mass"] for p in per) for name, per in openloop.items()}

    lik = read(raw / "likelihood.json")
    if lik:
        nll = {k.removeprefix("twostep/"): mean_nll(v) for k, v in lik.items()
               if k.startswith("twostep/") and isinstance(v, list) and v and "nll" in v[0]}
        out["twostep nll"] = {k: {"mean": st.mean(v), "CI": bootstrap(v), "per participant": v} for k, v in nll.items()}
        look = lik["twostep/lookup"]
        for name in ["repeat", "table"]:
            out["twostep nll"][name] = {"mean": st.mean(look[name]), "CI": bootstrap(look[name]),
                                        "per participant": look[name]}
        out["lookup counts"] = look["table counts"]
    logic = read(raw / "logic.json")
    if logic:
        meta = logic["logic/meta"]
        out["logic"] = {}
        for key, per in logic.items():
            if key == "logic/meta":
                continue
            answers = [[x for x, k in zip(p["nll"], p["kinds"], strict=True) if k == "press"] for p in per]
            per_participant = [sum(a) / len(a) for a in answers]
            by_question = [st.mean(a[q] for a in answers) for q in range(20)]
            out["logic"][key.removeprefix("logic/")] = {
                "mean": st.mean(per_participant), "CI": bootstrap(per_participant), "per question": by_question}
        out["logic"]["chance"] = math.log(5)
        # how often people chose each question's most popular answer: the content-free ceiling of "pick the usual one"
        modal = []
        for q in range(20):
            texts = [m["options"][q][m["answers"][q]] for m in meta]
            top = max(set(texts), key=texts.count)
            modal.append(texts.count(top) / len(texts))
        out["logic"]["share choosing the most popular answer"] = modal
        out["logic"]["participants"] = len(meta)
    (HERE / "data" / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("coins per day", "valid mass")}, indent=1))
    for name, s in out["stays"].items():
        print(name, {k: round(v, 3) for k, v in s["pooled"].items()}, "MF", round(s["model-free index"], 3),
              s["model-free CI"], "MB", round(s["model-based index"], 3), [round(x, 3) for x in s["model-based CI"]])
    for k, v in out.get("twostep nll", {}).items():
        print("nll", k, round(v["mean"], 4), [round(x, 4) for x in v["CI"]])
    for k, v in out.get("logic", {}).items():
        if isinstance(v, dict):
            print("logic", k, round(v["mean"], 4), [round(x, 4) for x in v["CI"]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
