"""Counts in the public Psych-101 training file: rows, scored responses, and what the go/no-go task scores.

    curl -L -o prompts_training.jsonl https://huggingface.co/datasets/marcelbinz/Psych-101/resolve/main/prompts_training.jsonl
    python articles/centaur/psych101.py prompts_training.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def main(path):
    rows = responses = 0
    experiments = Counter()
    go = {"transcripts": 0, "scored": 0, "scored equal to the instructed key": 0, "press nothing (never scored)": 0}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows += 1
        responses += r["text"].count("<<")
        experiments[r["experiment"]] += 1
        if "gonogo" in r["experiment"]:
            key = re.search(r"press button (\w)", r["text"]).group(1)
            scored = re.findall(r"<<(.*?)>>", r["text"])
            go["transcripts"] += 1
            go["scored"] += len(scored)
            go["scored equal to the instructed key"] += sum(s == key for s in scored)
            go["press nothing (never scored)"] += r["text"].count("press nothing")
    out = {"rows": rows, "responses": responses, "experiment files": len(experiments), "go/no-go": go}
    (Path(__file__).parent / "data" / "psych101.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
