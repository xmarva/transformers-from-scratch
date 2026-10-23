"""Execute built lesson notebooks top to bottom, the way a learner would.

    uv run python tools/run.py --smoke        # every lesson on a tiny slice of data (what CI does)
    uv run python tools/run.py 01             # lesson 01 in full, e.g. to export it with real outputs

Executed notebooks are written to build/executed/<lesson>.ipynb. The Colab build is used because it is
platform-neutral: it carries no Kaggle-only cells.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def run(notebook: Path, out: Path, smoke: bool, timeout: int) -> float:
    nb = nbformat.read(notebook, as_version=4)
    env_before = os.environ.get("TFS_SMOKE")
    os.environ["TFS_SMOKE"] = "1" if smoke else "0"
    try:
        start = time.perf_counter()
        NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}
        ).execute()
        elapsed = time.perf_counter() - start
    finally:
        if env_before is None:
            os.environ.pop("TFS_SMOKE", None)
        else:
            os.environ["TFS_SMOKE"] = env_before
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    return elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lessons", nargs="*", help="lesson prefixes, e.g. 01; default: all")
    parser.add_argument("--smoke", action="store_true", help="tiny data and few steps")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed per cell")
    args = parser.parse_args(argv)

    notebooks = sorted((ROOT / "notebooks" / "colab").glob("*.ipynb"))
    if args.lessons:
        notebooks = [n for n in notebooks if any(n.stem.startswith(p) for p in args.lessons)]
    if not notebooks:
        print("no lesson notebooks to run")
        return 0
    for nb in notebooks:
        out = ROOT / "build" / "executed" / nb.name
        out.parent.mkdir(parents=True, exist_ok=True)
        elapsed = run(nb, out, args.smoke, args.timeout)
        print(f"ran {nb.stem} in {elapsed:.0f}s -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
