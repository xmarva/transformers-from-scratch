"""Copy the text of a rebuilt lesson into its executed notebook, when only the text changed since the run.

A full run of a lesson on Kaggle can take an hour; a sentence rewritten after looking at its outputs shouldn't cost
another. This takes the markdown cells of the freshly built Kaggle notebook and puts them into
build/executed/<lesson>.ipynb, and refuses if any code cell differs: then the outputs would no longer belong to the
code, and the lesson has to run again.

    uv run python tools/build.py a01 && uv run python tools/refresh_text.py a01_bert
"""

import argparse
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lesson", help="lesson folder name, e.g. a01_bert")
    args = parser.parse_args(argv)
    built = nbformat.read(ROOT / "notebooks" / "kaggle" / args.lesson / "notebook.ipynb", as_version=4)
    path = ROOT / "build" / "executed" / f"{args.lesson}.ipynb"
    executed = nbformat.read(path, as_version=4)
    if len(built.cells) != len(executed.cells):
        sys.exit(f"{len(built.cells)} cells built, {len(executed.cells)} executed: run the lesson again")
    changed = 0
    for new, old in zip(built.cells, executed.cells, strict=True):
        if new.cell_type != old.cell_type:
            sys.exit(f"cell {new.get('id')}: {new.cell_type} built, {old.cell_type} executed: run the lesson again")
        if new.cell_type == "code" and new.source != old.source:
            sys.exit(f"cell {new.get('id')}: the code changed since the run: run the lesson again")
        if new.cell_type == "markdown" and new.source != old.source:
            old.source = new.source
            changed += 1
    nbformat.write(executed, path)
    print(f"{changed} markdown cells updated in {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
