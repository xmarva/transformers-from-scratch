"""Build the notebooks of every lesson from its single source.

A lesson lives in lessons/NN_name/ (the first course) or architectures/aNN_name/ (the second, tools/courses.py) as
    lesson.py    the lesson in jupytext "percent" format: the only file you edit
    lesson.toml  title, number and runtime needs (GPU, internet)

Cell tags in lesson.py control the targets:
    include:<name>   replaced by the cells of <course folder>/_shared/<name>.py
    only-kaggle      kept only in the Kaggle notebook (likewise only-colab)

Outputs, committed so that Colab can open them straight from GitHub and Kaggle can receive them:
    notebooks/colab/NN_name.ipynb
    notebooks/kaggle/NN_name/notebook.ipynb + kernel-metadata.json

Usage:
    uv run python tools/build.py            # build every lesson
    uv run python tools/build.py 01         # build lessons whose folder starts with 01
    uv run python tools/build.py --check    # fail if the committed notebooks are out of date
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import jupytext
import nbformat
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from courses import COURSES, Course, course_of, lesson_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "xmarva/transformers-from-scratch"
RAW = f"https://raw.githubusercontent.com/{REPO}"
KAGGLE_USER = "qmarva"
TARGETS = ("kaggle", "colab")
BUILD_TAGS = ("only-kaggle", "only-colab")


KAGGLE_TITLE_MAX = 50  # Kaggle refuses longer notebook titles


def kaggle_title(meta: dict, course: Course = COURSES[0]) -> str:
    title = f"{course.kaggle} · {meta['number']:02d} {meta['short_title']}"
    if len(title) > KAGGLE_TITLE_MAX:
        raise ValueError(f"Kaggle title {title!r} has {len(title)} characters, at most {KAGGLE_TITLE_MAX} allowed")
    return title


def kaggle_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def links(lesson: str, meta: dict, course: Course = COURSES[0]) -> dict[str, str]:
    return {
        "colab": f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/colab/{lesson}.ipynb",
        "kaggle": f"https://www.kaggle.com/code/{KAGGLE_USER}/{kaggle_slug(kaggle_title(meta, course))}",
        "github": f"https://github.com/{REPO}",
    }


def header_cell(lesson: str, meta: dict, target: str, ref: str, course: Course = COURSES[0]) -> nbformat.NotebookNode:
    url = links(lesson, meta, course)
    badge = {
        "colab": f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url['colab']})",
        "kaggle": f"[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({url['kaggle']})",
    }
    other = "colab" if target == "kaggle" else "kaggle"
    lines = [
        f"![{meta['cover_alt']}]({RAW}/{ref}/assets/covers/{lesson.split('_', 1)[0]}-en.png)",
        "",
        f"**{course.title}** · Lesson {meta['number']} of {course.lessons} · [all lessons and code]({url['github']}) "
        f"&nbsp; {badge[other]}",
    ]
    return nbformat.v4.new_markdown_cell("\n".join(lines))


def read_cells(path: Path) -> list[nbformat.NotebookNode]:
    return jupytext.read(path, fmt="py:percent").cells


def expand_includes(cells: list, shared: Path) -> list:
    out = []
    for cell in cells:
        includes = [t.split(":", 1)[1] for t in cell.metadata.get("tags", []) if t.startswith("include:")]
        if includes:
            for name in includes:
                out.extend(copy.deepcopy(read_cells(shared / f"{name}.py")))
        else:
            out.append(cell)
    return out


def for_target(cells: list, target: str, ref: str) -> list:
    out = []
    for cell in cells:
        tags = cell.metadata.get("tags", [])
        if any(t.startswith("only-") and t != f"only-{target}" for t in tags):
            continue
        cell = copy.deepcopy(cell)
        kept = [t for t in tags if t not in BUILD_TAGS]
        if kept:
            cell.metadata["tags"] = kept
        else:
            cell.metadata.pop("tags", None)
        # helpers and images come from the same git ref: main for releases, a branch for drafts
        cell.source = cell.source.replace("{{TFS_REF}}", ref).replace(f"{RAW}/main/", f"{RAW}/{ref}/")
        out.append(cell)
    return out


def notebook(cells: list, meta: dict, target: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if target == "colab":
        nb.metadata["colab"] = {"provenance": []}
        if meta.get("gpu"):
            nb.metadata["accelerator"] = "GPU"
            nb.metadata["colab"]["gpuType"] = "T4"
    for i, cell in enumerate(nb.cells):  # deterministic ids keep diffs and --check stable
        cell["id"] = f"cell-{i:03d}"
    return nb


def kernel_metadata(lesson: str, meta: dict, course: Course = COURSES[0]) -> dict:
    title = kaggle_title(meta, course)
    return {
        "id": f"{KAGGLE_USER}/{kaggle_slug(title)}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": not meta.get("published", False),
        "enable_gpu": bool(meta.get("gpu")),
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_lesson(lesson_dir: Path, out_dir: Path, ref: str) -> dict[Path, str]:
    """Return {output path: file content} for one lesson."""
    meta = tomllib.loads((lesson_dir / "lesson.toml").read_text(encoding="utf-8"))
    lesson = lesson_dir.name
    course = course_of(lesson_dir)
    cells = expand_includes(read_cells(lesson_dir / "lesson.py"), lesson_dir.parent / "_shared")
    files = {}
    for target in TARGETS:
        nb = notebook([header_cell(lesson, meta, target, ref, course), *for_target(cells, target, ref)], meta, target)
        text = nbformat.writes(nb) + "\n"
        if target == "colab":
            files[out_dir / "colab" / f"{lesson}.ipynb"] = text
        else:
            files[out_dir / "kaggle" / lesson / "notebook.ipynb"] = text
            files[out_dir / "kaggle" / lesson / "kernel-metadata.json"] = (
                json.dumps(kernel_metadata(lesson, meta, course), indent=2, ensure_ascii=False) + "\n"
            )
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lessons", nargs="*", help="lesson folder prefixes, e.g. 01 or a01; default: all")
    parser.add_argument("--check", action="store_true", help="fail if committed notebooks differ from a fresh build")
    parser.add_argument("--ref", default="main", help="git ref for the tfs helpers and images (a branch for drafts)")
    parser.add_argument("--lessons-dir", type=Path, help="build the lessons of this folder instead of every course")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args(argv)

    if args.lessons_dir:
        dirs = sorted(d for d in args.lessons_dir.iterdir() if d.is_dir() and (d / "lesson.py").exists())
    else:
        dirs = [d for d in lesson_dirs() if (d / "lesson.py").exists()]
    if args.lessons:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in args.lessons)]
    stale = []
    for d in dirs:
        for path, text in build_lesson(d, args.out_dir, args.ref).items():
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != text:
                    stale.append(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                print(f"built {path.relative_to(args.out_dir.parent)}")
    if stale:
        print("Out of date, run `uv run python tools/build.py`:", *stale, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
