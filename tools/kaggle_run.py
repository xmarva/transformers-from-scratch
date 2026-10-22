"""Run a lesson in full on a Kaggle GPU, from a git branch, and download the executed notebook.

Lessons that train need a GPU, which CI and most laptops don't have. This pushes a private Kaggle script that
installs the branch's helpers, executes the branch's Kaggle notebook top to bottom with nbclient, and saves the
executed copy; then waits for it and downloads the result to build/executed/<lesson>.ipynb.

Usage (the branch must be pushed first):
    uv run python tools/kaggle_run.py 03_training --branch lesson-03

Needs the Kaggle CLI credentials in ~/.kaggle, as for publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "xmarva/transformers-from-scratch"
KERNEL = "qmarva/tfs-lesson-runner"

RUNNER = '''
import subprocess, sys, time, urllib.request

BRANCH, LESSON, REPO = {branch!r}, {lesson!r}, {repo!r}
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/{{REPO}}@{{BRANCH}}"])
import nbformat
from nbclient import NotebookClient

url = f"https://raw.githubusercontent.com/{{REPO}}/{{BRANCH}}/notebooks/kaggle/{{LESSON}}/notebook.ipynb"
nb = nbformat.reads(urllib.request.urlopen(url).read().decode(), as_version=4)
for cell in nb.cells:  # the committed notebook points at main; run the branch's helpers and figures
    cell.source = cell.source.replace(f"{{REPO}}@main", f"{{REPO}}@{{BRANCH}}")
    cell.source = cell.source.replace(f"{{REPO}}/main/", f"{{REPO}}/{{BRANCH}}/")
start = time.time()
try:
    resources = {{"metadata": {{"path": "/kaggle/working"}}}}
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources=resources).execute()
finally:
    nbformat.write(nb, "/kaggle/working/executed.ipynb")
    print(f"ran in {{(time.time() - start) / 60:.1f}} min", flush=True)
'''


def kaggle(*args: str) -> str:
    result = subprocess.run(["uvx", "kaggle", *args], capture_output=True, text=True, check=True)
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lesson", help="lesson folder name, e.g. 03_training")
    parser.add_argument("--branch", required=True, help="pushed git branch to run")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "run_lesson.py").write_text(RUNNER.format(branch=args.branch, lesson=args.lesson, repo=REPO))
        metadata = {
            "id": KERNEL,
            "title": "tfs lesson runner",
            "code_file": "run_lesson.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        print(kaggle("kernels", "push", "-p", str(folder)).strip())

    while True:
        time.sleep(60)
        status = kaggle("kernels", "status", KERNEL)
        if any(word in status for word in ("COMPLETE", "ERROR", "CANCEL")):
            print(status.strip())
            break

    with tempfile.TemporaryDirectory() as tmp:
        kaggle("kernels", "output", KERNEL, "-p", tmp, "--file-pattern", "executed.ipynb")
        out = ROOT / "build" / "executed" / f"{args.lesson}.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(tmp) / "executed.ipynb", out)
        print(f"executed notebook -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
