import ast
import subprocess
from pathlib import Path

import jupytext
import pytest
import tomllib

LESSONS = Path(__file__).resolve().parents[1] / "lessons"
SOURCES = sorted(LESSONS.glob("*/lesson.py")) + sorted((LESSONS / "_shared").glob("*.py"))
MAX_CODE_LINE = 92  # what a Kaggle or Colab cell shows without a horizontal scrollbar


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_code_lines_fit_a_notebook_cell(path):
    cells = jupytext.read(path, fmt="py:percent").cells
    long = [
        line
        for cell in cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if len(line) > MAX_CODE_LINE
    ]
    assert not long, "\n".join(long)


def files_written(path):
    """The files a lesson writes: `%%writefile` cells (`-a` appends) and `write("path", text)` calls."""
    files = {}
    for cell in jupytext.read(path, fmt="py:percent").cells:
        if cell.cell_type != "code":
            continue
        first, _, body = cell.source.partition("\n")
        if first.startswith("%%writefile"):
            name = first.split()[-1]
            if "-a" in first.split():
                files[name] += body + "\n"
            else:
                files[name] = body + "\n"
            continue
        python = []
        for line in cell.source.splitlines():
            if not line.lstrip().startswith(("!", "%")):  # shell commands and magics aren't Python
                python.append(line)
        for node in ast.walk(ast.parse("\n".join(python))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "write":
                name, text = node.args[0].value, node.args[1].value
                files[name] = text.lstrip("\n")
    return files


def released_file(release, name):
    """The file as the package's release tag has it, or as the working tree has it before the tag exists."""
    root = LESSONS.parent
    tagged = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{release}"], cwd=root,
                            capture_output=True)
    if tagged.returncode != 0:
        return (root / name).read_text(encoding="utf-8")
    shown = subprocess.run(["git", "show", f"{release}:{name}"], cwd=root, capture_output=True, check=True)
    return shown.stdout.decode("utf-8")


PACKAGE_LESSONS = []
for toml_path in sorted(LESSONS.glob("*/lesson.toml")):
    if "package_release" in tomllib.loads(toml_path.read_text(encoding="utf-8")):
        PACKAGE_LESSONS.append(toml_path.parent.name)


@pytest.mark.parametrize("lesson", PACKAGE_LESSONS)
def test_a_lesson_writes_the_package_it_releases(lesson):
    release = tomllib.loads((LESSONS / lesson / "lesson.toml").read_text(encoding="utf-8"))["package_release"]
    written = files_written(LESSONS / lesson / "lesson.py")
    package_files = [name for name in written if name.startswith(("perevod/", ".github/"))]
    assert package_files
    for name in package_files:
        assert written[name] == released_file(release, name), name
