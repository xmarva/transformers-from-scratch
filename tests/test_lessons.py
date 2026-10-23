from pathlib import Path

import jupytext
import pytest

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
