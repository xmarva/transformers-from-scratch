import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.figures import cover  # noqa: E402
from tools.figures.lib import Figure  # noqa: E402


def small_figure():
    fig = Figure(320, 120, alt="Two tokens merge into one")
    spans = fig.row(["н", "и"], 20, 20)
    fig.arrow((60, 56), (60, 76))
    fig.chip(20, 80, "ни", strong=True)
    fig.text(20 + spans[-1][0], 14, "pair", kind="label")
    return fig


def test_clean_figure_has_no_problems_and_renders_both_themes():
    fig = small_figure()
    assert fig.problems() == []
    light, dark = fig.to_svg("light"), fig.to_svg("dark")
    assert light != dark and "Two tokens merge into one" in light


def test_problems_catch_overflow_and_overlapping_labels():
    fig = small_figure()
    fig.chip(300, 20, "overflowing")
    fig.text(20, 100, "first label")
    fig.text(30, 100, "second label")
    found = "\n".join(fig.problems())
    assert "outside canvas" in found and "text overlap" in found


def test_english_cover_rejects_cyrillic():
    cover.check_latin(cover.frame("01_x.ipynb", ["Title"], ["hook"], ""))
    with pytest.raises(ValueError, match="Cyrillic"):
        cover.check_latin(cover.frame("01_x.ipynb", ["Токенизация"], ["hook"], ""))


def test_cover_has_no_author_name_and_motif_bounds_are_enforced():
    svg = cover.frame("01_x.ipynb", ["Title"], ["hook"], "")
    assert "Koroleva" not in svg
    cover.fits(cover.MOTIF_X1)
    with pytest.raises(ValueError):
        cover.fits(cover.MOTIF_X1 + 1)
