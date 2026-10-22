"""Matplotlib style of the course: the same fonts and concept colors as the static figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from .palette import palette

FONTS_DIR = Path(__file__).parent / "fonts"
SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"
# matplotlib takes a missing glyph (␣, �) from the next family in the list; DejaVu ships with matplotlib
SANS_STACK = [SANS, "DejaVu Sans"]
MONO_STACK = [MONO, "DejaVu Sans Mono"]


def register_fonts() -> None:
    """Make the bundled IBM Plex fonts (Latin + Cyrillic) available to matplotlib."""
    # a fallback font without a medium weight is fine; don't print a warning about it into every notebook
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    known = {f.fname for f in font_manager.fontManager.ttflist}
    for path in FONTS_DIR.glob("*.ttf"):
        if str(path) not in known:
            font_manager.fontManager.addfont(str(path))


def use_course_style(theme: str = "light") -> dict[str, str]:
    """Apply the course style to every following plot and return the palette, e.g. `c = use_course_style()`."""
    register_fonts()
    c = palette(theme)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "figure.figsize": (6.4, 3.8),
            "font.family": SANS_STACK,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": c["muted"],
            "axes.edgecolor": c["grid"],
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": c["grid"],
            "grid.linewidth": 1,
            "axes.prop_cycle": plt.cycler(color=[c["enc"], c["dec"], c["att"], c["data"]]),
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": c["ink"],
            "axes.titlecolor": c["ink"],
            "figure.facecolor": c["card"],
            "axes.facecolor": c["card"],
            "savefig.facecolor": c["card"],
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "svg.fonttype": "path",
        }
    )
    return c
