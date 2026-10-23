"""Draw token sequences as rows of chips, in the course style, from data computed in the notebook."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .palette import palette
from .style import MONO, register_fonts

ROLES = ("data", "enc", "dec", "att")


def token_chips(
    rows: Sequence[Sequence[str]],
    *,
    labels: Sequence[str] | None = None,
    strong: Callable[[int, int, str], bool] | None = None,
    role: str = "data",
    theme: str = "light",
    size: float = 12,
    title: str | None = None,
):
    """Draw each sequence of tokens as a row of chips and return the figure.

    rows    token sequences, one per row, e.g. the same word before and after a BPE merge
    labels  optional label for each row, drawn on the left
    strong  predicate (row, column, token) -> bool that marks chips to emphasize
    role    concept color: "data" (tokens), "enc", "dec" or "att"
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    register_fonts()
    c = palette(theme)
    char = 0.6 * size  # IBM Plex Mono advance width is 0.6 em
    pad, gap, height, step = 0.55 * size, 0.35 * size, 1.9 * size, 2.6 * size
    label_w = (max(len(s) for s in labels) * char + 2 * gap) if labels else 0.0

    def shown(token: str) -> str:
        return token.replace(" ", "␣")  # make spaces inside tokens visible

    def chip_w(token: str) -> float:
        return len(shown(token)) * char + 2 * pad

    width = label_w + max(sum(chip_w(t) for t in row) + gap * (len(row) - 1) for row in rows) + 2
    top = 1.6 * size if title else 0.0
    total_h = top + step * (len(rows) - 1) + height + 2

    fig = plt.figure(figsize=(width / 72, total_h / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(total_h, 0)
    ax.axis("off")
    if title:
        ax.text(0, size * 1.0, title, fontsize=size, fontweight="semibold", color=c["ink"], va="baseline")
    for r, row in enumerate(rows):
        y = top + r * step + 1
        if labels:
            ax.text(0, y + height / 2, labels[r], fontsize=size * 0.85, family=MONO, color=c["muted"], va="center")
        x = label_w + 1
        for col, token in enumerate(row):
            w = chip_w(token)
            hot = bool(strong and strong(r, col, token))
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    height,
                    boxstyle=f"round,pad=0,rounding_size={0.35 * size}",
                    facecolor=c[role] if hot else c[f"{role}_tint"],
                    edgecolor=c[role],
                    linewidth=1.2,
                )
            )
            ax.text(
                x + w / 2,
                y + height / 2,
                shown(token),
                fontsize=size,
                family=MONO,
                color=c["card"] if hot else c[f"{role}_ink"],
                ha="center",
                va="center",
                fontweight="medium",
            )
            x += w + gap
    return fig
