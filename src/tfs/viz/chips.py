"""Draw token sequences as rows of chips, in the course style, from data computed in the notebook."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .palette import palette
from .style import MONO_STACK, register_fonts

ROLES = ("data", "enc", "dec", "att")


def token_chips(
    rows: Sequence[Sequence[str]],
    *,
    labels: Sequence[str] | None = None,
    ids: Sequence[Sequence[int]] | None = None,
    strong: Callable[[int, int, str], bool] | None = None,
    role: str = "data",
    mark_spaces: bool = True,
    theme: str = "light",
    size: float = 12,
    max_width: float = 540,
    title: str | None = None,
):
    """Draw each sequence of tokens as a row of chips and return the figure.

    rows       token sequences, one per row, e.g. the same word before and after a BPE merge
    labels     optional label for each row, drawn on the left
    ids        optional token ids, one sequence per row, drawn under the chips
    strong     predicate (row, column, token) -> bool that marks chips to emphasize
    role       concept color: "data" (tokens), "enc", "dec" or "att"
    mark_spaces  draw spaces inside tokens as ␣; turn off for chips that hold whole sentences
    max_width  in points; a longer row continues on the next line, so the chips never shrink to fit a cell
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if ids is not None and [len(r) for r in ids] != [len(r) for r in rows]:
        raise ValueError("ids must have one id per token in every row")
    register_fonts()
    c = palette(theme)
    char = 0.6 * size  # IBM Plex Mono advance width is 0.6 em
    pad, gap, height = 0.55 * size, 0.35 * size, 1.9 * size
    line_step = height + 0.7 * size + (1.3 * size if ids is not None else 0)  # ids go under the chips
    row_gap = 0.6 * size  # extra air between rows, so the lines of one wrapped row stay together
    id_size, label_size = 0.8 * size, 0.85 * size
    label_w = (max(len(s) for s in labels) * 0.6 * label_size + 2 + 2 * gap) if labels else 0.0

    def shown(token: str) -> str:
        return token.replace(" ", "␣") if mark_spaces else token  # make spaces inside tokens visible

    def chip_w(r: int, col: int) -> float:
        w = len(shown(rows[r][col])) * char + 2 * pad
        if ids is not None:  # an id longer than its token widens the chip, so ids never collide
            w = max(w, len(str(ids[r][col])) * 0.6 * id_size + 0.6 * size)
        return w

    # break every row into lines that fit: (row, first column, last column + 1)
    lines = []
    for r, row in enumerate(rows):
        start, used = 0, 0.0
        for col in range(len(row)):
            w = chip_w(r, col)
            if col > start and label_w + used + w > max_width:
                lines.append((r, start, col))
                start, used = col, 0.0
            used += w + gap
        lines.append((r, start, len(row)))

    width = label_w + 2 + max(sum(chip_w(r, k) + gap for k in range(a, b)) - gap for r, a, b in lines)
    top = 1.6 * size if title else 0.0
    new_rows = sum(1 for k in range(1, len(lines)) if lines[k][0] != lines[k - 1][0])
    total_h = top + line_step * len(lines) - 0.7 * size + row_gap * new_rows + 2

    fig = plt.figure(figsize=(width / 72, total_h / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(total_h, 0)
    ax.axis("off")
    if title:
        ax.text(0, size * 1.0, title, fontsize=size, fontweight="semibold", color=c["ink"], va="baseline")
    y = top + 1
    for k, (r, a, b) in enumerate(lines):
        if k > 0 and r != lines[k - 1][0]:
            y += row_gap
        if labels and a == 0:
            ax.text(2, y + height / 2, labels[r], fontsize=label_size, family=MONO_STACK, color=c["muted"], va="center")
        x = label_w + 1
        for col in range(a, b):
            token, w = rows[r][col], chip_w(r, col)
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
                family=MONO_STACK,
                color=c["card"] if hot else c[f"{role}_ink"],
                ha="center",
                va="center",
                fontweight="medium",
            )
            if ids is not None:
                ax.text(
                    x + w / 2,
                    y + height + 0.95 * size,
                    str(ids[r][col]),
                    fontsize=id_size,
                    family=MONO_STACK,
                    color=c["muted"],
                    ha="center",
                    va="center",
                )
            x += w + gap
        y += line_step
    return fig
