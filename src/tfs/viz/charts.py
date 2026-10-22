"""Two charts the lessons draw again and again: a matrix of weights with token labels, and labelled bars."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .palette import palette
from .style import MONO_STACK


def heatmap(
    matrix,
    rows=None,
    cols=None,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    role="att",
    low="card",
    vmin=0.0,
    vmax=None,
    annotate=False,
    fmt="{:.2f}",
    cell=0.3,
    theme="light",
):
    """Draw a matrix (attention weights, a mask) as a grid of cells, one hue from white to the concept color.

    rows, cols  labels, e.g. the query and key tokens; columns are labelled at the bottom, turned upright
    low         palette color of the lowest value: "card" (white) for weights, "grid" (grey) for a mask
    annotate    write every value in its cell
    cell        size of one cell in inches
    """
    c = palette(theme)
    values = np.asarray(matrix, dtype=float)
    n_rows, n_cols = values.shape
    colors = LinearSegmentedColormap.from_list("course", [c[low], c[role]])
    label_room = 0.11 * max([len(str(label)) for label in (rows or [])] + [3])
    fig_w = cell * n_cols + label_room + 0.3
    fig_h = cell * n_rows + (0.11 * max([len(str(label)) for label in (cols or [])] + [3]) if cols else 0.2) + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.pcolormesh(values, cmap=colors, vmin=vmin, vmax=vmax, edgecolors=c["card"], linewidth=1.2)
    ax.invert_yaxis()  # the first row on top, like a printed matrix
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if rows is not None:
        ax.set_yticks(np.arange(n_rows) + 0.5, rows, family=MONO_STACK, fontsize=9)
    else:
        ax.set_yticks([])
    if cols is not None:
        ax.set_xticks(np.arange(n_cols) + 0.5, cols, family=MONO_STACK, fontsize=9, rotation=90)
    else:
        ax.set_xticks([])
    if annotate:
        top = vmax if vmax is not None else values.max()
        for r in range(n_rows):
            for k in range(n_cols):
                dark = values[r, k] > 0.6 * top
                ax.text(k + 0.5, r + 0.5, fmt.format(values[r, k]), ha="center", va="center", fontsize=8,
                        color=c["card"] if dark else c["ink"])
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig


def bars(values, *, colors=None, title=None, fmt="{:,.0f}", theme="light", width=6.4):
    """Horizontal bars for a {label: number} dict, first item on top, each bar labelled with its value."""
    c = palette(theme)
    labels, numbers = list(values), list(values.values())
    fig, ax = plt.subplots(figsize=(width, 0.6 + 0.42 * len(labels)))
    drawn = ax.barh(labels, numbers, color=colors or c["data"], height=0.66)
    ax.bar_label(drawn, labels=[fmt.format(n) for n in numbers], padding=4, fontsize=9, color=c["muted"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(numbers) * 1.28)
    ax.set_xticks([])  # every bar carries its own number
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    if title:
        ax.set_title(title)
    return fig
