"""Figures, charts and the cover of the post "Minds as computers" on xmarva.github.io.

Everything a figure shows is computed when it is drawn: the state graph of Alcuin's river puzzle by breadth-first
search, the McCulloch-Pitts units' truth tables by running them, the soundness of the two inference rules by checking
every interpretation over a two-person world. The colour charts read data/color.json (color_geometry.py).

    uv run --group articles python tools/render.py --article minds_as_computers \
        --out <site>/assets/posts/minds-as-computers
"""

import json
from collections import deque
from functools import cache
from itertools import product
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import Box, Figure

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- Alcuin's river crossing

CAST = "FWGC"  # farmer, wolf, goat, cabbage; 0 on the near bank, 1 on the far one


def safe(state):
    farmer, wolf, goat, cabbage = state
    return not (wolf == goat != farmer or goat == cabbage != farmer)


def crossings(state):
    """The farmer rows across alone or with one passenger from his bank."""
    farmer = state[0]
    for k in range(len(CAST)):
        if state[k] != farmer:
            continue
        nxt = list(state)
        nxt[0] = 1 - farmer
        nxt[k] = 1 - farmer
        if safe(tuple(nxt)):
            yield tuple(nxt)


@cache
def river():
    states = [s for s in product((0, 1), repeat=len(CAST)) if safe(s)]
    start, goal = (0,) * len(CAST), (1,) * len(CAST)
    depth, parents, queue = {start: 0}, {start: []}, deque([start])
    while queue:
        state = queue.popleft()
        for nxt in crossings(state):
            if nxt not in depth:
                depth[nxt] = depth[state] + 1
                parents[nxt] = [state]
                queue.append(nxt)
            elif depth[nxt] == depth[state] + 1:
                parents[nxt].append(state)
    edges = {tuple(sorted((s, t))) for s in states for t in crossings(s)}
    on_path, todo = set(), [goal]  # every edge of every shortest solution, walked back from the goal
    while todo:
        state = todo.pop()
        for parent in parents[state]:
            on_path.add(tuple(sorted((parent, state))))
            todo.append(parent)
    return states, depth, edges, on_path, start, goal


def banks(state):
    near = "".join(c for c, side in zip(CAST, state, strict=True) if side == 0)
    far = "".join(c for c, side in zip(CAST, state, strict=True) if side == 1)
    return f"{near or '·'} | {far or '·'}"


def river_graph() -> Figure:
    states, depth, edges, on_path, start, goal = river()
    columns = {}
    for s in sorted(states, key=lambda s: (depth[s], s)):
        columns.setdefault(depth[s], []).append(s)
    solutions = 1
    fig = Figure(
        760,
        330,
        alt=f"The {len(states)} safe arrangements of farmer, wolf, goat and cabbage out of {2 ** len(CAST)}, joined "
        f"by the crossings between them. Breadth-first search finds the far bank in {depth[goal]} crossings, by two "
        "routes that differ only in whether the wolf or the cabbage goes over first",
    )
    fig.text(14, 28, "Alcuin's river puzzle as a search through symbols", kind="title")
    fig.text(14, 50, "F farmer, W wolf, G goat, C cabbage; near bank | far bank. Every safe arrangement and every "
             "crossing", kind="note")
    w, h = 74, 32
    pos = {}
    for d, column in columns.items():
        x = 16 + d * 93
        for k, s in enumerate(column):
            y = 168 - (len(column) - 1) * 32 + k * 64
            pos[s] = (x, y)
    for a, b in sorted(edges):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        if xa > xb:
            (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
        hot = (a, b) in on_path
        fig.line((xa + w, ya + h / 2), (xb, yb + h / 2), role="data" if hot else "muted", width=2.2 if hot else 1.2)
    for s, (x, y) in pos.items():
        fig.els.append(Box(x, y, w, h, "data", banks(s), mono=True, strong=s in (start, goal), r=7, size=12))
    fig.text(pos[start][0] + w / 2, pos[start][1] - 10, "start", kind="label", anchor="middle")
    fig.text(pos[goal][0] + w / 2, pos[goal][1] - 10, "goal", kind="label", anchor="middle")
    for d in columns:
        fig.text(16 + d * 93 + w / 2, 268, str(d), kind="small", anchor="middle")
    fig.text(16, 288, "crossings made", kind="small")
    solutions = sum(1 for _ in _paths(start, goal, depth))
    fig.text(14, 318, f"{len(states)} of {2 ** len(CAST)} arrangements are safe; {solutions} shortest solutions, "
             f"{depth[goal]} crossings each", kind="count_hot")
    return fig


def _paths(state, goal, depth):
    if state == goal:
        yield [state]
        return
    for nxt in crossings(state):
        if depth.get(nxt) == depth[state] + 1:
            for rest in _paths(nxt, goal, depth):
                yield [state] + rest


# ---------------------------------------------------------------- McCulloch-Pitts units

def fires(weights, threshold, inputs):
    return int(sum(w * x for w, x in zip(weights, inputs, strict=True)) >= threshold)


UNITS = [("AND", (1, 1), 2), ("OR", (1, 1), 1), ("NOT", (-1,), 0)]


def mcculloch_pitts() -> Figure:
    fig = Figure(
        760,
        300,
        alt="Three McCulloch-Pitts units. Each adds its weighted inputs and fires if the sum reaches its threshold. "
        "Weights 1 and 1 with threshold 2 compute AND, with threshold 1 OR; weight -1 with threshold 0 computes NOT. "
        "The truth tables are computed by running the units",
    )
    fig.text(14, 28, "Logic out of threshold units", kind="title")
    fig.text(14, 50, "a unit fires (1) when the weighted sum of its inputs reaches its threshold θ; McCulloch & "
             "Pitts, 1943", kind="note")
    for k, (name, weights, threshold) in enumerate(UNITS):
        x0 = 14 + k * 250
        fig.box(x0, 70, 232, 214, role="plain", r=10)
        fig.text(x0 + 14, 96, name, kind="subtitle")
        cy = 140
        inputs = "ab"[: len(weights)]
        for j, (w, name_in) in enumerate(zip(weights, inputs, strict=True)):
            iy = cy - 16 + j * 32 if len(weights) == 2 else cy
            fig.box(x0 + 14, iy - 12, 28, 24, role="data", label=name_in, r=6)
            fig.line((x0 + 44, iy), (x0 + 112, cy), role="att", width=1.4)
            fig.text(x0 + 72, iy - 6 if j == 0 else iy + 16, f"{w:+d}", kind="small", anchor="middle")
        fig.box(x0 + 114, cy - 20, 60, 40, role="att", label=f"θ = {threshold}", r=20)
        fig.arrow((x0 + 176, cy), (x0 + 210, cy), role="att")
        rows = list(product((0, 1), repeat=len(weights)))
        header = "   ".join(inputs) + "   out"
        fig.text(x0 + 30, 196, header, kind="small")
        for r, row in enumerate(rows):
            out = fires(weights, threshold, row)
            line = "   ".join(str(v) for v in row) + f"    {out}"
            fig.text(x0 + 30, 216 + r * 18, line, kind="mono_att" if out else "mono")
    return fig


# ---------------------------------------------------------------- syntax that respects semantics

def holds(formula, world):
    """Truth of a formula of the example in a world: (people who are G, people who are F)."""
    g, f = world
    truth = {
        "Ga": "a" in g,
        "Fa": "a" in f,
        "Gb": "b" in g,
        "(Ga ∧ Fa)": "a" in g and "a" in f,
        "∃x (Gx ∧ Fx)": bool(g & f),
    }
    return truth[formula]


@cache
def worlds():
    """Every way to say which of two people, a and b, are G and which are F."""
    people = ("a", "b")
    subsets = [frozenset(p for p, keep in zip(people, mask, strict=True) if keep) for mask in product((0, 1), repeat=2)]
    return [(g, f) for g in subsets for f in subsets]


def counterexamples(premises, conclusion):
    return [w for w in worlds() if all(holds(p, w) for p in premises) and not holds(conclusion, w)]


def show(people):
    return "{" + ", ".join(sorted(people)) + "}"


def derivation() -> Figure:
    steps = [("Ga", "premise"), ("Fa", "premise"), ("(Ga ∧ Fa)", "∧-introduction"), ("∃x (Gx ∧ Fx)", "∃-introduction")]
    every = worlds()
    premises_true = [w for w in every if holds("Ga", w) and holds("Fa", w)]
    assert not counterexamples(["Ga", "Fa"], "∃x (Gx ∧ Fx)")
    bad = counterexamples(["∃x (Gx ∧ Fx)"], "Gb")
    fig = Figure(
        760,
        330,
        alt=f"A derivation in a language of thought: from Ga and Fa the rule of conjunction gives (Ga ∧ Fa), and "
        f"existential introduction gives ∃x (Gx ∧ Fx). The rules look only at the shapes of the formulas. Of all "
        f"{len(every)} ways to assign G and F to two people, the premises are true in {len(premises_true)}, and the "
        f"conclusion is true in each of them. A rule that is not sound, from ∃x (Gx ∧ Fx) to Gb, fails in "
        f"{len(bad)} worlds",
    )
    fig.text(14, 28, "Rules that see shapes, conclusions that stay true", kind="title")
    fig.text(14, 50, "G: is tall, F: has red hair, a: Georgina. The rules rewrite strings; truth is checked in "
             "worlds", kind="note")
    fig.text(14, 88, "syntax: what the machine does", kind="subtitle")
    for k, (formula, rule) in enumerate(steps):
        y = 104 + k * 46
        fig.chip(14, y, formula, role="att", strong=k == len(steps) - 1, size=14, h=32)
        fig.text(180, y + 21, rule, kind="label")
    fig.text(392, 88, "semantics: checked in every world", kind="subtitle")
    fig.text(392, 110, f"the premises hold in {len(premises_true)} of {len(every)} worlds", kind="small")
    for k, (g, f) in enumerate(premises_true):
        y = 122 + k * 30
        fig.text(392, y + 17, f"G = {show(g)}, F = {show(f)}", kind="mono")
        fig.chip(640, y, "true", role="enc", strong=True, h=24, size=12)
    fig.text(392, 262, "a rule that is not sound: ∃x (Gx ∧ Fx), therefore Gb", kind="small")
    g, f = bad[0]
    fig.text(392, 286, f"G = {show(g)}, F = {show(f)}", kind="mono")
    fig.chip(640, 268, "false", role="dec", strong=True, h=24, size=12)
    fig.text(392, 314, f"one of {len(bad)} worlds where it fails", kind="small")
    return fig


# ---------------------------------------------------------------- Marr's three levels

LEVELS = [
    ("computational theory", "what is computed, and why",
     "addition: the total must not depend on order or grouping",
     "a sequence of crossings that never leaves a pair alone"),
    ("representation and algorithm", "how: the format and the steps",
     "decimal digits, adding column by column with carries", "arrangements of the two banks; breadth-first search"),
    ("implementation", "what physically does it",
     "gears, or transistors", "Python on a laptop, or a child's neurons"),
]


def marr() -> Figure:
    fig = Figure(
        760,
        316,
        alt="Marr's three levels of explanation for two systems. Computational theory: a cash register adds because "
        "the total must not depend on order or grouping; the river puzzle needs a sequence of crossings that never "
        "leaves a dangerous pair alone. Representation and algorithm: decimal digits with carries; arrangements of "
        "the banks searched breadth-first. Implementation: gears or transistors; Python on a laptop or a child's "
        "neurons",
    )
    fig.text(14, 28, "Three questions about one computation", kind="title")
    fig.text(14, 50, "David Marr's levels of explanation (Vision, 1982), for his cash register and for Alcuin's puzzle",
             kind="note")
    fig.text(268, 84, "a cash register", kind="subtitle")
    fig.text(518, 84, "the river puzzle", kind="subtitle")
    for k, (level, question, register, puzzle) in enumerate(LEVELS):
        y = 98 + k * 70
        fig.box(14, y, 240, 60, role="att" if k < 2 else "neutral", r=8)
        fig.text(26, y + 26, level, kind="body")
        fig.text(26, y + 46, question, kind="small")
        for x, text in ((268, register), (518, puzzle)):
            fig.box(x, y, 236 if x == 518 else 240, 60, role="plain", r=8)
            for j, line in enumerate(_wrap(text, 29)):
                fig.text(x + 12, y + 26 + j * 18, line, kind="note")
    return fig


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line]


# ---------------------------------------------------------------- the Chinese Room

QUESTION, ANSWER = "你会说中文吗？", "会，说得很流利。"  # "Do you speak Chinese?" "Yes, fluently."


def chinese_room() -> Figure:
    fig = Figure(
        760,
        300,
        alt=f"Searle's Chinese Room. A question in Chinese, {QUESTION} ('Do you speak Chinese?'), comes in through "
        f"a slot. Inside, a man who knows no Chinese finds its shape in a rule book written in English and sends back "
        f"the shapes the book prescribes, {ANSWER} ('Yes, fluently.'). Outside, the room seems to understand Chinese",
    )
    fig.text(14, 28, "The Chinese Room", kind="title")
    fig.text(14, 50, "Searle, 1980: the room answers like a native speaker; the man inside matches shapes", kind="note")
    fig.text(14, 98, "in through the slot", kind="label")
    fig.box(14, 108, 150, 36, role="data", label=QUESTION, r=8)
    fig.text(14, 162, "\"Do you speak Chinese?\"", kind="small")
    fig.arrow((168, 126), (204, 126), role="data")
    fig.box(208, 76, 344, 196, role="neutral", r=14)
    fig.text(224, 100, "the room", kind="subtitle")
    fig.box(224, 112, 312, 64, role="plain", r=8)
    fig.text(236, 134, "rule book, in English:", kind="small")
    fig.text(236, 158, f"if you get {QUESTION}", kind="mono")
    fig.box(224, 184, 312, 36, role="plain", r=8)
    fig.text(236, 207, f"send back {ANSWER}", kind="mono")
    fig.text(224, 244, "a man who knows no Chinese,", kind="note")
    fig.text(224, 262, "matching shapes against the book", kind="note")
    fig.arrow((556, 202), (592, 202), role="enc")
    fig.text(596, 170, "out through the slot", kind="label")
    fig.box(596, 184, 150, 36, role="enc", label=ANSWER, r=8)
    fig.text(596, 238, "\"Yes, fluently.\"", kind="small")
    return fig


# ---------------------------------------------------------------- colour from text

@cache
def color():
    return json.loads((DATA / "color.json").read_text(encoding="utf-8"))


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, 0.905, note, ha="left", va="top", fontsize=10, color=c["muted"])


LABELLED = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "teal", "navy", "lime", "brown", "cyan")


def chart_color_map(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = color()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.03, wspace=0.08)
    panels = [("seen: the colours in CIELAB", d["lab_ab"]), (f"read: the model's layer {d['layer']}", d["model_ab"])]
    for ax, (title, points) in zip(axes, panels, strict=True):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.scatter(xs, ys, c=d["hex"], s=70, edgecolors=c["line"], linewidths=0.8, zorder=3)
        for x, y, word in zip(xs, ys, d["words"], strict=True):
            if word in LABELLED:
                ax.text(x, y + 0.014, word, fontsize=8.5, ha="center", va="bottom", color=c["muted"])
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
    _caption(fig, "A model that has never seen a colour",
             f"50 colour names: their colours (left) and {d['model'].split('/')[1]}'s states for them (right)", c)
    return fig


def chart_color_layers(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = color()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.74, bottom=0.17)
    layers = list(range(len(d["rho_by_layer"])))
    ax.plot(layers, d["rho_by_layer"], color=c["att"], marker="o", markersize=4)
    ax.axhline(d["null_99"], color=c["muted"], linestyle="--", linewidth=1.2)
    ax.text(layers[-1], d["null_99"] + 0.012, "99% of shuffled words fall below", fontsize=9.5, color=c["muted"],
            ha="right", va="bottom")
    ax.set_xlabel("layer (0: the embedding)")
    ax.set_ylabel("Spearman's ρ")
    ax.set_ylim(0, 0.5)
    _caption(fig, "Colour geometry, layer by layer",
             "how well distances between the model's states for 50 colour names follow distances in CIELAB", c)
    return fig


def chart_alts():
    d = color()
    return {
        "color-map": f"Fifty colour names placed by their colours in CIELAB and by the hidden states of "
        f"{d['model'].split('/')[1]} at layer {d['layer']}, rotated and scaled onto CIELAB. Blues, purples and reds "
        f"keep their neighbourhoods; the agreement of the two sets of distances is Spearman's rho = {d['rho']:.2f}",
        "color-layers": f"Spearman's rho between the model's distances and CIELAB distances, by layer: about 0.1 at "
        f"the embedding, between 0.3 and 0.4 from layer 6 on, against {d['null_99']:.2f} for shuffled words",
    }


# ---------------------------------------------------------------- cover

def cover():
    states, depth, edges, on_path, start, goal = river()
    out = [cv.label(cv.MOTIF_X0, 128, "ALCUIN'S PUZZLE, SEARCHED")]
    columns = {}
    for s in sorted(states, key=lambda s: (depth[s], s)):
        columns.setdefault(depth[s], []).append(s)
    pos = {}
    for d, column in columns.items():
        x = cv.MOTIF_X0 + 18 + d * 68
        for k, s in enumerate(column):
            pos[s] = (x, 250 - (len(column) - 1) * 40 + k * 80)
    for a, b in sorted(edges):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        color = cv.RED if (a, b) in on_path else cv.LINE
        out.append(f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" stroke="{color}" stroke-width="2.5"/>')
    for s, (x, y) in pos.items():
        fill = cv.RED if s in (start, goal) else cv.BG
        out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{fill}" stroke="{cv.RED}" stroke-width="2.5"/>')
    cv.fits(max(x for x, _ in pos.values()) + 11)
    out.append(cv.label(cv.MOTIF_X0, 400, "THE SAME RULES, TWO READINGS"))
    for row, (text, kind) in enumerate([("(Ga ∧ Fa)", "plain"), ("∃x (Gx ∧ Fx)", "hot")]):
        chip, w = cv.chip(cv.MOTIF_X0, 420 + row * 56, text, size=18, kind=kind)
        out.append(chip)
    return cv.frame("blog / cognitive science", ["Minds as", "computers"], ["Symbols, rules and", "what they mean"],
                    "".join(out), footer="Philosophy of mind, computed")


COVER = cover

FIGURES = {
    "river": river_graph,
    "mcculloch-pitts": mcculloch_pitts,
    "derivation": derivation,
    "marr": marr,
    "chinese-room": chinese_room,
}

CHARTS = {
    "color-map": chart_color_map,
    "color-layers": chart_color_layers,
}
