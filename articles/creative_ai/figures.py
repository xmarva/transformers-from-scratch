"""Figures, charts and the cover of the post "Can AI be creative?" on xmarva.github.io.

Brains are fsaverage6 cortex coloured by Yeo et al.'s networks in Schaefer et al.'s 400-parcel atlas
(tools/figures/brain.py). The colour diagram is computed from the CIE 2015 2-degree observer and the Stockman & Sharpe
cone fundamentals as the colour-science package tabulates them.

    uv run --group articles python tools/render.py --article creative_ai --out <site>/assets/posts/creative-ai
"""

import json
import re
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import brain
from tools.figures import cover as cv
from tools.figures.lib import Box, Figure

DATA = Path(__file__).parent / "data"


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


def _brain_greys(theme):
    return ("#E9ECF1", "#BCC2CC") if theme == "light" else ("#4A5160", "#2E343E")


# ---------------------------------------------------------------- the networks of creative thought

NETWORKS = [  # (Yeo network in the atlas, name in the post, palette role)
    ("Default", "default mode", "dec"),
    ("Cont", "executive control", "enc"),
    ("SalVentAttn", "salience", "data"),
]


def chart_networks(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = _style(theme)
    labels = brain.schaefer_networks("left")
    colors = {network: c[role] for network, _, role in NETWORKS}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.8, bottom=0.2, wspace=0.0)
    for ax, view in zip(axes, ("lateral", "medial"), strict=True):
        brain.draw(ax, "left", view, labels, colors, base=_brain_greys(theme))
        ax.text(0.5, -0.02, f"left hemisphere, {view}", transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=c["muted"])
    handles = [Patch(facecolor=c[role], label=name) for _, name, role in NETWORKS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=10)
    _caption(fig, "Three networks that work together in creative thought",
             "Yeo et al.'s networks in the Schaefer 400-parcel atlas, on an inflated fsaverage cortex", c)
    return fig


# ---------------------------------------------------------------- a colour no light can make

@cache
def macleod_boynton():
    """The spectral locus and the cone-only points in the MacLeod-Boynton diagram of the CIE 2015 2-degree observer.

    l = L / (L + M) and s = S / (L + M), with L and M weighted as in the observer's luminance, Y = 0.68990 L + 0.34832 M
    (CIE 170-2; the colour-science tables reproduce it exactly), and s scaled to 1 at its largest spectral value."""
    import colour

    cones = colour.colorimetry.MSDS_CMFS_LMS["Stockman & Sharpe 2 Degree Cone Fundamentals"]
    xyz = colour.MSDS_CMFS["CIE 2015 2 Degree Standard Observer"]
    wavelengths = [w for w in cones.wavelengths if w in set(xyz.wavelengths) and 390 <= w <= 700]
    lms = np.array([cones[w] for w in wavelengths])
    target = np.array([xyz[w] for w in wavelengths])
    to_xyz, *_ = np.linalg.lstsq(lms, target, rcond=None)  # the observer's own LMS -> XYZ matrix
    wl, wm = to_xyz[0, 1], to_xyz[1, 1]  # Y row: luminance from L and M

    def point(v):
        lum = wl * v[..., 0] + wm * v[..., 1]
        return wl * v[..., 0] / lum, v[..., 2] / lum

    l_locus, s_locus = point(lms)
    scale = s_locus.max()
    return np.array(wavelengths), l_locus, s_locus / scale, scale, (wl, wm)


def chart_olo(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    wavelengths, l_axis, s_axis, _, _ = macleod_boynton()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.15)
    ax.fill(np.r_[l_axis, l_axis[0]], np.r_[s_axis, s_axis[0]], color=c["att_tint"], zorder=1)
    ax.plot(l_axis, s_axis, color=c["att"], linewidth=2, zorder=2)
    ax.plot([l_axis[-1], l_axis[0]], [s_axis[-1], s_axis[0]], color=c["att"], linewidth=1.2, linestyle="--", zorder=2)
    for w in (450, 480, 520, 560, 600, 650):
        k = int(np.argmin(np.abs(wavelengths - w)))
        ax.plot(l_axis[k], s_axis[k], "o", color=c["att"], markersize=4, zorder=3)
        ax.text(l_axis[k] + 0.012, s_axis[k] + 0.02, f"{w} nm", fontsize=8.5, color=c["muted"], zorder=3)
    ax.plot([0], [0], "D", color=c["dec"], markersize=9, zorder=4)
    ax.annotate("only M cones signal:\nwhere olo sits", (0, 0), xytext=(0.06, 0.32), fontsize=10, color=c["dec_ink"],
                arrowprops={"arrowstyle": "-", "color": c["dec"], "lw": 1})
    ax.text(0.86, 0.6, "every colour a light\ncan make", fontsize=10, color=c["att_ink"], ha="center")
    ax.set_xlim(-0.04, 1.02)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("l_axis = L / (L + M)")
    ax.set_ylabel("s_axis = S / (L + M), scaled")
    _caption(fig, "A colour outside the map",
             "MacLeod-Boynton diagram, CIE 2015 2° observer: the spectral colours and the purples (dashed)", c)
    return fig


# ---------------------------------------------------------------- move 37

COLUMNS = "ABCDEFGHJKLMNOPQRST"  # Go boards skip the letter I


@cache
def move37():
    return json.loads((DATA / "move37.json").read_text(encoding="utf-8"))


def chart_move37(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    c = _style(theme)
    game = move37()
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_axes((0.04, 0.03, 0.62, 0.8))
    ax.set_xlim(-1.2, 18.6)
    ax.set_ylim(-1.2, 18.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-0.6, -0.6), 19.2, 19.2, color=c["card_alt"], zorder=0))
    for k in range(19):
        ax.plot([0, 18], [k, k], color=c["line"], linewidth=0.8, zorder=1)
        ax.plot([k, k], [0, 18], color=c["line"], linewidth=0.8, zorder=1)
        ax.text(k, -1.0, COLUMNS[k], ha="center", va="center", fontsize=7.5, color=c["muted"])
        ax.text(-1.0, k, str(k + 1), ha="center", va="center", fontsize=7.5, color=c["muted"])
    for x in (3, 9, 15):
        for y in (3, 9, 15):
            ax.add_patch(Circle((x, y), 0.12, color=c["muted"], zorder=2))
    for number, (color, point) in enumerate(game["moves"], start=1):
        x, y = ord(point[0]) - ord("a"), 18 - (ord(point[1]) - ord("a"))
        fill, edge = ("#16181C", "#5B6575") if color == "B" else ("#F7F7F5", "#5B6575")
        ax.add_patch(Circle((x, y), 0.46, facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=3))
        if number >= 36:
            ax.text(x, y, str(number), ha="center", va="center", fontsize=8, fontweight="semibold", zorder=4,
                    color="#F7F7F5" if color == "B" else "#16181C")
        if number == 37:
            ax.add_patch(Circle((x, y), 0.68, facecolor="none", edgecolor=c["dec"], linewidth=2.2, zorder=4))
    side = fig.add_axes((0.69, 0.1, 0.3, 0.7))
    side.axis("off")
    notes = [
        ("move 37, Black, P10", c["ink"], "semibold"),
        ("a shoulder hit on the fifth line,", c["muted"], "normal"),
        ("against White 36 at Q11", c["muted"], "normal"),
        ("", c["muted"], "normal"),
        ("AlphaGo's estimate that a human", c["muted"], "normal"),
        ("would play it: 1 in 10,000", c["dec_ink"], "semibold"),
        ("", c["muted"], "normal"),
        ("\"It's not a human move.", c["ink"], "normal"),
        ("I've never seen a human", c["ink"], "normal"),
        ("play this move. … So beautiful.\"", c["ink"], "normal"),
        ("Fan Hui, European champion", c["muted"], "normal"),
    ]
    for k, (line, color, weight) in enumerate(notes):
        side.text(0, 1 - k * 0.075, line, fontsize=10, color=color, fontweight=weight, va="top",
                  transform=side.transAxes)
    _caption(fig, "Move 37",
             "AlphaGo (Black) against Lee Sedol, game 2, Seoul, 10 March 2016: the position after move 37", c,
             top=0.98)
    return fig


# ---------------------------------------------------------------- Boden's three kinds

def boden() -> Figure:
    from tools.figures.lib import Line, Rect

    fig = Figure(
        760,
        330,
        alt="Margaret Boden's three kinds of creativity drawn on a space of ideas. Combinational: two familiar ideas "
        "from distant parts of the space are joined. Exploratory: a path inside the space's rules reaches a place "
        "nobody visited. Transformational: one rule of the space changes, and ideas that were impossible become "
        "possible",
    )
    fig.text(14, 28, "Three ways to be creative", kind="title")
    fig.text(14, 50, "after Margaret Boden: a conceptual space is the set of ideas a style or a set of rules allows",
             kind="note")
    rng = np.random.default_rng(3)
    titles = [("combinational", "join familiar ideas", "that sit far apart"),
              ("exploratory", "follow the rules to", "a place nobody visited"),
              ("transformational", "change a rule: ideas", "that could not exist now can")]
    for k, (name, line1, line2) in enumerate(titles):
        x0, y0, w, h = 14 + k * 250, 70, 232, 180
        fig.box(x0, y0, w, h, role="plain", r=12)
        fig.text(x0 + 14, y0 + 24, name, kind="subtitle")
        inner = (x0 + 26, y0 + 40, w - 52, h - 64)
        fig.els.append(Rect(*inner, "neutral", tint=True, r=10))
        points = [(inner[0] + 10 + rng.uniform(0, inner[2] - 20), inner[1] + 10 + rng.uniform(0, inner[3] - 20))
                  for _ in range(16)]
        for px, py in points:
            fig.els.append(Rect(px - 3, py - 3, 6, 6, "neutral", r=3))
        if k == 0:
            a, b = min(points), max(points)
            fig.els.append(Line([a, ((a[0] + b[0]) / 2, min(a[1], b[1]) - 26), b], "dec", 2.0))
            for px, py in (a, b):
                fig.els.append(Rect(px - 5, py - 5, 10, 10, "dec", r=5))
        elif k == 1:
            path = [(inner[0] + 14, inner[1] + inner[3] - 14), (inner[0] + 60, inner[1] + inner[3] - 40),
                    (inner[0] + 110, inner[1] + 40), (inner[0] + inner[2] - 12, inner[1] + 12)]
            fig.els.append(Line(path, "dec", 2.0))
            fig.els.append(Rect(path[-1][0] - 5, path[-1][1] - 5, 10, 10, "dec", r=5))
        else:
            gx0, gy0 = inner[0] - 8, inner[1] - 8  # the space after the rule changed: larger, drawn dashed
            gx1, gy1 = inner[0] + inner[2] + 18, inner[1] + inner[3] + 8
            fig.els.append(Line([(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1), (gx0, gy0)], "dec", 1.6, dash=True))
            fig.els.append(Rect(inner[0] + inner[2] + 4, inner[1] + inner[3] / 2 - 5, 10, 10, "dec", r=5))
        fig.text(x0 + 14, y0 + h + 26, line1, kind="note")
        fig.text(x0 + 14, y0 + h + 44, line2, kind="note")
    return fig


# ---------------------------------------------------------------- imagining without a hippocampus

HASSABIS = [  # (measure, patients, controls), Hassabis et al. 2007, PNAS: means of 5 patients and 10 controls
    ("richness of the scene", 27.54, 45.06),
    ("spatial coherence", 0.10, 3.68),
    ("quality, rated by the person", 3.88, 7.13),
]


def chart_hassabis(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.7, bottom=0.12, wspace=0.35)
    for ax, (name, patients, controls) in zip(axes, HASSABIS, strict=True):
        ax.bar([0, 1], [patients, controls], color=[c["dec"], c["enc"]], width=0.6)
        for x, v in ((0, patients), (1, controls)):
            ax.text(x, v, f"{v:g}", ha="center", va="bottom", fontsize=9.5, color=c["ink"])
        ax.set_xticks([0, 1], ["patients", "controls"], fontsize=9.5)
        ax.set_title(name, fontsize=10.5)
        ax.set_yticks([])
        ax.set_ylim(0, max(patients, controls) * 1.25)
        for side in ("left",):
            ax.spines[side].set_visible(False)
    _caption(fig, "Imagining a scene without a hippocampus",
             "Hassabis et al. 2007: five patients with hippocampal damage and ten controls imagined new scenes", c)
    return fig


# ---------------------------------------------------------------- people and chatbots on the Alternate Uses Task

@cache
def aut():
    return json.loads((DATA / "aut.json").read_text(encoding="utf-8"))


AUT_GROUPS = [("Human", "256 people"), ("Copy.Ai", "Copy.Ai"), ("ChatGPT3", "ChatGPT-3.5"), ("ChatGPT4", "ChatGPT-4")]


def chart_aut(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rows = aut()["rows"]
    rng = np.random.default_rng(0)  # horizontal jitter only, so the dots do not hide each other
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.14, wspace=0.08)
    for ax, key, title in ((axes[0], "mean_rating", "mean rating of all ideas"),
                           (axes[1], "best_rating", "rating of the best idea")):
        for k, (who, _) in enumerate(AUT_GROUPS):
            values = np.array([r[key] for r in rows if r["who"] == who])
            color = c["enc"] if who == "Human" else c["dec"]
            ax.scatter(k + rng.uniform(-0.28, 0.28, len(values)), values, s=10 if who == "Human" else 22,
                       color=color, alpha=0.45 if who == "Human" else 0.85, linewidths=0)
            ax.plot([k - 0.34, k + 0.34], [values.mean()] * 2, color=c["ink"], linewidth=2)
        ax.set_xticks(range(len(AUT_GROUPS)), [label for _, label in AUT_GROUPS], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("rating by six judges, 1 to 5")
    best = max(r["best_rating"] for r in rows if r["who"] == "Human")
    bots = [r for r in rows if r["who"] != "Human"]
    beaten = sum(r["best_rating"] < best for r in bots)
    _caption(fig, f"On average the chatbots win; the best person beats {beaten} of {len(bots)} sessions",
             "Koivisto & Grassini 2023: uses for a rope, box, pencil, candle; dot: a person or a session; line: "
             "the mean", c)
    return fig


# ---------------------------------------------------------------- Doshi & Hauser 2024

DOSHI_HAUSER = {  # from the paper: novelty and usefulness as % change against writers without AI help; similarity as
    # the increase in % of the similarity range of the stories written without AI
    "novelty": (5.4, 8.1),
    "usefulness": (3.7, 9.0),
    "similarity of stories": (10.7, 8.9),
}


def chart_doshi(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.72, bottom=0.16)
    names = list(DOSHI_HAUSER)
    for k, name in enumerate(names):
        one, five = DOSHI_HAUSER[name]
        color = c["dec"] if name.startswith("similarity") else c["enc"]
        ax.barh(k - 0.18, one, height=0.32, color=color, alpha=0.55)
        ax.barh(k + 0.18, five, height=0.32, color=color)
        unit = "% of the range" if name.startswith("similarity") else "%"
        ax.text(one + 0.2, k - 0.18, f"+{one}{unit}, one AI idea", va="center", fontsize=9, color=c["ink"])
        ax.text(five + 0.2, k + 0.18, f"+{five}{unit}, up to five", va="center", fontsize=9, color=c["ink"])
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xlabel("increase against writers without AI")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _caption(fig, "Better stories, more alike",
             "Doshi & Hauser 2024: 293 writers, short stories rated by 600 readers; GPT-4 offered story ideas", c)
    return fig


# ---------------------------------------------------------------- variation without selection

@cache
def temperature():
    return json.loads((DATA / "temperature.json").read_text(encoding="utf-8"))


def idea(sample, limit=None):
    """The continuation up to its first full stop, comma or line break, cut where non-Latin text begins and, with
    `limit`, at the last whole word that fits."""
    text = re.split(r"[.,\n]", sample.strip())[0]
    shown = re.match(r"[\x20-\x7e]*", text).group(0).strip()
    if limit and len(shown) > limit:
        shown = shown[: limit - 1].rsplit(" ", 1)[0]
    return shown + ("…" if shown != text else "")


def first_words(row):
    return {idea(s).split()[0].lower() for s in row["samples"] if idea(s).split()}


def variation() -> Figure:
    d = temperature()
    rows = d["rows"]
    fig = Figure(
        760,
        116 + 142 * len(rows),
        alt=f"Eight continuations of \"{d['prompt']}\" at each of three temperatures. "
        + "; ".join(f"at {r['temperature']}: {len(first_words(r))} different first words, mean log-probability "
                    f"{np.mean(r['mean_logprob']):.1f} per token" for r in rows)
        + ". Low temperature repeats one idea, high temperature produces new and mostly broken text",
    )
    fig.text(14, 28, "Variation without selection", kind="title")
    fig.text(14, 50, f"eight samples from {d['model'].split('/')[1].split('-Instruct')[0]} at each temperature, "
             "continuing:", kind="note")
    fig.text(14, 72, f"{d['prompt']} …", kind="mono")
    for k, row in enumerate(rows):
        y0 = 106 + k * 142
        fig.text(14, y0, f"temperature {row['temperature']}", kind="subtitle")
        fig.text(746, y0, f"{len(first_words(row))} distinct ideas · log-prob {np.mean(row['mean_logprob']):.1f} "
                 "per token", kind="count_hot" if k == len(rows) - 1 else "count", anchor="end")
        for j, sample in enumerate(row["samples"]):
            x = 14 + (j % 2) * 368
            y = y0 + 14 + (j // 2) * 28
            fig.els.append(Box(x, y, 360, 24, "plain" if k == 1 else "neutral", idea(sample, limit=48), mono=True,
                               r=6, size=11.5))
    return fig


def chart_alts():
    rows = aut()["rows"]
    mean = {who: np.mean([r["mean_rating"] for r in rows if r["who"] == who]) for who, _ in AUT_GROUPS}
    return {
        "hassabis": "Imagined scenes of five patients with hippocampal damage against ten controls: richness 27.54 "
        "against 45.06, spatial coherence 0.10 against 3.68, the person's own quality rating 3.88 against 7.13",
        "aut": f"Ratings of ideas in the Alternate Uses Task, per person or chatbot session. Mean rating: humans "
        f"{mean['Human']:.2f}, Copy.Ai {mean['Copy.Ai']:.2f}, ChatGPT-3.5 {mean['ChatGPT3']:.2f}, ChatGPT-4 "
        f"{mean['ChatGPT4']:.2f}; the humans spread from the lowest scores to near the top",
        "doshi": "With GPT-4 story ideas, novelty rose by 5.4% (one idea) and 8.1% (up to five), usefulness by 3.7% "
        "and 9.0%, and the similarity of the stories to each other by 10.7% and 8.9% of its range without AI",
        "move37": "The Go board after move 37 of the second game between AlphaGo and Lee Sedol: AlphaGo's black "
        "stone at P10, a shoulder hit on the fifth line against White's stone at Q11, which AlphaGo estimated a human "
        "would play with a probability of 1 in 10,000",
        "networks": "The left hemisphere seen from the side and from the middle, with three of Yeo's seven networks "
        "coloured: the default mode network on the medial prefrontal and posterior cingulate cortex, the angular "
        "gyrus and the lateral temporal lobe; the executive control network on the lateral prefrontal and parietal "
        "cortex; the salience network on the insula and the anterior cingulate",
        "olo": "The MacLeod-Boynton chromaticity diagram. Every colour a light can produce lies inside the spectral "
        "locus, which runs from l of about 0.6 at short wavelengths to nearly 1 at long ones; a signal from the M "
        "cones alone sits at l = 0, far outside it",
    }


# ---------------------------------------------------------------- cover

def cover():
    game = move37()
    step, x0, y0 = 21, cv.MOTIF_X0 + 70, 160
    out = [cv.label(cv.MOTIF_X0, 128, "ALPHAGO - LEE SEDOL, GAME 2, MOVE 37")]
    for k in range(19):
        out.append(f'<line x1="{x0 + k * step}" y1="{y0}" x2="{x0 + k * step}" y2="{y0 + 18 * step}" '
                   f'stroke="{cv.LINE}" stroke-width="1"/>')
        out.append(f'<line x1="{x0}" y1="{y0 + k * step}" x2="{x0 + 18 * step}" y2="{y0 + k * step}" '
                   f'stroke="{cv.LINE}" stroke-width="1"/>')
    for number, (color, point) in enumerate(game["moves"], start=1):
        cx = x0 + (ord(point[0]) - ord("a")) * step
        cy = y0 + (ord(point[1]) - ord("a")) * step  # SGF rows count from the top, like SVG
        if number == 37:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="none" stroke="{cv.RED}" stroke-width="3"/>')
            out.append(f'<circle cx="{cx}" cy="{cy}" r="9.5" fill="{cv.RED}"/>')
        elif color == "B":  # black stones dark with a light rim, white stones light, as on a real board
            out.append(f'<circle cx="{cx}" cy="{cy}" r="8.8" fill="#0B0C0E" stroke="{cv.CHIP_INK}" '
                       'stroke-width="1.6"/>')
        else:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="9.5" fill="{cv.CHIP_INK}"/>')
    cv.fits(x0 + 18 * step + 10)
    out.append(cv.label(cv.MOTIF_X0, 590, "A HUMAN WOULD PLAY IT: 1 IN 10,000"))
    return cv.frame("blog / cognitive science", ["Can AI be", "creative?"], ["And how creative", "are we?"],
                    "".join(out), footer="Cognitive science, measured")


COVER = cover

CHARTS = {
    "move37": chart_move37,
    "hassabis": chart_hassabis,
    "aut": chart_aut,
    "doshi": chart_doshi,
    "networks": chart_networks,
    "olo": chart_olo,
}

FIGURES = {
    "variation": variation,
    "boden": boden,
}
