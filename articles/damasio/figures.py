"""Figures, charts and the cover of the post "Somatic Marker Hypothesis by Antonio Damasio" on xmarva.github.io.

The cortex is fsaverage6 with Destrieux et al.'s gyri and sulci, the slice the MNI152 template with the Harvard-Oxford
subcortical atlas (tools/figures/brain.py).

    uv run --group articles python tools/render.py --article damasio --out <site>/assets/posts/antonio-damasio
"""

import numpy as np

from tools.figures import brain
from tools.figures import cover as cv
from tools.figures.lib import Box, Figure


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.07, note, ha="left", va="top", fontsize=10, color=c["muted"])


def _greys(theme):
    return ("#E9ECF1", "#BCC2CC") if theme == "light" else ("#4A5160", "#2E343E")


# ---------------------------------------------------------------- the regions of the hypothesis

REGIONS = [  # (name in the post, palette role, Destrieux labels on the surface, Harvard-Oxford labels on the slice)
    ("ventromedial prefrontal", "dec",
     ("G_rectus", "G_subcallosal", "S_suborbital", "G_orbital", "S_orbital_med-olfact"), ()),
    ("insula", "data", ("G_insular_short", "G_Ins_lg_and_S_cent_ins", "S_circular_insula_ant",
                        "S_circular_insula_inf", "S_circular_insula_sup"), ("Insular Cortex",)),
    ("somatosensory", "enc", ("G_postcentral",), ()),
    ("amygdala", "att", (), ("Left Amygdala", "Right Amygdala")),
]
SLICE_Y = -4  # mm, MNI: a coronal plane through both amygdalae and the insula


def chart_regions(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = _style(theme)
    labels = brain.destrieux("left")
    surface = {label: c[role] for _, role, names, _ in REGIONS for label in names}
    volume = {label: c[role] for _, role, _, names in REGIONS for label in names}
    fig = plt.figure(figsize=(7.6, 3.5))
    left = fig.add_axes((0.0, 0.2, 0.34, 0.6))
    middle = fig.add_axes((0.35, 0.2, 0.34, 0.6))
    right = fig.add_axes((0.72, 0.22, 0.27, 0.56))
    for ax, view in ((left, "lateral"), (middle, "medial")):
        brain.draw(ax, "left", view, labels, surface, base=_greys(theme))
        ax.text(0.5, -0.03, f"left hemisphere, {view}", transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                color=c["muted"])
    tissue = ("#2E3542", "#F1F3F6") if theme == "light" else ("#1E232B", "#C9CED7")
    brain.coronal(right, SLICE_Y, volume, tissue=tissue, background=c["card"])
    right.text(0.5, -0.05, f"coronal slice, y = {SLICE_Y} mm", transform=right.transAxes, ha="center", va="top",
               fontsize=9.5, color=c["muted"])
    handles = [Patch(facecolor=c[role], label=name) for name, role, *_ in REGIONS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9.5,
               handlelength=1.2, columnspacing=1.6)
    _caption(fig, "Where the hypothesis lives",
             "fsaverage cortex with Destrieux's gyri; MNI152 slice with the Harvard-Oxford amygdala and insula", c)
    return fig


# ---------------------------------------------------------------- the Iowa Gambling Task

# Bechara, Damasio, Damasio & Anderson 1994, Cognition 50, Fig. 1 score card (the same schedule as PEBL's iowa.pbl):
# every card of a deck pays the same gain; some cards also take a loss, at these positions within each block of 10
GAIN = {"A": 100, "B": 100, "C": 50, "D": 50}
LOSSES = {
    "A": [0, 0, 150, 0, 300, 0, 200, 0, 250, 350, 0, 350, 0, 250, 200, 0, 300, 150, 0, 0,
          0, 300, 0, 350, 0, 200, 250, 150, 0, 0, 350, 200, 250, 0, 0, 0, 150, 300, 0, 0],
    "B": [0, 0, 0, 0, 0, 0, 0, 0, 1250, 0, 0, 0, 0, 1250, 0, 0, 0, 0, 0, 0,
          1250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1250, 0, 0, 0, 0, 0, 0, 0, 0],
    "C": [0, 0, 50, 0, 50, 0, 50, 0, 50, 50, 0, 25, 75, 0, 0, 0, 25, 75, 0, 50,
          0, 0, 0, 50, 25, 50, 0, 0, 75, 50, 0, 0, 0, 25, 25, 0, 75, 0, 50, 75],
    "D": [0, 0, 0, 0, 0, 0, 0, 0, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
          0, 0, 0, 0, 0, 0, 0, 0, 250, 0, 0, 0, 0, 250, 0, 0, 0, 0, 0, 0],
}


def running_total(deck):
    return np.cumsum([GAIN[deck] - loss for loss in LOSSES[deck]])


def check_schedule():
    for deck in GAIN:
        totals = running_total(deck)
        expected = -250 if deck in "AB" else 250
        assert all(totals[9::10] == expected * np.arange(1, 5)), deck  # the paper's net per block of 10 cards


def chart_decks(theme):
    import matplotlib.pyplot as plt

    check_schedule()
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.78, bottom=0.12)
    cards = np.arange(1, 41)
    styles = {"A": (c["dec"], "-"), "B": (c["dec"], "--"), "C": (c["enc"], "-"), "D": (c["enc"], "--")}
    for deck, (color, dash) in styles.items():
        ax.plot(cards, running_total(deck), color=color, linestyle=dash, linewidth=2, label=f"deck {deck}")
    ax.text(41, running_total("A")[-1], "A and B", va="center", fontsize=10, color=c["ink"])
    ax.text(41, running_total("C")[-1], "C and D", va="center", fontsize=10, color=c["ink"])
    ax.legend(loc="lower left", ncol=4, fontsize=9.5, title="solid: frequent small losses; dashed: rare large ones",
              title_fontsize=9.5)
    peak = int(np.argmax(running_total("B")))
    ax.annotate(f"deck B: +${running_total('B')[peak]:,} after {peak + 1} cards,\nthen a single loss of $1,250",
                (peak + 1, running_total("B")[peak]), xytext=(12, 900), fontsize=9.5, color=c["muted"],
                arrowprops={"arrowstyle": "-", "color": c["muted"], "lw": 0.8})
    ax.axhline(0, color=c["line"], linewidth=1)
    ax.set_ylim(-2700, 1300)  # room below the curves for the legend
    ax.set_xlim(1, 40)
    ax.set_xlabel("cards taken from the deck")
    ax.set_ylabel("running total, $")
    _caption(fig, "Four decks: two that pay big and lose more",
             "the Iowa Gambling Task's schedule (Bechara et al. 1994): what 40 cards from each deck add up to", c)
    return fig


def igt_timeline() -> Figure:
    fig = Figure(
        760,
        334,
        alt="The four periods of Bechara and colleagues' 1997 experiment, in cards: before the first loss, up to "
        "about card 10; pre-hunch, to about card 50 for normal participants and card 80 for patients; hunch, cards "
        "50 to 80, normal participants only; conceptual, after about card 80. Normal participants generated "
        "anticipatory skin conductance responses from the pre-hunch period on; the patients never did, and the three "
        "who reached the conceptual period still chose the bad decks",
    )
    fig.text(14, 28, "The skin before the story", kind="title")
    fig.text(14, 50, "Bechara et al. 1997: 10 normal participants, 6 with ventromedial damage; typical card numbers",
             kind="note")
    x0, scale = 150, 5.9  # px per card, 100 cards
    fig.line((x0, 286), (x0 + 100 * scale, 286), width=1)
    for card in (0, 10, 20, 50, 80, 100):
        x = x0 + card * scale
        fig.line((x, 286), (x, 292), width=1)
        fig.text(x, 306, str(card), kind="small", anchor="middle")
    fig.text(x0 + 10 * scale + 4, 78, "first loss", kind="small")
    fig.text(x0 + 50 * scale, 322, "cards taken", kind="small", anchor="middle")
    rows = [
        ("normal", 96, [(0, 10, "neutral", None), (10, 50, "att", "pre-hunch"), (50, 80, "enc", "hunch"),
                        (80, 100, "enc", "conceptual")],
         ["anticipatory sweating before the bad decks; yet",
          '"by card 20, all indicated that they did not have a clue"']),
        ("patients", 196, [(0, 10, "neutral", None), (10, 80, "att", "pre-hunch"), (80, 100, "dec", "conceptual")],
         ["no anticipatory sweating, ever;", "the 3 of 6 who reach the conceptual period still choose badly"]),
    ]
    for name, y, spans, notes in rows:
        fig.text(14, y + 20, name, kind="subtitle")
        for a, b, role, label in spans:
            fig.els.append(Box(x0 + a * scale + 1, y, (b - a) * scale - 2, 30, role, label,
                               strong=role in ("enc", "dec"), r=6, size=12))
        fig.text(14, y + 52, "skin", kind="label")
        for k, line in enumerate(notes):
            fig.text(x0 + 10 * scale, y + 52 + k * 16, line, kind="small")
    return fig


def loops() -> Figure:
    fig = Figure(
        760,
        318,
        alt="Damasio's two loops. In the body loop, a situation evokes, through the ventromedial prefrontal cortex "
        "and the amygdala, a change in the body, which the somatosensory cortices and the insula register, biasing "
        "the decision. In the as-if body loop, the prefrontal cortex signals the somatosensory cortices directly, "
        "the body being bypassed",
    )
    fig.text(14, 28, "Two routes for a somatic marker", kind="title")
    fig.text(14, 50, "Damasio 1996: the body loop and the \"as if\" body loop, which bypasses the body", kind="note")
    fig.box(14, 132, 120, 44, role="data", label="a situation", r=10)
    fig.box(180, 132, 150, 44, role="dec", label="vmPFC, amygdala", strong=True, r=10)
    fig.box(400, 72, 140, 44, role="neutral", label="the body", r=10)
    fig.box(392, 192, 170, 44, role="att", label="somatosensory, insula", r=10)
    fig.box(610, 132, 136, 44, role="enc", label="biased choice", strong=True, r=10)
    fig.arrow((136, 154), (178, 154))
    fig.arrow((332, 144), (398, 98), role="dec", width=1.8)
    fig.arrow((470, 118), (470, 190), role="dec", width=1.8)
    fig.arrow((332, 166), (398, 210), role="att", width=1.8)
    fig.arrow((564, 214), (608, 162))
    fig.text(352, 92, "body loop", kind="label", role="dec", anchor="end")
    fig.text(478, 160, "heart, gut, skin", kind="small")
    fig.text(352, 232, "as-if loop", kind="label", role="att", anchor="end")
    fig.text(14, 284, "\"in most circumstances an 'as if' body loop is engaged instead, and the body, that is the "
             "periphery,", kind="note")
    fig.text(14, 302, "is bypassed entirely\"", kind="note")
    return fig


def chart_alts():
    return {
        "decks": "Running totals over 40 cards of each deck in the Iowa Gambling Task: decks A and B fall to minus "
        "1,000 dollars, decks C and D rise to plus 1,000 dollars; deck B climbs to plus 800 dollars after eight cards "
        "before its first loss of 1,250 dollars",
        "regions": "The left hemisphere from the side and from the middle, with the ventromedial prefrontal and "
        "orbital cortex, the insula and the somatosensory cortex of the postcentral gyrus "
        f"coloured; and a coronal slice at y = {SLICE_Y} mm showing both amygdalae and the insulae",
    }


# ---------------------------------------------------------------- cover

def cover():
    check_schedule()
    x0, x1, y_top, y_bottom = cv.MOTIF_X0 + 4, cv.MOTIF_X1 - 80, 170, 430
    low, high = -1900, 1100
    out = [cv.label(cv.MOTIF_X0, 128, "IOWA GAMBLING TASK, 40 CARDS PER DECK")]

    def y(value):
        return y_top + (high - value) / (high - low) * (y_bottom - y_top)

    out.append(f'<line x1="{x0}" y1="{y(0):.1f}" x2="{x1}" y2="{y(0):.1f}" stroke="{cv.LINE}" stroke-width="1"/>')
    for deck in "DCBA":
        totals = [0, *running_total(deck)]
        points = " ".join(f"{x0 + k * (x1 - x0) / 40:.1f},{y(v):.1f}" for k, v in enumerate(totals))
        bad = deck in "AB"
        dash = ' stroke-dasharray="8 6"' if deck in "BD" else ""
        out.append(f'<polyline points="{points}" fill="none" stroke="{cv.RED if bad else cv.CHIP_INK}" '
                   f'stroke-width="3" stroke-linejoin="round"{dash}/>')
    out.append(cv.text(x1 + 12, y(1000) + 6, "C, D", 18, cv.CHIP_INK, cv.MONO, 500))
    out.append(cv.text(x1 + 12, y(-1000) + 6, "A, B", 18, cv.RED, cv.MONO, 500))
    cv.fits(x1 + 12 + 60)
    out.append(cv.label(cv.MOTIF_X0, 500, "THE SKIN REACTS BEFORE THE STORY"))
    chip, _ = cv.chip(cv.MOTIF_X0, 520, "to know but not to feel", size=18, kind="hot")
    out.append(chip)
    return cv.frame("blog / cognitive science", ["Somatic", "markers"], ["Antonio Damasio on feeling", "and deciding"],
                    "".join(out), footer="Cognitive science, measured")


COVER = cover

CHARTS = {
    "regions": chart_regions,
    "decks": chart_decks,
}

FIGURES = {
    "timeline": igt_timeline,
    "loops": loops,
}
