"""Figures, charts and the cover of the post "Is Centaur a Model of the Mind?" on xmarva.github.io.

Colours: teal is what the experimenter wrote (instructions), amber is what the participant did, violet is a model,
rose is a model without part of its input.

    uv run --group articles python tools/render.py --article centaur --out <site>/assets/posts/centaur
"""

import csv
import json
import re
import textwrap
from functools import cache
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import MONO_ADVANCE, Figure

DATA = Path(__file__).parent / "data"


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


@cache
def transcript():
    return json.loads((DATA / "transcript.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- what Centaur reads

HEADER_LINES = (0, 2, 8, 12)  # the role, the genies, the carpets, the goal; the other nine lines are elided


def _mono_line(fig, x, y, line, chip_role="data"):
    """A line of the transcript in monospace, each <<response>> drawn as a chip; returns the right edge."""
    step = 13 * MONO_ADVANCE
    for part in re.split(r"(<<.*?>>)", line):
        if part.startswith("<<"):
            w = fig.chip(x + 2, y - 19, part[2:-2], role=chip_role, strong=True, h=26)
            x += w + 4
        elif part:
            fig.text(x, y, part, kind="mono")
            x += len(part) * step
    return x


def transcript_figure() -> Figure:
    d = transcript()
    lines = []
    for k in HEADER_LINES:
        lines += [("head", s) for s in textwrap.wrap(d["header"][k], 86)]
        if k != HEADER_LINES[-1]:
            lines.append(("gap", "⋯"))
    trial_lines = []
    for trial in d["trials"][:3]:
        first, second = trial.split(" You see ", 1)
        trial_lines += [("trial", first), ("trial", "You see " + second)]
    fig = Figure(
        760,
        150 + 24 * len(lines) + 34 * len(trial_lines),
        alt="The start of one participant's session in the magic-carpet two-step task as Psych-101-style text: "
        "instructions about carpets, mountains and genies, then one line per day, in which the participant's key "
        f"presses are wrapped in double angle brackets, for example You press <<M>>. This session has "
        f"{d['responses']} such responses, and Centaur's training loss is computed only on them",
    )
    fig.text(14, 28, "What Centaur reads", kind="title")
    fig.text(14, 50, "one participant's session as text; training and scoring use only the boxed key presses",
             kind="note")
    y = 88
    fig.text(14, y, "the instructions, 13 lines (four shown)", kind="label")
    y += 26
    for kind, s in lines:
        if kind == "gap":
            fig.text(14, y, s, kind="label")
        else:
            fig.text(14, y, s, kind="mono", role="enc")
        y += 24
    y += 14
    fig.text(14, y, f"the session, {d['responses']} responses (the first three days shown)", kind="label")
    y += 34
    for _, s in trial_lines:
        right = _mono_line(fig, 14, y, s)
        assert right < 746, s
        y += 34
    return fig


# ---------------------------------------------------------------- what Psych-101 is made of

TYPE_ROLES = {"Multi-armed bandits": "att", "Decision-making": "data", "Markov decision processes": "enc",
              "Memory": "dec", "Supervised learning": "neutral", "Miscellaneous": "neutral"}


@cache
def experiments():
    rows = list(csv.DictReader((DATA / "experiments.csv").open(encoding="utf-8"), delimiter=";"))
    return [r for r in rows if r["split"] == "train"]


def chart_psych101(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rows = sorted(experiments(), key=lambda r: -float(r["num_choices"]))
    total = sum(float(r["num_choices"]) for r in rows)
    top = rows[:12]
    rest = rows[12:]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.subplots_adjust(left=0.43, right=0.95, top=0.82, bottom=0.1)
    colors = {t: (c[r] if r != "neutral" else c["muted"]) for t, r in TYPE_ROLES.items()}
    names = [f"{r['task_name']}, {r['paper_ref'].split(',')[0].split(' (')[0]}" for r in top] + [
        f"{len(rest)} more datasets"]
    values = [float(r["num_choices"]) / 1e6 for r in top] + [sum(float(r["num_choices"]) for r in rest) / 1e6]
    ys = list(range(len(values)))[::-1]
    ax.barh(ys, values, color=[colors[r["task_type"]] for r in top] + [c["grid"]], height=0.7)
    for y, v in zip(ys, values, strict=True):
        ax.text(v + 0.03, y, f"{v:.2f} M", va="center", fontsize=9, color=c["muted"])
    ax.set_yticks(ys, [n if len(n) < 52 else n[:50] + "…" for n in names], fontsize=9)
    ax.set_xlabel("choices, millions")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    share = sum(float(r["num_choices"]) for r in rows[:4]) / total
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in TYPE_ROLES if t != "Miscellaneous"]
    labels = [t.lower() if t != "Supervised learning" else "supervised learning, other" for t in TYPE_ROLES
              if t != "Miscellaneous"]
    ax.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8.5)
    _caption(fig, "What Psych-101 is made of",
             f"{len(rows)} datasets, {total / 1e6:.1f} M choices with held-out participants; the four largest hold "
             f"{share:.0%}", c)
    return fig


# ---------------------------------------------------------------- the debate

DEBATE = [  # (month, who, what)
    ("Oct 2024", "Binz et al., preprint", "\"the first real candidate for a unified model…\""),
    ("Jun 2025", "Bowers et al.", "\"Centaur: a model without a theory\""),
    ("Jul 2025", "Binz et al., Nature", "\"a first step\"; the press release: \"AI that thinks like us\""),
    ("Jul 2025", "Xie & Zhu", "beats cognitive models even with the task description removed"),
    ("Aug 2025", "Namazova et al.", "run on its own, it misses reversal and the horizon effect"),
    ("Sep 2025", "Orr et al.", "\"a unified model of behavior sans cognition\""),
    ("Dec 2025", "Liu & Ding", "told to always answer J, it still fits people"),
    ("May 2026", "Binz et al.", "Psych-201: 208,021 participants, 25.9 M responses"),
    ("Aug 2026", "Oh & Gobet", "0.6B matches 70B; stimuli and feedback carry the signal"),
    ("Oct 2026", "Binz & Schulz", "CORE: 452,608 parameters beat Centaur in 47 of 62 studies"),
]


def debate() -> Figure:
    fig = Figure(
        760,
        96 + 32 * len(DEBATE),
        alt="A timeline of the Centaur debate: " + "; ".join(f"{m}, {w}: {t}" for m, w, t in DEBATE),
    )
    fig.text(14, 28, "Two years of arguing about Centaur", kind="title")
    fig.text(14, 50, "the authors' own papers (violet) and the replies (amber)", kind="note")
    for k, (month, who, what) in enumerate(DEBATE):
        y = 92 + 32 * k
        fig.text(14, y, month, kind="label")
        fig.rect(98, y - 9, 8, 8, role="att" if who.startswith("Binz") else "data", r=4)
        fig.text(120, y, who, kind="subtitle")
        fig.text(290, y, what, kind="body")
    fig.line((102, 84), (102, 92 + 32 * (len(DEBATE) - 1)), role="neutral")
    return fig


# ---------------------------------------------------------------- playing the task

@cache
def results():
    return json.loads((DATA / "results.json").read_text(encoding="utf-8"))


AGENTS = [  # (key in results.json, title, palette role)
    ("people", "24 people", "data"),
    ("minitaur", "Minitaur", "att"),
    ("minitaur, no instructions", "Minitaur, no instructions", "dec"),
    ("llama", "Llama 3.1 8B", "neutral"),
]


def chart_stays(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    r = results()["stays"]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.72, bottom=0.2, wspace=0.12)
    for ax, (key, title, role) in zip(axes, AGENTS, strict=True):
        d = r[key]
        solid = c["muted"] if role == "neutral" else c[role]
        tint = c["grid"] if role == "neutral" else c[role + "_tint"]
        for g, reward in enumerate(["1", "0"]):
            for b, (trans, color) in enumerate([("c", solid), ("r", tint)]):
                x = g * 2.6 + b
                ax.bar(x, d["pooled"][reward + trans], color=color, edgecolor=solid, linewidth=1.2, width=0.9)
        ax.set_xticks([0.5, 3.1], ["reward", "none"], fontsize=9.5)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_title(title, fontsize=10, color=c["ink"], fontweight="semibold")
        lo, hi = d["model-based CI"]
        ax.text(1.8, 1.18, f"{d['model-based index']:+.2f}", ha="center", va="top", fontsize=10.5,
                color=c["ink"], fontweight="semibold")
        ax.text(1.8, 1.07, f"[{lo:+.2f}, {hi:+.2f}]", ha="center", va="top", fontsize=8.5, color=c["muted"])
    axes[0].set_ylabel("P(same carpet next day)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c["muted"], edgecolor=c["muted"]),
               plt.Rectangle((0, 0), 1, 1, facecolor=c["grid"], edgecolor=c["muted"])]
    fig.legend(handles, ["after a common flight", "after a rare flight"], loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    _caption(fig, "Predicting people is not the same as playing like them",
             "the magic-carpet task; the number is the model-based index with its 95% interval over 24 sessions", c)
    return fig


# ---------------------------------------------------------------- taking things away

PUBLISHED = {  # magic-carpet task, NLL per response, from results/*.csv and generalization/results/privileged.csv
    "Centaur 70B (paper)": 0.5103,  # in github.com/marcelbinz/Llama-3.1-Centaur-70B
    "hybrid model, fitted to these people (paper)": 0.5882,
    "hybrid model (paper)": 0.6070,
}
LADDER = [  # (label, key in results.json or PUBLISHED, palette role)
    ("Centaur 70B, published", "Centaur 70B (paper)", "att"),
    ("Minitaur", "minitaur/full", "att"),
    ("Minitaur, no instructions", "minitaur/no instructions", "dec"),
    ("six-number lookup table", "table", "data"),
    ("hybrid model, fitted to these people", "hybrid model, fitted to these people (paper)", "enc"),
    ("hybrid model, as in the paper", "hybrid model (paper)", "enc"),
    ("Llama 3.1 8B", "llama/full", "neutral"),
    ("Llama 3.1 8B, no instructions", "llama/no instructions", "neutral"),
    ("repeat the last choice", "repeat", "data"),
    ("Minitaur, no history", "minitaur/no history", "dec"),
]


def chart_ladder(theme):
    import math

    import matplotlib.pyplot as plt

    c = _style(theme)
    r = results()["twostep nll"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.4, right=0.9, top=0.82, bottom=0.12)
    for k, (_, key, role) in enumerate(LADDER):
        y = len(LADDER) - 1 - k
        color = c["muted"] if role == "neutral" else c[role]
        if key in PUBLISHED:
            ax.scatter([PUBLISHED[key]], [y], s=60, facecolor=c["card"], edgecolor=color, linewidth=2, zorder=3)
            value = PUBLISHED[key]
        else:
            lo, hi = r[key]["CI"]
            value = r[key]["mean"]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, alpha=0.5, zorder=2)
            ax.scatter([value], [y], s=60, color=color, zorder=3)
        ax.text(1.02, y, f"{value:.3f}", va="center", ha="left", fontsize=9, color=c["muted"],
                transform=ax.get_yaxis_transform())
    ax.axvline(math.log(2), color=c["muted"], linestyle="--", linewidth=1.2)
    ax.text(math.log(2) + 0.002, len(LADDER) - 0.45, "coin flip", fontsize=9, color=c["muted"])
    ax.set_yticks(range(len(LADDER)), [label for label, _, _ in LADDER][::-1], fontsize=9.5)
    ax.set_xlim(0.48, 0.745)
    ax.set_xlabel("negative log-likelihood per choice (lower is better)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _caption(fig, "What predicts the magic-carpet choices",
             "24 held-out sessions; filled: measured here, with 95% intervals; hollow: from the paper's repository", c)
    return fig


def chart_logic(theme):
    import math

    import matplotlib.pyplot as plt

    c = _style(theme)
    r = results()["logic"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.17, wspace=0.3)
    bars = [("minitaur/full", "Minitaur", "att"), ("minitaur/no questions", "no questions", "dec"),
            ("llama/full", "Llama", "neutral"), ("llama/no questions", "no questions", "neutral")]
    for k, (key, _, role) in enumerate(bars):
        color = c["muted"] if role == "neutral" else c[role]
        tint = k % 2 == 1
        lo, hi = r[key]["CI"]
        a.bar(k + (k > 1) * 0.5, r[key]["mean"], color=c["grid"] if tint and role == "neutral" else
              (c[role + "_tint"] if tint else color), edgecolor=color, linewidth=1.2, width=0.8)
        a.text(k + (k > 1) * 0.5, r[key]["mean"] + 0.02, f"{r[key]['mean']:.2f}", ha="center", fontsize=9,
               color=c["ink"])
    a.axhline(math.log(5), color=c["muted"], linestyle="--", linewidth=1.2)
    a.set_xticks([0, 1, 2.5, 3.5], ["Minitaur", "no\nquestions", "Llama", "no\nquestions"], fontsize=9)
    a.set_ylim(1.2, 1.8)
    a.set_ylabel("NLL per answer")
    a.set_title(f"{r['participants']} people, 20 answers each", fontsize=10, color=c["muted"], fontweight="normal")
    share = r["share choosing the most popular answer"]
    for key, role, label in [("minitaur/full", "att", "Minitaur, with the question"),
                             ("minitaur/no questions", "dec", "Minitaur, without it")]:
        b.scatter([100 * x for x in share], r[key]["per question"], s=28, color=c[role], label=label, zorder=3)
    b.axhline(math.log(5), color=c["muted"], linestyle="--", linewidth=1.2, label="random among 5, both panels")
    b.set_xlabel("people choosing the most popular answer, %")
    b.set_ylabel("NLL per answer")
    b.set_title("one dot per question", fontsize=10, color=c["muted"], fontweight="normal")
    b.legend(frameon=False, fontsize=8.5, loc="lower left")
    _caption(fig, "In the logic problems, the question is the signal",
             "twenty LSAT-style problems from a study the paper held out; the answer letters differ for every "
             "person", c)
    return fig


# ---------------------------------------------------------------- cover

def cover():
    d = transcript()
    m = re.match(r"You are presented with magic carpets (\w) and (\w)\. You press <<(\w)>>\. "
                 r"You end up on (\w+) Mountain\. You see lamp (\w) and lamp (\w)\. You rub lamp <<(\w)>>\. "
                 r"You receive (\d) coins\.", d["trials"][0])
    carpet, mountain, lamp, coins = m.group(3), m.group(4), m.group(7), m.group(8)
    out = [cv.label(cv.MOTIF_X0, 128, "ONE DAY OF THE MAGIC-CARPET TASK")]
    y = 186
    for before, response, after in [("You press ", carpet, "."), (f"You end up on {mountain} Mountain.", None, ""),
                                    ("You rub lamp ", lamp, "."), (f"You receive {coins} coins.", None, "")]:
        out.append(cv.text(cv.MOTIF_X0, y, before, 22, cv.CHIP_INK, cv.MONO))
        x = cv.MOTIF_X0 + len(before) * 0.6 * 22
        if response:
            chip, w = cv.chip(x + 4, y - 30, response, size=22, kind="hot", h=42)
            out.append(chip)
            out.append(cv.text(x + w + 8, y, after, 22, cv.CHIP_INK, cv.MONO))
            cv.fits(x + w + 8 + 14)
        else:
            cv.fits(x)
        y += 62
    out.append(cv.label(cv.MOTIF_X0, 470, "THE MODEL IS TRAINED ON THE RED BOXES ONLY"))
    chip, _ = cv.chip(cv.MOTIF_X0, 490, f"{d['responses']} of them in one session", size=18, kind="warm")
    out.append(chip)
    return cv.frame("blog / cognitive science", ["Is Centaur", "a model of", "the mind?"],
                    ["A Llama fine-tuned on", "160 experiments"], "".join(out), footer="Cognitive science, measured")


COVER = cover

FIGURES = {
    "transcript": transcript_figure,
    "debate": debate,
}

CHARTS = {
    "psych101": chart_psych101,
    "ladder": chart_ladder,
    "logic": chart_logic,
    "stays": chart_stays,
}


def chart_alts():
    rows = experiments()
    total = sum(float(r["num_choices"]) for r in rows)
    r = results()
    n, st, lg = r["twostep nll"], r["stays"], r["logic"]
    ladder = "; ".join(f"{label} {PUBLISHED[key] if key in PUBLISHED else n[key]['mean']:.3f}"
                       for label, key, _ in LADDER)
    return {
        "ladder": "Negative log-likelihood per choice on the magic-carpet task, lower is better, against 0.693 for a "
        f"coin flip: {ladder}",
        "logic": f"Logic problems: Minitaur scores {lg['minitaur/full']['mean']:.2f} per answer with the questions and "
        f"{lg['minitaur/no questions']['mean']:.2f} without them, Llama {lg['llama/full']['mean']:.2f} and "
        f"{lg['llama/no questions']['mean']:.2f}, against {lg['chance']:.2f} for a random pick among five. "
        "Per question, "
        "Minitaur's score with the question falls where more people agree on one answer; without the question it stays "
        "near chance",
        "stays": "Probability of taking the same carpet the day after a reward or no reward, after a common or a rare "
        "flight. " + "; ".join(f"{title}: model-based index {st[key]['model-based index']:+.2f}"
                               for key, title, _ in AGENTS)
        + ". People show the crossing pattern of model-based learning; the models do not",
        "psych101": f"The {len(rows)} training datasets of Psych-101 by number of choices, "
        f"{total / 1e6:.1f} million in "
        "all with the held-out participants. The largest are the THINGS odd-one-out task, the balloon analogue risk "
        "task, choices13k and decisions from experience; together the four largest hold about half of all choices",
    }
