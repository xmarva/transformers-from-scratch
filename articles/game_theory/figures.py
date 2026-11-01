"""Figures, charts and the cover of the post "Is it rational to be good?" on xmarva.github.io.

Everything is computed when it is drawn: payoffs of the repeated prisoner's dilemma exactly, from the Markov chain
of two memory-one strategies (games.py); the populations that learn by imitation from data/evolution.json
(simulate.py).

    uv run --group articles python tools/render.py --article game_theory --out <site>/assets/posts/game-theory-ethics
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from games import STRATEGIES, P, R, S, T, payoffs, strategy, tournament  # noqa: E402

from tools.figures import cover as cv  # noqa: E402
from tools.figures.lib import Figure  # noqa: E402

DATA = HERE / "data"
LONG = 0.99654  # Axelrod's second tournament: another round with this probability, 289 rounds on average
NICE = ("Always cooperate", "Tit for tat", "Grim trigger", "Win-stay, lose-shift", "Generous tit for tat",
        "Generous ZD")  # never the first to defect


def evolution():
    return json.loads((DATA / "evolution.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- one round

def dilemma() -> Figure:
    fig = Figure(
        760,
        300,
        alt="The prisoner's dilemma with Axelrod's payoffs. Both cooperate: 3 points each. You cooperate and they "
        "defect: 0 for you, 5 for them. You defect and they cooperate: 5 for you, 0 for them. Both defect: 1 each. "
        "Whatever the other does, defecting gives you more, and two players who reason this way get 1 each",
    )
    fig.text(14, 28, "The prisoner's dilemma, one round", kind="title")
    fig.text(14, 50, "points for you, then for them (Axelrod's numbers)", kind="note")
    x0, y0, cw, ch = 214, 96, 260, 64
    fig.text(x0 + cw / 2, y0 - 12, "they cooperate", kind="label", anchor="middle")
    fig.text(x0 + cw + 12 + cw / 2, y0 - 12, "they defect", kind="label", anchor="middle")
    cells = [
        [("enc", f"{R}, {R}", "reward for both"), ("data", f"{S}, {T}", "you are exploited")],
        [("data", f"{T}, {S}", "you exploit them"), ("dec", f"{P}, {P}", "punishment for both")],
    ]
    for row, label in enumerate(("you cooperate", "you defect")):
        y = y0 + row * (ch + 12)
        fig.text(x0 - 14, y + ch / 2 + 4, label, kind="label", anchor="end")
        for col, (role, score, meaning) in enumerate(cells[row]):
            x = x0 + col * (cw + 12)
            fig.box(x, y, cw, ch, role=role, r=8)
            fig.text(x + 18, y + 40, score, kind="lead")
            fig.text(x + cw - 16, y + 39, meaning, kind="note", anchor="end")
    fig.text(14, 270, f"Whatever they do, defecting pays you more: {T} instead of {R}, {P} instead of {S}.",
             kind="body")
    fig.text(14, 290, f"So both defect and get {P} each, with {R} each on the table.", kind="body")
    return fig


# ---------------------------------------------------------------- the shadow of the future

def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, 0.905, note, ha="left", va="top", fontsize=10, color=c["muted"])


def shadow_lines():
    tft = STRATEGIES["Tit for tat"]
    ws = [k / 200 for k in range(0, 195)]
    lines = {"cooperate": [], "always defect": [], "defect, then make up, in turns": []}
    for w in ws:
        lines["cooperate"].append(payoffs(STRATEGIES["Always cooperate"], tft, w)[0])
        lines["always defect"].append(payoffs(STRATEGIES["Always defect"], tft, w)[0])
        lines["defect, then make up, in turns"].append(payoffs(STRATEGIES["Alternator"], tft, w)[0])
    return ws, lines


def chart_shadow(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    ws, lines = shadow_lines()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.15)
    colors = {"cooperate": c["enc"], "always defect": c["dec"], "defect, then make up, in turns": c["data"]}
    for name, ys in lines.items():
        ax.plot(ws, ys, color=colors[name], linewidth=2.2, label=name)
    for w, text in (((T - R) / (T - P), "w = 1/2"), ((T - R) / (R - S), "w = 2/3")):
        ax.axvline(w, color=c["muted"], linestyle="--", linewidth=1)
        ax.text(w + 0.008, 4.75, text, fontsize=9.5, color=c["muted"], va="top")
    ax.axvspan((T - R) / (R - S), ws[-1], color=c["enc"], alpha=0.08, linewidth=0)
    ax.set_xlim(0, ws[-1])
    ax.set_ylim(0.8, 5.1)
    ax.set_xlabel("w, the probability that there will be another round")
    ax.set_ylabel("points per round")
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    _caption(fig, "Against tit for tat, defecting pays only when the future is short",
             "what each way of playing earns per round; the relationship lasts 1/(1 − w) rounds on average", c)
    return fig


# ---------------------------------------------------------------- the tournament

def tournament_results(noise=0.0):
    names = list(STRATEGIES)
    table = tournament(names, LONG, noise)
    rows = []
    for i, name in enumerate(names):
        wins = 0
        for j in range(len(names)):
            if j != i and table[i, j] > table[j, i] + 1e-9:
                wins += 1
        rows.append({"name": name, "score": float(table[i].mean()), "wins": wins})
    rows.sort(key=lambda row: -row["score"])
    return rows


LABELS = {  # where each name goes, in data coordinates: layout only; points that nearly coincide get leaders
    "Generous ZD": (0.45, 2.525), "Generous tit for tat": (0.45, 2.47), "Always cooperate": (0.9, 2.415),
    "Tit for tat": (0.9, 2.375), "Win-stay, lose-shift": (0.9, 2.295), "Grim trigger": (2.35, 2.345),
    "Alternator": (6.3, 2.18), "Random": (5.3, 2.12), "Always defect": (9.65, 2.075),
    "Suspicious tit for tat": (8.6, 1.98), "Extortioner": (8.7, 1.9), "Joss": (7.65, 1.855),
}


def chart_tournament(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rows = tournament_results()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.12)
    for row in rows:
        nice = row["name"] in NICE
        ax.scatter(row["wins"], row["score"], s=60, color=c["enc"] if nice else c["dec"], zorder=3,
                   edgecolors=c["page"], linewidths=1)
        x, y = LABELS[row["name"]]
        right = x > row["wins"]
        ax.annotate(row["name"], (row["wins"], row["score"]), xytext=(x, y), textcoords="data", fontsize=9,
                    color=c["ink"], ha="left" if right else "right", va="center",
                    arrowprops={"arrowstyle": "-", "color": c["line"], "linewidth": 0.8,
                                "shrinkA": 2, "shrinkB": 5})
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(1.8, 2.66)
    ax.set_xlabel("matches won, out of 11")
    ax.set_ylabel("mean points per round")
    ax.text(-0.4, 2.62, "never the first to defect", color=c["enc"], fontsize=9.5, va="center")
    ax.text(11.4, 2.62, "defect first, at least sometimes", color=c["dec"], fontsize=9.5, ha="right",
            va="center")
    _caption(fig, "Winning matches and winning the tournament",
             "12 strategies, each against all the others and itself, 289 rounds on average, no mistakes", c)
    return fig


# ---------------------------------------------------------------- a mistake and its echo

ROUNDS, MISTAKE = 16, 3  # the mistake happens in round 4


def play(p, q, rng):
    # one game with one mistake: player one means to cooperate in round 4 and defects
    moves = []
    for t in range(ROUNDS):
        if t == 0:
            a_c, b_c = p[0], q[0]
        else:
            a, b = moves[-1]
            a_c = p[1 + ("CD".index(a) * 2 + "CD".index(b))]
            b_c = q[1 + ("CD".index(b) * 2 + "CD".index(a))]
        a = "C" if rng.random() < a_c else "D"
        b = "C" if rng.random() < b_c else "D"
        if t == MISTAKE:
            a = "D"
        moves.append((a, b))
    return moves


def echo_length(moves):
    # rounds after the mistake until both cooperate again
    for t in range(MISTAKE + 1, len(moves)):
        if moves[t] == ("C", "C"):
            return t - MISTAKE
    return len(moves) - MISTAKE


def generous_sample():
    # a run of median length among many, so the picture shows a typical echo, not a lucky one
    g = STRATEGIES["Generous tit for tat"]
    lengths = []
    for seed in range(2000):
        lengths.append(echo_length(play(g, g, random.Random(seed))))
    median = sorted(lengths)[len(lengths) // 2]
    for seed in range(2000):
        moves = play(g, g, random.Random(seed))
        if echo_length(moves) == median:
            return moves, median, sum(lengths) / len(lengths)


def echo() -> Figure:
    pairs = []
    for name in ("Tit for tat", "Win-stay, lose-shift"):
        pairs.append((name, play(STRATEGIES[name], STRATEGIES[name], random.Random(0))))
    moves, median, mean = generous_sample()
    pairs.insert(1, ("Generous tit for tat", moves))
    fig = Figure(
        760,
        392,
        alt="Sixteen rounds between two players of the same strategy, with one mistake in round 4: player one means "
        "to cooperate and defects. Two tit for tat players then punish each other in turns for the rest of the game. "
        f"Two generous tit for tat players stop the echo when one of them forgives, here after {median} rounds. Two "
        "win-stay, lose-shift players defect together once and are back to cooperating in round 6",
    )
    fig.text(14, 28, "One mistake between two players of the same strategy", kind="title")
    fig.text(14, 50, "player one means to cooperate in round 4 and defects; red: defect, pale: cooperate",
             kind="note")
    x0, cell, gap = 236, 26, 5
    fig.text(x0 - 10, 84, "round", kind="small", anchor="end")
    for t in range(ROUNDS):
        if t in (0, 4, 9, 15):
            fig.text(x0 + t * (cell + gap) + cell / 2, 84, str(t + 1), kind="small", anchor="middle")
    for k, (name, moves) in enumerate(pairs):
        y = 98 + k * 88
        fig.text(14, y + 22, name, kind="body")
        fig.text(14, y + 40, "player one", kind="small")
        fig.text(14, y + 40 + cell + gap, "player two", kind="small")
        for t, (a, b) in enumerate(moves):
            x = x0 + t * (cell + gap)
            for lane, move in enumerate((a, b)):
                yy = y + 22 + lane * (cell + gap)
                if move == "C":
                    fig.rect(x, yy, cell, cell, role="enc", tint=True, r=4)
                else:
                    fig.rect(x, yy, cell, cell, role="dec", r=4)
    fig.text(x0 + MISTAKE * (cell + gap) + cell / 2, 68, "mistake", kind="count_hot", anchor="middle")
    fig.text(14, 384, f"generous tit for tat: one run of median length; over 2000 runs the echo lasts {mean:.1f} "
             "rounds on average", kind="small")
    return fig


# ---------------------------------------------------------------- how forgiving to be

def forgiveness_lines():
    qs = [k / 200 for k in range(0, 121)]
    lines = {"always defect": [], "defect every other round": [], "residents, 5% mistakes": []}
    for q in qs:
        g = strategy(1, 1, q, 1, q)
        lines["always defect"].append(payoffs(STRATEGIES["Always defect"], g, 0.999)[0])
        lines["defect every other round"].append(payoffs(STRATEGIES["Alternator"], g, 0.999)[0])
        lines["residents, 5% mistakes"].append(payoffs(g, g, 0.99, 0.05)[0])
    return qs, lines


def chart_forgiveness(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    qs, lines = forgiveness_lines()
    safe = 1 - (T - R) / (R - S)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 4.1), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.7, bottom=0.25, wspace=0.28)
    left.axhline(R, color=c["enc"], linewidth=2.2, label="the residents")
    left.plot(qs, lines["always defect"], color=c["dec"], linewidth=2.2, label="invader: always defects")
    left.plot(qs, lines["defect every other round"], color=c["data"], linewidth=2.2,
              label="invader: every other round")
    right.plot(qs, lines["residents, 5% mistakes"], color=c["enc"], linewidth=2.2)
    for ax in (left, right):
        ax.axvline(safe, color=c["muted"], linestyle="--", linewidth=1)
        ax.set_xlabel("q, the chance to forgive a defection")
        ax.set_xlim(0, qs[-1])
    left.text(safe + 0.01, 3.52, "q = 1/3", fontsize=9.5, color=c["muted"])
    left.set_ylim(0.9, 3.6)
    left.set_ylabel("points per round")
    left.set_title("who gains from the forgiveness", fontsize=10.5)
    right.set_title("what it buys with 5% mistakes", fontsize=10.5)
    fig.legend(*left.get_legend_handles_labels(), frameon=False, loc="lower left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.06, 0.0))
    _caption(fig, "How forgiving to be, in a population of generous tit for tat players",
             "invaders profit once they earn more than the residents; the residents gain from every forgiven mistake",
             c)
    return fig


# ---------------------------------------------------------------- populations

COLORS_BY_STRATEGY = {"Always defect": "dec", "Tit for tat": "att", "Always cooperate": "data",
                      "Generous tit for tat": "enc"}


def chart_cycles(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    d = evolution()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.1, hspace=0.38)
    shown = 300  # the first 300 thousand steps of the run
    titles = {"without": "always defect, tit for tat, always cooperate",
              "with": "the same, and generous tit for tat"}
    for ax, key in zip(axes, ("without", "with"), strict=True):
        population = d["populations"][key]
        shares = population["run"]["shares"][: shown * 1000 // d["every"]]
        steps = [k * d["every"] / 1000 for k in range(len(shares))]
        columns = list(zip(*shares, strict=True))
        ax.stackplot(steps, columns, colors=[c[COLORS_BY_STRATEGY[n]] for n in population["names"]],
                     labels=population["names"], linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, steps[-1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50%", "100%"])
        ax.set_title(titles[key], fontsize=10.5, loc="left")
    axes[1].set_xlabel("thousands of steps: one player copies another, or tries something new")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper left", ncol=4, fontsize=9, bbox_to_anchor=(0.08, 0.88))
    _caption(fig, "A hundred players who copy what works",
             f"shares of each strategy over time; relationships of {round(1 / (1 - d['w']))} rounds on average, "
             f"{round(100 * d['noise'])}% mistakes", c)
    return fig


# ---------------------------------------------------------------- reputation

CASES = (  # the donor's act, the recipient's reputation, then the donor's new reputation under two rules
    ("helps", "good", "good", "good"),
    ("helps", "bad", "good", "good"),
    ("refuses", "good", "bad", "bad"),
    ("refuses", "bad", "bad", "good"),
)


def reputation() -> Figure:
    fig = Figure(
        760,
        300,
        alt="How an observer updates a donor's reputation. Image scoring looks only at the act: helping makes the "
        "donor good, refusing makes the donor bad. Standing also looks at whom the donor refused: refusing someone "
        "with a bad reputation keeps the donor good. The two rules differ only in that one case",
    )
    fig.text(14, 28, "What an observer concludes about a donor", kind="title")
    fig.text(14, 50, "a donor meets a recipient once; an observer updates the donor's reputation", kind="note")
    columns = ((14, "the donor"), (174, "the recipient was"), (370, "image scoring says"),
               (566, "standing says"))
    for x, label in columns:
        fig.text(x, 86, label, kind="label")
    for k, (act, recipient, scoring, standing) in enumerate(CASES):
        y = 98 + k * 46
        fig.box(14, y, 146, 36, role="enc" if act == "helps" else "dec", label=act, r=8)
        fig.box(174, y, 180, 36, role="plain", label=f"{recipient}", r=8)
        fig.box(370, y, 180, 36, role="enc" if scoring == "good" else "dec", label=scoring, r=8)
        fig.box(566, y, 180, 36, role="enc" if standing == "good" else "dec", label=standing, r=8,
                strong=scoring != standing)
    fig.text(14, 292, "The two rules differ in one case: refusing someone with a bad reputation.", kind="body")
    return fig


# ---------------------------------------------------------------- the stag hunt

STAG, HARE, ALONE = 4, 3, 0  # both hunt the stag; hunting hare, whatever the other does; hunting the stag alone


def chart_stag(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    ps = [k / 100 for k in range(101)]
    threshold = (HARE - ALONE) / (STAG - ALONE)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.15)
    ax.plot(ps, [p * STAG + (1 - p) * ALONE for p in ps], color=c["enc"], linewidth=2.2, label="hunt the stag")
    ax.plot(ps, [HARE for _ in ps], color=c["data"], linewidth=2.2, label="hunt hare")
    ax.axvline(threshold, color=c["muted"], linestyle="--", linewidth=1)
    ax.text(threshold - 0.01, 0.25, "p = 3/4", fontsize=9.5, color=c["muted"], ha="right")
    ax.annotate("", xy=(0.03, 4.55), xytext=(threshold - 0.03, 4.55),
                arrowprops={"arrowstyle": "->", "color": c["data"], "linewidth": 1.6})
    ax.annotate("", xy=(0.97, 4.55), xytext=(threshold + 0.03, 4.55),
                arrowprops={"arrowstyle": "->", "color": c["enc"], "linewidth": 1.6})
    ax.text(threshold / 2, 4.7, "fewer stag hunters each season", fontsize=9, color=c["data"], ha="center")
    ax.text((1 + threshold) / 2, 4.7, "more", fontsize=9, color=c["enc"], ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.1)
    ax.set_xlabel("p, the share of others who hunt the stag")
    ax.set_ylabel("expected catch")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    _caption(fig, "In the stag hunt, trust pays only if enough others trust",
             f"stag together {STAG}, hare alone {HARE}, stag alone {ALONE}; players copy whatever caught more", c)
    return fig


# ---------------------------------------------------------------- alts and the cover

def chart_alts():
    rows = tournament_results()
    best, worst = rows[0], rows[-1]
    defect = next(r for r in rows if r["name"] == "Always defect")
    d = evolution()["populations"]
    return {
        "shadow": "Points per round against tit for tat as a function of w, the probability of another round. "
        "Cooperating earns 3. Always defecting earns 5 - 4w and wins below w = 1/2; defecting and making up in "
        "turns earns 5/(1 + w) and wins below w = 2/3. Above 2/3 cooperating is the best reply",
        "tournament": f"Matches won against mean points per round for twelve strategies. {best['name']} scores "
        f"highest, {best['score']:.2f}, winning {best['wins']} matches; the nice strategies take the top places "
        f"without winning any. Always defect wins {defect['wins']} of 11 matches and scores {defect['score']:.2f}; "
        f"{worst['name']} is last with {worst['score']:.2f}",
        "forgiveness": "Left: residents who forgive a defection with probability q earn 3; an invader who "
        "defects every other round earns more once q passes 1/3, one who always defects once q passes 1/2. "
        "Right: with 5 percent mistakes the residents earn more the more they forgive",
        "cycles": "Shares of strategies in a population of a hundred over time. Without generous tit for tat the "
        "population cycles: defectors are replaced by tit for tat, which drifts into unconditional cooperation, "
        f"which defectors take over again; over twenty runs defectors hold the majority "
        f"{d['without']['defectors_in_majority']:.0%} of the time. With generous tit for tat it mostly holds the "
        f"population, and defectors hold the majority {d['with']['defectors_in_majority']:.0%} of the time",
        "stag": "Expected catch of hunting the stag (4 times the share p of others who hunt it) and of hunting hare "
        "(3). Hunting the stag pays when more than three quarters of the others hunt it; below that the share of "
        "stag hunters falls, above it it rises",
    }


def cover():
    tft = STRATEGIES["Tit for tat"]
    rows = [play(tft, tft, random.Random(0)), generous_sample()[0]]
    out = [cv.label(cv.MOTIF_X0, 128, "ONE MISTAKE, TWO STRATEGIES")]
    cell, gap = 24, 6
    for k, (label, moves) in enumerate(zip(("TIT FOR TAT", "GENEROUS TIT FOR TAT"), rows, strict=True)):
        y = 176 + k * 150
        out.append(cv.label(cv.MOTIF_X0, y, label))
        for t, pair in enumerate(moves):
            x = cv.MOTIF_X0 + t * (cell + gap)
            for lane, move in enumerate(pair):
                yy = y + 18 + lane * (cell + gap)
                if move == "D":
                    out.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" rx="4" fill="{cv.RED}"/>')
                else:
                    out.append(f'<rect x="{x}" y="{yy}" width="{cell}" height="{cell}" rx="4" fill="none" '
                               f'stroke="{cv.LINE}" stroke-width="2"/>')
    cv.fits(cv.MOTIF_X0 + ROUNDS * (cell + gap) - gap)
    title, hook = ["Is it rational", "to be good?"], ["Game theory and the", "ethics of cooperation"]
    return cv.frame("blog / philosophy", title, hook, "".join(out), footer="The repeated prisoner's dilemma")


COVER = cover

FIGURES = {
    "dilemma": dilemma,
    "echo": echo,
    "reputation": reputation,
}

CHARTS = {
    "shadow": chart_shadow,
    "tournament": chart_tournament,
    "forgiveness": chart_forgiveness,
    "cycles": chart_cycles,
    "stag": chart_stag,
}
