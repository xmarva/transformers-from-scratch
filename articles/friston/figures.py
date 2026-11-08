"""Figures, charts and the cover of the post on Friston's free energy principle on xmarva.github.io.

The worked example is Bogacz's (2017): a creature infers the size v of a food item from light intensity u, with
g(v) = v^2, a prior N(3, 1), sensory variance 1 and an observation u = 2. Everything in the charts is computed here.
The expected-free-energy numbers follow Smith, Friston & Whyte (2022), section 2.4.

    uv run --group articles python tools/render.py --article friston --out <site>/assets/posts/<slug>
"""

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

VP, SP, SU, U = 3.0, 1.0, 1.0, 2.0  # prior mean and variance, sensory variance, observed light intensity
V = np.linspace(-3, 7, 20001)  # sizes, for integrals on a grid
DV = V[1] - V[0]


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


def log_joint(v):
    """ln p(u, v) = ln p(v) + ln p(u | v) for the observed u."""
    prior = -0.5 * np.log(2 * np.pi * SP) - (v - VP) ** 2 / (2 * SP)
    likelihood = -0.5 * np.log(2 * np.pi * SU) - (U - v ** 2) ** 2 / (2 * SU)
    return prior + likelihood


def surprisal():
    return -np.log(np.sum(np.exp(log_joint(V))) * DV)


def free_energy(mean, sd):
    """F[q, u] = E_q[ln q(v) - ln p(u, v)] for a Gaussian belief q = N(mean, sd^2), by integration on the grid."""
    log_q = -(V - mean) ** 2 / (2 * sd * sd) - 0.5 * np.log(2 * np.pi * sd * sd)
    return np.sum(np.exp(log_q) * (log_q - log_joint(V))) * DV


# ---------------------------------------------------------------- free energy bounds surprisal

def chart_bound(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    means = np.linspace(0.8, 2.4, 81)
    sds = np.linspace(0.05, 1.5, 146)
    best = [min(free_energy(m, s) for s in sds) for m in means]
    fixed = [free_energy(m, 0.6) for m in means]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.15)
    ax.axhline(surprisal(), color=c["dec"], linewidth=1.6, label="surprisal, −ln p(u)")
    ax.plot(means, fixed, color=c["line"], linewidth=1.4, label="F for beliefs of width 0.6")
    ax.plot(means, best, color=c["att"], linewidth=2, label="F for the best width at each centre")
    ax.set_ylim(2.5, 8)
    ax.set_xlabel("centre of the belief q about the food's size")
    ax.set_ylabel("nats")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    _caption(fig, "Free energy is never below surprisal; good beliefs nearly touch it",
             "Gaussian beliefs about the size of the food in Bogacz's example; F computed by integration", c)
    return fig


# ---------------------------------------------------------------- perception as gradient descent

def chart_bogacz(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    grid = np.linspace(0.01, 5, 2000)
    posterior = np.exp(log_joint(grid))
    posterior /= np.sum(posterior) * (grid[1] - grid[0])
    mode = grid[np.argmax(posterior)]
    mean = np.sum(grid * posterior) * (grid[1] - grid[0])
    dt, steps = 0.01, 1000
    phi = VP
    path = [phi]
    for _ in range(steps):  # Bogacz's exercise 2: gradient ascent on ln p(phi) + ln p(u | phi)
        phi += dt * ((VP - phi) / SP + (U - phi ** 2) / SU * 2 * phi)
        path.append(phi)
    phi, e_p, e_u = VP, 0.0, 0.0
    network = [phi]
    for _ in range(2000):  # exercise 3: the same with prediction-error nodes
        phi, e_p, e_u = (phi + dt * (e_u * 2 * phi - e_p), e_p + dt * (phi - VP - SP * e_p),
                         e_u + dt * (U - phi ** 2 - SU * e_u))
        network.append(phi)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.17, wspace=0.28)
    axes[0].plot(grid, posterior, color=c["att"])
    axes[0].axvline(mode, color=c["att"], linestyle="--", linewidth=1)
    axes[0].axvline(mean, color=c["muted"], linestyle=":", linewidth=1)
    axes[0].text(2.5, posterior.max() * 0.9, f"dashed: most probable, {mode:.2f}", fontsize=8.5, color=c["muted"])
    axes[0].text(2.5, posterior.max() * 0.76, f"dotted: mean, {mean:.2f}", fontsize=8.5, color=c["muted"])
    axes[0].set_xlabel("size of the food, v")
    axes[0].set_title("exact posterior, by Bayes' rule", fontsize=10)
    axes[0].set_yticks([])
    t = np.arange(len(path)) * dt
    axes[1].plot(np.arange(len(network)) * dt, network, color=c["line"], linewidth=1.4,
                 label="with prediction-error units")
    axes[1].plot(t, path, color=c["enc"], linewidth=2, label="gradient of free energy")
    axes[1].axhline(mode, color=c["att"], linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 10)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("the creature's guess, φ")
    axes[1].set_title("the guess, starting from the prior 3", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8.5)
    _caption(fig, "Perception as descent on free energy finds the most probable size",
             "Bogacz's example: light u = 2, prior mean 3, both variances 1, g(v) = v²; computed here", c)
    return fig


# ---------------------------------------------------------------- the decompositions

def decompose() -> Figure:
    fig = Figure(
        760, 300,
        alt="Three ways to write the same variational free energy. First: the divergence between the belief and the "
        "true posterior, which is never negative, plus the surprisal of the observation, so free energy is an upper "
        "bound on surprisal. Second: complexity, how far the belief moved from the prior, minus accuracy, how well "
        "the believed causes predict the observation. Third: the negative of the evidence lower bound that "
        "variational autoencoders maximise",
    )
    fig.text(14, 28, "One quantity, three readings", kind="title")
    fig.text(14, 50, "F[q, o] = E_q[ ln q(x) − ln p(o, x) ]", kind="mono")
    rows = [
        ("as a bound", [("divergence from the true posterior, ≥ 0", "att"), ("+  surprisal, −ln p(o)", "dec")],
         "so F ≥ surprisal, equal only when the belief is exact"),
        ("as a trade-off", [("complexity: belief vs prior", "data"), ("−  accuracy: fit to the data", "enc")],
         "the simplest explanation that fits: Occam's razor"),
        ("in machine learning", [("−  ELBO", "neutral")], "what a variational autoencoder maximises"),
    ]
    for k, (name, parts, note) in enumerate(rows):
        y = 78 + k * 70
        fig.text(14, y + 20, name, kind="subtitle")
        x = 190
        for label, role in parts:
            w = 268 if len(parts) > 1 else 120
            fig.box(x, y, w, 30, role=role, label=label)
            x += w + 10
        fig.text(190, y + 52, note, kind="note")
    return fig


# ---------------------------------------------------------------- predictive coding hierarchy

def hierarchy() -> Figure:
    fig = Figure(
        760, 340,
        alt="A predictive-coding hierarchy of three levels above the senses. Each level sends a prediction down to "
        "the level below and receives back the prediction error, what the prediction failed to explain, weighted "
        "by its precision. Precision works like a volume knob on each error: turned up, the guess follows the data; "
        "turned down, it follows expectations. In Friston's account attention is precision",
    )
    fig.text(14, 28, "Predictive coding: predictions go down, weighted errors go up", kind="title")
    fig.text(14, 50, "Rao & Ballard (1999); Friston (2005); precision as gain", kind="note")
    levels = ["level 3: objects, scenes", "level 2: shapes, textures", "level 1: edges", "the senses"]
    for k, name in enumerate(levels):
        y = 74 + k * 66
        fig.box(14, y, 260, 36, role="data" if k == 3 else "neutral", label=name)
        if k < 3:
            fig.arrow((110, y + 37), (110, y + 65), role="att")
            fig.arrow((180, y + 65), (180, y + 37), role="dec")
    fig.text(100, 128, "prediction", kind="small", role="att", anchor="end")
    fig.text(190, 128, "error × precision", kind="small", role="dec")
    fig.text(330, 96, "each level keeps a guess and two errors:", kind="body")
    fig.text(330, 118, "from below, the data it failed to predict;", kind="body")
    fig.text(330, 140, "from above, how far it drifted from the prior", kind="body")
    fig.text(330, 186, "precision, the inverse of variance, is a volume", kind="body")
    fig.text(330, 208, "knob on each error:", kind="body")
    fig.box(330, 222, 200, 30, role="dec", label="senses loud: follow data")
    fig.box(540, 222, 206, 30, role="att", label="prior loud: follow belief")
    fig.text(330, 284, "in Friston's account, attending to something", kind="note")
    fig.text(330, 306, "means turning up the precision of its errors", kind="note")
    return fig


# ---------------------------------------------------------------- expected free energy

def chart_efe(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)

    def risk(a, s, pref):
        predicted = a @ s
        return float(np.sum(predicted * (np.log(predicted) - np.log(pref))))

    def ambiguity(a, s):
        entropies = -np.sum(a * np.log(a), axis=0)
        return float(entropies @ s)

    a_risk = np.array([[0.9, 0.1], [0.1, 0.9]])
    preferences = np.array([1.0, np.exp(-16)])
    risks = [risk(a_risk, np.array([0.9, 0.1]), preferences), risk(a_risk, np.array([0.5, 0.5]), preferences)]
    a_amb = np.array([[0.4, 0.2], [0.6, 0.8]])
    ambiguities = [ambiguity(a_amb, np.array([0.9, 0.1])), ambiguity(a_amb, np.array([0.1, 0.9]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.2, wspace=0.35)
    panels = [(axes[0], risks, "risk: how far from preferred outcomes",
               ["policy 1\nlikely to the goal", "policy 2\na coin flip"]),
              (axes[1], ambiguities, "ambiguity: how murky the view",
               ["policy 1\ntoward the murky state", "policy 2\ntoward the clear one"])]
    for ax, values, title, names in panels:
        ax.bar([0, 1], values, color=[c["att"], c["dec"]], width=0.55)
        for k, value in enumerate(values):
            ax.text(k, value * 1.03, f"{value:.2f}", ha="center", fontsize=9, color=c["muted"])
        ax.set_xticks([0, 1], names, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
    _caption(fig, "Expected free energy = risk + ambiguity, in nats; lower is chosen",
             "two-state examples from Smith, Friston & Whyte (2022), recomputed here", c)
    return fig


# ---------------------------------------------------------------- the Markov blanket

def blanket() -> Figure:
    fig = Figure(
        760, 330,
        alt="A Markov blanket. External states act on sensory states; sensory states act on internal states; "
        "internal states act on active states; active states act on external states. Sensory and active states form "
        "the blanket: given the blanket, internal and external states are independent. A cell membrane is an "
        "example; a candle flame, whose molecules are swept away almost instantly, is not",
    )
    fig.text(14, 28, "A Markov blanket: what separates a thing from its world", kind="title")
    fig.text(14, 50, "internal and external states touch only through the blanket (Friston, 2013, 2019)", kind="note")
    fig.box(14, 130, 150, 50, role="neutral", label="external η")
    fig.box(230, 80, 200, 50, role="data", label="sensory s", strong=True)
    fig.box(230, 180, 200, 50, role="data", label="active a", strong=True)
    fig.box(496, 130, 150, 50, role="enc", label="internal μ")
    fig.arrow((165, 145), (229, 108))
    fig.arrow((431, 108), (495, 145))
    fig.arrow((495, 165), (431, 202))
    fig.arrow((229, 202), (165, 165))
    fig.line((218, 70), (442, 70), (442, 240), (218, 240), (218, 70), role="data", dash=True)
    fig.text(330, 262, "the blanket: sensory and active states", kind="label", role="data", anchor="middle")
    fig.text(14, 296, "μ ⊥ η | (s, a): once the blanket is known, the inside tells you nothing more about the outside",
             kind="note")
    fig.text(14, 318, "a cell membrane is one; a candle flame isn't, its molecules are swept away almost at once",
             kind="note")
    return fig


# ---------------------------------------------------------------- the scorecard

def evidence() -> Figure:
    rows = [
        ("free energy bounds surprisal", "a derivation", "true", "enc"),
        ("brains signal expectation violations", "mismatch negativity; mouse L2/3 neurons", "supported", "enc"),
        ("cortex runs hierarchical predictive coding", "reviews say mixed or modest", "mixed", "data"),
        ("neurons in a dish minimise free energy", "one preparation, <4% error", "supported once", "data"),
        ("aberrant precision in psychosis, autism", "fits some data; direction disputed", "mixed", "data"),
        ("active inference beats RL in AI", "small benchmarks, no clear wins", "weak", "dec"),
        ("every persisting system does it (2019)", "holds in narrow cases", "contested", "dec"),
        ("the principle itself", "a principle, by design", "not testable", "neutral"),
    ]
    fig = Figure(
        760, 84 + len(rows) * 36 + 6,
        alt="A scorecard of claims. That free energy bounds surprisal is a derivation and true. That brains signal "
        "violations of expectation is supported by the mismatch negativity and by mismatch neurons in mouse visual "
        "cortex. That cortex runs hierarchical predictive coding has mixed or modest support. That neurons in a dish "
        "minimise free energy is supported in one preparation. Aberrant precision in psychosis and autism: mixed. "
        "That active inference beats reinforcement learning in AI: weak. That every persisting system minimises "
        "free energy, the 2019 physics: contested, holding in narrow cases. The principle itself: not testable, by "
        "design",
    )
    fig.text(14, 28, "What the free energy framework claims, and how the evidence stands", kind="title")
    fig.text(14, 50, "from the reviews and studies cited in this post", kind="note")
    for k, (claim, evidence_text, status, role) in enumerate(rows):
        y = 70 + k * 36
        fig.text(14, y + 19, claim, kind="body")
        fig.text(342, y + 19, evidence_text, kind="small")
        fig.box(626, y, 120, 28, role=role, label=status)
    return fig


def chart_alts():
    return {
        "bound": "Free energy against the centre of a Gaussian belief about the size of the food. A horizontal line "
        "marks the surprisal of the observation, about 3.24 nats. Free energy for beliefs of a fixed width and for "
        "the best width at each centre always stays above the line; the best belief, centred near 1.5, comes within "
        "a few hundredths of a nat of it",
        "bogacz": "Left: the exact posterior over the food's size, peaked at 1.57 with a mean of 1.48. Right: the "
        "creature's guess over time, starting from its prior expectation of 3. Following the gradient of free energy "
        "it settles at 1.57 within about one time unit; a network with explicit prediction-error units oscillates "
        "and settles at the same value later",
        "efe": "Two bar charts. Risk: a policy expected to reach the preferred outcome 82% of the time has risk 2.41 "
        "nats, a policy that is a coin flip 7.31. Ambiguity: a policy heading toward the murky state has ambiguity "
        "0.66 nats, one heading toward the clearly observed state 0.52. Lower expected free energy is preferred",
    }


def cover():
    out = [cv.label(cv.MOTIF_X0, 150, "ONE QUANTITY")]
    svg, spans, right = cv.chip_row(["F ≥ −ln p(o)"], cv.MOTIF_X0, 172, size=18, kinds={0: "hot"})
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 262, "MINIMISED THREE WAYS"))
    svg, spans, right = cv.chip_row(["perceive", "learn", "act"], cv.MOTIF_X0, 284, size=18)
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 374, "TESTED, PARTLY"))
    svg, spans, right = cv.chip_row(["mixed", "contested", "untestable"], cv.MOTIF_X0, 396, size=17,
                                    kinds={0: "warm", 1: "warm", 2: "warm"})
    cv.fits(right)
    out.append(svg)
    return cv.frame("blog / cognitive science", ["The free energy", "principle"],
                    ["Friston's theory: the maths,", "the evidence, the critics"], "".join(out),
                    footer="Cognitive science, measured")


CHARTS = {"bound": chart_bound, "bogacz": chart_bogacz, "efe": chart_efe}
FIGURES = {"decompose": decompose, "hierarchy": hierarchy, "blanket": blanket, "evidence": evidence}
COVER = cover
