"""Static figures and covers of lesson 6.

The search figures are toys whose numbers are computed here, with fixed seeds: a two-dimensional objective for
grid against random search, the budget arithmetic of successive halving, and a Gaussian-process fit for Bayesian
optimization. The GPT-3 figure shows the table of the paper (Brown et al., 2020, Table 2.1).
"""

import math
import random

from tools.figures import cover as cv
from tools.figures.lib import Figure

# ---------------------------------------------------------------- hero: learning rates of real models

GPT3 = [  # Brown et al. (2020), Table 2.1: parameters and learning rate of the eight GPT-3 models
    ("125M", 125e6, 6.0e-4), ("350M", 350e6, 3.0e-4), ("760M", 760e6, 2.5e-4), ("1.3B", 1.3e9, 2.0e-4),
    ("2.7B", 2.7e9, 1.6e-4), ("6.7B", 6.7e9, 1.2e-4), ("13B", 13.0e9, 1.0e-4), ("175B", 175e9, 0.6e-4),
]
LLAMA3_405B = (405e9, 8e-5)  # Llama Team (2024), §3.4.1: the peak learning rate of the 405B model


def hero() -> Figure:
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    repo = "xmarva/tfs-transformer-en-ru"
    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        ours = json.load(f)
    ours_params = 0
    with safe_open(hf_hub_download(repo, "model.safetensors"), framework="pt") as weights:
        for name in weights.keys():
            if name != "pe":  # the sinusoids are not learned
                ours_params += math.prod(weights.get_slice(name).get_shape())
    ours_lr = ours["d_model"] ** -0.5 * 4000 ** -0.5  # the peak of lesson 3's schedule
    points = [(label, p, lr, "data") for label, p, lr in GPT3]
    points.append(("Llama 3 405B", LLAMA3_405B[0], LLAMA3_405B[1], "dec"))
    points.append(("ours, lesson 3", ours_params, ours_lr, "enc"))
    fig = Figure(
        760,
        380,
        alt="Peak learning rates against model size: the eight GPT-3 models fall from 6e-4 at 125 million parameters "
        "to 6e-5 at 175 billion; Llama 3 405B uses 8e-5; our ten-million-parameter model trained at about 1e-3. "
        "Bigger models take smaller learning rates",
    )
    fig.text(14, 28, "Bigger models, smaller learning rates", kind="title")
    fig.text(14, 50, "peak learning rate against parameters; GPT-3: its paper's Table 2.1, Llama 3: §3.4.1",
             kind="note")
    x0, x1, y0, y1 = 70, 720, 300, 84

    def px(params):
        return x0 + (x1 - x0) * (math.log10(params) - 6.5) / (12 - 6.5)

    def py(lr):
        return y0 + (y1 - y0) * (math.log10(lr) - math.log10(3e-5)) / (math.log10(2e-3) - math.log10(3e-5))

    fig.line((x0, y0), (x1, y0), role="muted", width=1)
    fig.line((x0, y0), (x0, y1), role="muted", width=1)
    for exp in range(7, 13):
        fig.text(px(10**exp), y0 + 20, f"10^{exp}", kind="small", anchor="middle")
    for lr, label in [(1e-4, "1e-4"), (1e-3, "1e-3")]:
        fig.text(x0 - 8, py(lr) + 4, label, kind="small", anchor="end")
    gpt = [pt for pt in points if pt[3] == "data"]
    for a, b in zip(gpt, gpt[1:]):
        fig.line((px(a[1]), py(a[2])), (px(b[1]), py(b[2])), role="data", width=1.4)
    for _, params, lr, role in points:
        fig.rect(px(params) - 5, py(lr) - 5, 10, 10, role=role, r=5)
    fig.text(px(125e6) + 10, py(6.0e-4) - 10, "GPT-3, 125M", kind="label")
    fig.text(px(175e9) - 10, py(0.6e-4) + 22, "GPT-3, 175B", kind="label", anchor="end")
    fig.text(px(405e9) + 4, py(8e-5) - 14, "Llama 3, 405B", kind="label", anchor="end")
    fig.text(px(ours_params) + 10, py(ours_lr) + 4, "ours (lesson 3)", kind="label")
    fig.text(x0, 356, "parameters, log scale", kind="label")
    return fig


# ---------------------------------------------------------------- grid against random search


def _objective(x, y):
    """A toy loss over two hyperparameters in [0, 1]; only x matters much (the learning rate, say)."""
    return 1.6 - 0.6 * math.exp(-((x - 0.68) ** 2) / (2 * 0.08**2)) + 0.05 * y


def _random_wins(draws=1000):
    """How often 9 random trials beat the 3 × 3 grid, over many draws."""
    grid = [((i + 0.5) / 3, (j + 0.5) / 3) for i in range(3) for j in range(3)]
    grid_best = min(_objective(*pt) for pt in grid)
    rng = random.Random(0)
    wins = 0
    for _ in range(draws):
        trials = [(rng.random(), rng.random()) for _ in range(9)]
        if min(_objective(*pt) for pt in trials) < grid_best:
            wins += 1
    return wins / draws


def grid_random() -> Figure:
    rng = random.Random(3)
    grid = [((i + 0.5) / 3, (j + 0.5) / 3) for i in range(3) for j in range(3)]
    rand = [(rng.random(), rng.random()) for _ in range(9)]
    xs = [k / 200 for k in range(201)]
    curve = [_objective(x, 0.5) for x in xs]
    best_x = xs[curve.index(min(curve))]
    fig = Figure(
        760,
        418,
        alt="Nine trials of grid search and nine of random search on a toy problem where only the horizontal "
        "hyperparameter matters. The grid tries only three distinct values of it and misses the minimum; random "
        "search tries nine different values and lands close to it",
    )
    fig.text(14, 28, "Nine trials: a grid tries 3 values of what matters, random search 9", kind="title")
    fig.text(14, 50, "toy loss; only the horizontal hyperparameter matters (above each panel: its effect)",
             kind="note")
    lo, hi = min(curve), max(curve)
    for p, (title, points) in enumerate([("grid search", grid), ("random search", rand)]):
        x0, y0, size = 30 + p * 380, 150, 190
        fig.text(x0, 82, title, kind="subtitle")
        prev = None
        for x, c in zip(xs, curve):  # the effect of the important hyperparameter, drawn above the square
            pt = (x0 + x * size, y0 - 12 - 44 * (c - lo) / (hi - lo))  # low loss at the bottom
            if prev:
                fig.line(prev, pt, role="muted", width=1.4)
            prev = pt
        fig.line((x0 + best_x * size, y0 - 60), (x0 + best_x * size, y0 + size), role="dec", width=1,
                 dash=True)
        fig.rect(x0, y0, size, size, role="data", tint=True, r=4)
        for x, y in points:
            fig.rect(x0 + x * size - 5, y0 + (1 - y) * size - 5, 10, 10, role="data", r=5)
            fig.line((x0 + x * size, y0 + size), (x0 + x * size, y0 + size + 10), role="data", width=2)
        best = min(points, key=lambda pt: _objective(*pt))
        fig.text(x0 + 200, y0 + 100, "best trial:", kind="small")
        fig.text(x0 + 200, y0 + 116, f"loss {_objective(*best):.3f}", kind="count")
        fig.text(x0, y0 + size + 28, "important →", kind="small")
    fig.text(14, 384, "dashed: the true minimum", kind="note")
    fig.text(14, 402, f"Over 1,000 draws of 9 random trials, random search beats this grid in {_random_wins():.0%} "
             "of them.", kind="note")
    return fig


# ---------------------------------------------------------------- successive halving

N, ETA, RUNGS, FULL = 18, 3, [0.5, 1.5, 4.5], 20  # configurations, keep 1/ETA, cumulative epochs per rung


def halving() -> Figure:
    alive, cost, rows = N, 0.0, []
    previous = 0.0
    for budget in RUNGS:
        rows.append((alive, budget))
        cost += alive * (budget - previous)
        previous = budget
        alive = max(1, alive // ETA)
    cost += FULL - RUNGS[-1]  # the survivor trains to the end
    everyone = N * FULL
    fig = Figure(
        760,
        330,
        alt=f"Successive halving with {N} configurations: all train for {RUNGS[0]} epoch, the best third continues to "
        f"{RUNGS[1]} epochs, the best third of those to {RUNGS[2]}, and the winner to {FULL}. The search costs "
        f"{cost:.0f} epochs instead of {everyone} for training every configuration to the end",
    )
    fig.text(14, 28, "Successive halving: everyone starts, the best third goes on", kind="title")
    for r, (count, budget) in enumerate(rows):
        y = 70 + r * 56
        fig.text(14, y + 20, f"{count} models", kind="mono")
        fig.text(120, y + 20, f"to {budget:g} epochs", kind="label")
        for k in range(count):
            x = 260 + k * 26
            survives = r < len(rows) - 1 and k < count // ETA
            winner = r == len(rows) - 1 and k == 0
            fig.rect(x, y + 4, 20, 22, role="att" if survives or winner else "data", tint=not (survives or winner),
                     r=3)
    y = 70 + len(rows) * 56
    fig.text(14, y + 20, "1 model", kind="mono")
    fig.text(120, y + 20, f"to {FULL} epochs", kind="label")
    fig.rect(260, y + 4, 20, 22, role="att", r=3)
    fig.text(14, 316, f"cost: {cost:.0f} epochs of training, against {everyone} to train all {N} to the end",
             kind="body")
    return fig


# ---------------------------------------------------------------- Bayesian optimization, one dimension


def _gp(train_x, train_y, xs, length=0.12, noise=1e-4):
    """Gaussian-process regression with an RBF kernel, written out for a handful of points."""
    import numpy as np

    def kernel(a, b):
        return np.exp(-((np.asarray(a)[:, None] - np.asarray(b)[None, :]) ** 2) / (2 * length**2))

    k = kernel(train_x, train_x) + noise * np.eye(len(train_x))
    k_star = kernel(xs, train_x)
    mean_y = np.mean(train_y)
    alpha = np.linalg.solve(k, np.asarray(train_y) - mean_y)
    mean = mean_y + k_star @ alpha
    var = 1 - np.einsum("ij,ji->i", k_star, np.linalg.solve(k, k_star.T))
    return mean, np.sqrt(np.clip(var, 1e-12, None))


def bayes() -> Figure:
    import numpy as np
    from math import erf

    def loss(x):
        return 0.6 + 0.35 * math.sin(7 * x + 0.6) + 0.5 * (x - 0.55) ** 2

    tried = [0.1, 0.45, 0.8, 0.95]
    ys = [loss(x) for x in tried]
    xs = np.linspace(0, 1, 120)
    mean, std = _gp(tried, ys, xs)
    best = min(ys)
    ei = []
    for m, s in zip(mean, std):  # expected improvement over the best loss so far
        z = (best - m) / s
        cdf = 0.5 * (1 + erf(z / math.sqrt(2)))
        pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        ei.append((best - m) * cdf + s * pdf)
    next_x = float(xs[int(np.argmax(ei))])
    fig = Figure(
        760,
        380,
        alt=f"Bayesian optimization in one dimension: after {len(tried)} trials, a Gaussian process gives a mean "
        "estimate of the loss and a band of uncertainty; expected improvement is highest where the estimate is "
        f"low or the uncertainty large, and the next trial goes there, at {next_x:.2f}",
    )
    fig.text(14, 28, "Bayesian optimization: a model of the loss decides the next trial", kind="title")
    x0, w = 40, 680
    top, h = 70, 180
    lo, hi = float(min(mean - 2 * std)), float(max(mean + 2 * std))

    def px(x):
        return x0 + x * w

    def py(y):
        return top + h * (hi - y) / (hi - lo)

    for k in range(len(xs) - 1):  # the uncertainty band, as thin columns
        y_top, y_bottom = py(mean[k] + 2 * std[k]), py(mean[k] - 2 * std[k])
        fig.rect(px(xs[k]), y_top, w / len(xs) + 0.6, max(1, y_bottom - y_top), role="att", tint=True, r=0)
    for k in range(len(xs) - 1):
        fig.line((px(xs[k]), py(mean[k])), (px(xs[k + 1]), py(mean[k + 1])), role="att", width=2)
        fig.line((px(xs[k]), py(loss(xs[k]))), (px(xs[k + 1]), py(loss(xs[k + 1]))), role="muted", width=1,
                 dash=True)
    for x, y in zip(tried, ys):
        fig.rect(px(x) - 5, py(y) - 5, 10, 10, role="data", r=5)
    fig.text(x0, top - 6, "loss: the true curve (dashed), the model's estimate and its uncertainty", kind="label")
    base, eh = 330, 60
    peak = max(ei)
    for k in range(len(xs) - 1):
        fig.line((px(xs[k]), base - eh * ei[k] / peak), (px(xs[k + 1]), base - eh * ei[k + 1] / peak),
                 role="enc", width=2)
    fig.line((px(next_x), base - eh - 6), (px(next_x), base + 4), role="dec", width=1.4, dash=True)
    fig.text(px(next_x) + 6, base - eh, "next trial", kind="label", role="dec")
    fig.text(x0, base + 24, "expected improvement over the best trial so far", kind="label")
    return fig


# ---------------------------------------------------------------- W&B Sweeps


def sweep() -> Figure:
    fig = Figure(
        760,
        300,
        alt="How a W&B sweep runs: the sweep controller on W&B's servers holds the search space and the method; "
        "agents on one or more machines ask it for a configuration, train with it, and log the metric back; the "
        "controller uses the results to choose the next configuration and can stop runs that fall behind",
    )
    fig.text(14, 28, "A W&B sweep: one controller, any number of agents", kind="title")
    fig.box(14, 110, 230, 90, role="neutral", label="")
    fig.text(129, 140, "sweep controller", kind="block", role="att", anchor="middle")
    fig.text(129, 162, "on W&B's servers", kind="label", anchor="middle")
    fig.text(129, 182, "space · method · stopping", kind="small", anchor="middle")
    for k, name in enumerate(["agent: your Kaggle GPU", "agent: your Colab GPU", "agent: any machine"]):
        y = 70 + k * 72
        fig.box(470, y, 276, 44, role="enc", label=name)
        fig.arrow((248, 140 + k * 6), (466, y + 14), role="att")
        fig.arrow((466, y + 30), (248, 160 + k * 6), role="data")
    fig.text(300, 104, "a configuration", kind="label", role="att")
    fig.text(300, 226, "the metric, every epoch", kind="label", role="data")
    fig.text(14, 286, "Agents can join and leave at any time; the controller decides what each one trains next.",
             kind="note")
    return fig


# ---------------------------------------------------------------- covers: successive halving, drawn big


def _cover(lang):
    english = lang == "en"
    out = [cv.label(cv.MOTIF_X0, 128, "18 CONFIGURATIONS, ONE GPU" if english else "18 КОНФИГУРАЦИЙ, ОДИН GPU")]
    alive = N
    for r, budget in enumerate(RUNGS + [FULL]):
        y = 160 + r * 80
        count = alive if r < len(RUNGS) else 1
        for k in range(count):
            x = cv.MOTIF_X0 + k * 20
            on = (r < len(RUNGS) - 1 and k < count // ETA) or r >= len(RUNGS) - 1 and k == 0
            fill = cv.RED if on else cv.BG
            out.append(f'<rect x="{x}" y="{y}" width="15" height="36" rx="3" fill="{fill}" stroke="{cv.RED}" '
                       'stroke-width="1.6"/>')
        label = f"{budget:g} epochs" if english else f"{budget:g} эпох"
        out.append(cv.text(cv.MOTIF_X1 - 4, y + 26, label, 16, cv.MUTED, cv.MONO, anchor="end"))
        alive = max(1, alive // ETA)
    cv.fits(cv.MOTIF_X0 + 18 * 20)
    title = ["Hyperparameter", "tuning"] if english else ["Подбор гипер-", "параметров"]
    hook = ["An hour of a free GPU,", "spent where it counts"] if english else ["Час бесплатного GPU,", "потраченный с толком"]
    return cv.frame("06_sweeps.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "grid-random": grid_random,
    "halving": halving,
    "bayes": bayes,
    "sweep": sweep,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
