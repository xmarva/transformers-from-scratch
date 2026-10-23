"""Figures, charts and the cover of the post "Algebraic Foundations of Low-Rank Adaptation" on xmarva.github.io.

The charts about linear algebra are computed here, on matrices small enough to look at (2x2 maps, a random 48-column
projection) and on a photograph: skimage.data.astronaut(), Eileen Collins, NASA, in the public domain. The charts
about LoRA read data/lora.json, the results of experiment.py on a Kaggle T4.

    uv run --group articles python tools/render.py --article lora \
        --out <site>/assets/posts/lora-algorithm-for-llms
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

MONO = ["IBM Plex Mono", "DejaVu Sans Mono"]  # Plex Mono has no Greek letters

DATA = Path(__file__).parent / "data"
D = 768  # bert-base-uncased


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


def _signed(c):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("signed", [c["dec"], c["card"], c["enc"]])


# ---------------------------------------------------------------- what a matrix does to the plane

MAPS = [  # three 2x2 matrices, from comfortably invertible to rank one
    ("rank 2", np.array([[1.5, 0.5], [0.2, 0.9]])),
    ("rank 2, nearly 1", np.array([[1.3, 1.0], [0.75, 0.62]])),
    ("rank 1", np.outer([1.2, 0.8], [0.8, 0.6])),
]


def chart_maps(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(t), np.sin(t)])
    dots = np.linspace(0, 2 * np.pi, 13)[:-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.02, wspace=0.05)
    for ax, (title, m) in zip(axes, MAPS, strict=True):
        u, s, vt = np.linalg.svd(m)
        ax.plot(*circle, color=c["line"], linewidth=1.2, linestyle="--")
        ax.plot(*(m @ circle), color=c["enc"], linewidth=2.2)
        points = np.stack([np.cos(dots), np.sin(dots)])
        ax.plot(*points, "o", color=c["line"], markersize=3.5)
        ax.plot(*(m @ points), "o", color=c["enc"], markersize=4.5)
        for k in range(2):
            ax.annotate("", xy=vt[k], xytext=(0, 0),
                        arrowprops={"arrowstyle": "-|>", "color": c["muted"], "linewidth": 1.1})
            if s[k] > 1e-9:
                ax.annotate("", xy=s[k] * u[:, k], xytext=(0, 0),
                            arrowprops={"arrowstyle": "-|>", "color": c["dec"], "linewidth": 1.8})
        ax.set_xlim(-1.95, 1.95)
        ax.set_ylim(-2.25, 1.75)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="center", fontsize=11, pad=4)
        second = "0" if s[1] < 1e-9 else f"{s[1]:.2f}"
        ax.text(0, -1.98, f"σ₁ = {s[0]:.2f}   σ₂ = {second}", ha="center", va="top", fontsize=10,
                color=c["ink"], fontfamily=MONO)
    _caption(fig, "What a 2×2 matrix does to the unit circle",
             "dashed: the inputs; teal: where they land; grey arrows: the input axes vᵢ; rose: their images σᵢ uᵢ", c)
    return fig


# ---------------------------------------------------------------- a photograph, layer by layer

PHOTO_RANKS = [1, 4, 16, 64]


@cache
def photo():
    from skimage import color, data

    gray = color.rgb2gray(data.astronaut())
    u, s, vt = np.linalg.svd(gray)
    return gray, u, s, vt


def truncated(k):
    gray, u, s, vt = photo()
    return (u[:, :k] * s[:k]) @ vt[:k]


def relative_error(k):
    """Eckart-Young: the error of the best rank-k approximation, from the singular values alone."""
    gray, _, s, _ = photo()
    return np.sqrt(np.sum(s[k:] ** 2)) / np.linalg.norm(gray)


def chart_photo(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    gray, *_ = photo()
    m, n = gray.shape
    fig, axes = plt.subplots(1, 5, figsize=(7.6, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.73, bottom=0.17, wspace=0.06)
    panels = [(k, truncated(k)) for k in PHOTO_RANKS] + [(min(m, n), gray)]
    for ax, (k, image) in zip(axes, panels, strict=True):
        ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        full = k == min(m, n)
        ax.set_title("the photo" if full else f"rank {k}", loc="center", fontsize=11, pad=4)
        numbers = m * n if full else k * (m + n)
        ax.text(0.5, -0.05, f"{numbers:,} numbers", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=c["ink"], fontfamily=MONO)
        if not full:
            ax.text(0.5, -0.17, f"error {relative_error(k):.0%}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=9, color=c["muted"], fontfamily=MONO)
    _caption(fig, f"A {m}×{n} photograph as a sum of rank-one layers",
             "the first k terms of its singular value decomposition: k(m + n) numbers, with σᵢ folded into uᵢ", c)
    return fig


def chart_eckart_young(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    gray, _, s, _ = photo()
    n = len(s)
    ks = np.arange(1, n)
    grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    measured = [np.linalg.norm(gray - truncated(k)) / np.linalg.norm(gray) for k in grid]
    rng = np.random.default_rng(0)
    random = []
    for k in grid:  # the best matrix whose columns lie in k random directions: Q Q^T M
        q, _ = np.linalg.qr(rng.standard_normal((gray.shape[0], k)))
        random.append(np.linalg.norm(gray - q @ (q.T @ gray)) / np.linalg.norm(gray))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.75, bottom=0.16, wspace=0.28)
    left.plot(np.arange(1, n + 1), s, color=c["enc"], linewidth=1.8)
    left.set_yscale("log")
    left.set_xscale("log")
    left.set_xlabel("i")
    left.set_ylabel("σᵢ")
    left.set_title("singular values of the photo", fontsize=10.5, pad=6)
    right.plot(ks, 100 * np.sqrt(np.cumsum(s[::-1] ** 2)[::-1][1:]) / np.linalg.norm(gray), color=c["enc"],
               linewidth=1.8, label="√(σ²ₖ₊₁ + … + σ²ₙ), from σ alone")
    right.plot(grid, 100 * np.array(measured), "o", color=c["dec"], markersize=6,
               label="‖M − Mₖ‖, measured")
    right.plot(grid, 100 * np.array(random), "s", color=c["muted"], markersize=5,
               label="k random directions instead")
    right.set_xscale("log")
    right.set_xlabel("rank k")
    right.set_ylabel("error, % of ‖M‖")
    right.set_ylim(0, 100)
    right.set_title("error of the rank-k approximation", fontsize=10.5, pad=6)
    right.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.52))
    left.set_xticks([1, 10, 100, n], ["1", "10", "100", str(n)])
    right.set_xticks([1, 10, 100, n], ["1", "10", "100", str(n)])
    for ax in (left, right):
        ax.minorticks_off()
    _caption(fig, "The Eckart–Young theorem, checked on the photograph",
             "Frobenius norm; Mₖ keeps the first k terms of the SVD, and no rank-k matrix comes closer", c)
    return fig


# ---------------------------------------------------------------- the first step of LoRA

PROJECTION_RANKS = [1, 4, 16, 64, 1024]
PROJECTION_D = 48


def chart_projection(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(PROJECTION_RANKS), figsize=(7.6, 2.8))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.73, bottom=0.17, wspace=0.08)
    for ax, r in zip(axes, PROJECTION_RANKS, strict=True):
        a0 = rng.standard_normal((r, PROJECTION_D))  # entries of variance 1
        p = a0.T @ a0 / r  # its average over draws is the identity, for every r
        ax.imshow(p, cmap=_signed(c), vmin=-1.2, vmax=1.2, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"r = {r}", loc="center", fontsize=11, pad=4)
        off = np.linalg.norm(p - np.eye(PROJECTION_D)) / np.sqrt(PROJECTION_D)
        ax.text(0.5, -0.05, f"rank {min(r, PROJECTION_D)}", transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=c["ink"], fontfamily=MONO)
        ax.text(0.5, -0.17, f"off by {off:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=c["muted"], fontfamily=MONO)
    _caption(fig, f"A₀ᵀA₀ / (rσ²) for a random A₀ with r rows and {PROJECTION_D} columns",
             "teal +1, rose −1; off by: ‖P − I‖ / ‖I‖. On average over draws it is the identity for every r", c)
    return fig


def lora_graph() -> Figure:
    fig = Figure(
        760,
        324,
        alt="One LoRA layer. The input x goes through the frozen matrix W0 and, in parallel, through A, which "
        "squeezes it to r numbers z = A x, then through B and the scale alpha over r; the two outputs are added "
        "into y. Going back, the gradient g with respect to y gives B the gradient (alpha/r) g z-transposed and A "
        "the gradient (alpha/r) (B-transposed g) x-transposed",
    )
    fig.text(14, 28, "One LoRA layer, forward and back", kind="title")
    fig.text(14, 50, "the frozen path and the low-rank path read the same x; their outputs add up", kind="note")
    top, low = 100, 176
    fig.box(14, 136, 64, 34, role="data", label="x", r=8)
    fig.text(46, 190, "d_in", kind="small", anchor="middle")
    fig.line((78, 153), (100, 153), (100, top), (150, top), width=1.4)
    fig.arrow((100, 153), (100, low), (148, low), width=1.4)
    fig.box(150, top - 17, 300, 34, role="neutral", label="W₀ x    frozen, d_out × d_in", r=8)
    fig.box(150, low - 17, 86, 34, role="dec", label="A", r=8)
    fig.text(193, low + 34, "r × d_in", kind="small", anchor="middle")
    fig.arrow((238, low), (262, low), role="dec")
    fig.box(264, low - 15, 48, 30, role="dec", label="z", r=15, strong=True)
    fig.text(288, low + 34, "r numbers", kind="small", anchor="middle")
    fig.arrow((314, low), (338, low), role="dec")
    fig.box(340, low - 17, 86, 34, role="dec", label="B", r=8)
    fig.text(383, low + 34, "d_out × r", kind="small", anchor="middle")
    fig.arrow((428, low), (452, low), role="dec")
    fig.box(454, low - 17, 66, 34, role="dec", label="× α/r", r=8)
    fig.line((450, top), (574, top), (574, 120), width=1.4)
    fig.line((520, low), (574, low), (574, 156), role="dec", width=1.4)
    fig.box(556, 121, 36, 34, role="neutral", label="+", r=17)
    fig.arrow((592, 138), (622, 138), width=1.4)
    fig.box(624, 121, 70, 34, role="data", label="y", r=8)
    fig.text(659, 190, "d_out", kind="small", anchor="middle")
    # the way back
    fig.line((690, 238), (120, 238), role="dec", width=1.2, dash=True)
    fig.text(694, 232, "back", kind="small", anchor="end")
    fig.text(193, 266, "∂L/∂A = (α/r) (Bᵀg) xᵀ", kind="mono_dec", anchor="middle")
    fig.text(383, 266, "∂L/∂B = (α/r) g zᵀ", kind="mono_dec", anchor="middle")
    fig.text(659, 266, "g = ∂L/∂y", kind="mono_dec", anchor="middle")
    fig.text(14, 306, "B learns from what A lets through (z); A, from what B sends back (Bᵀg)", kind="note")
    return fig


# ---------------------------------------------------------------- charts from the experiment

@cache
def results():
    return json.loads((DATA / "lora.json").read_text(encoding="utf-8"))


def pick(group, **match):
    """Runs of one group; rsLoRA's r = 8 run is LoRA's, since the two scales agree there."""
    if group == "rsLoRA" and match.get("r") == 8:
        group = "LoRA"
    return [r for r in results()["runs"] if r["group"] == group and all(r.get(k) == v for k, v in match.items())]


def accuracy(runs, epoch=-1):
    return np.array([100 * r["accuracy"][epoch] for r in runs])


def ranks():
    return sorted({r["r"] for r in results()["runs"] if r["group"] == "LoRA"})


def _ramp(c, hue, n):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(hue, [c[f"{hue}_tint"], c[hue], c[f"{hue}_ink"]])
    return [cmap(t) for t in np.linspace(0.35, 1.0, n)]


def chart_scale(theme):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c = _style(theme)
    colors = _ramp(c, "enc", len(ranks()))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.7, bottom=0.15, wspace=0.12)
    for ax, (group, title) in zip(axes, [("LoRA", "scale α/r, as in the LoRA paper"),
                                         ("rsLoRA", "scale α′/√r, as in rsLoRA")], strict=True):
        for rank, color in zip(ranks(), colors, strict=True):
            run = pick(group, r=rank, seed=0)[0]
            ax.plot(run["log"]["step"], run["log"]["update"], color=color, linewidth=1.8)
            if rank in (ranks()[0], ranks()[-1]):
                ax.text(run["log"]["step"][-1] * 1.06, run["log"]["update"][-1], str(rank), fontsize=8.5,
                        color=c["ink"], va="center")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 313, 939], ["10", "100", "313", "939"])
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_xlabel("training step (313 per epoch)")
        ax.minorticks_off()
    axes[0].set_ylabel("‖ΔW‖, mean of 24 matrices")
    handles = [Line2D([], [], color=col, linewidth=2.2) for col in colors]
    fig.legend(handles, [f"r = {r}" for r in ranks()], loc="upper right", ncol=len(ranks()), fontsize=9,
               bbox_to_anchor=(0.985, 0.865), handlelength=1.4, columnspacing=1.0)
    _caption(fig, "How big the update grows, rank by rank",
             "LoRA on bert-base, query and value, SST-2; the two scales are equal at r = 8; seed 0", c)
    return fig


def final_loss(run, last=20):
    """Training loss over the last `last` logged points: the last 200 steps."""
    return float(np.mean(run["log"]["loss"][-last:]))


def chart_ranks(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.7))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.72, bottom=0.15, wspace=0.3)
    full = pick("full fine-tuning")
    xs = np.log2(ranks())
    rules = [("LoRA", c["dec"], -0.09, "α/r"), ("rsLoRA", c["att"], 0.09, "α′/√r")]
    for group, color, shift, label in rules:
        for ax, measure in ((left, lambda runs: np.array([final_loss(r) for r in runs])),
                            (right, accuracy)):
            values = [measure(pick(group, r=r)) for r in ranks()]
            for x, v in zip(xs, values, strict=True):
                ax.plot([x + shift] * 2, [v.min(), v.max()], color=color, linewidth=3, alpha=0.4,
                        solid_capstyle="round")
            ax.plot(xs + shift, [v.mean() for v in values], "o-", color=color, markersize=6, linewidth=1.4,
                    label=label)
    f = accuracy(full)
    right.axhspan(f.min(), f.max(), color=c["line"], alpha=0.35, linewidth=0)
    right.axhline(f.mean(), color=c["muted"], linewidth=1.1, linestyle="--")
    right.text(xs[0] - 0.2, f.max() + 0.05, "full fine-tuning", ha="left", va="bottom", fontsize=9,
               color=c["muted"])
    loss = np.array([final_loss(r) for r in full])
    left.axhline(loss.mean(), color=c["muted"], linewidth=1.1, linestyle="--")
    left.text(xs[-1] + 0.2, loss.mean() + 0.003, "full fine-tuning", ha="right", va="bottom", fontsize=9,
              color=c["muted"])
    for ax, title in ((left, "training loss, last 200 steps"), (right, "validation accuracy, %")):
        ax.set_xticks(xs, [str(r) for r in ranks()])
        ax.set_xlabel("rank r")
        ax.set_title(title, fontsize=10.5, pad=6)
    left.legend(loc="lower left", fontsize=9)
    _caption(fig, "What a higher rank buys, under the two scales",
             "bert-base on SST-2, 3 epochs; dot: mean of 3 seeds, bar: range; grey: full fine-tuning", c)
    return fig


ONE_FACTOR = [  # label, group, extra fields
    ("A and B, LoRA", "LoRA", {"r": 8}),
    ("only B; A stays random", "only B", {}),
    ("only A; B random, frozen", "only A", {}),
    ("LoRA+, B's rate 4 × A's", "LoRA+", {"ratio": 4}),
    ("LoRA+, B's rate 16 × A's", "LoRA+", {"ratio": 16}),
]


def chart_factors(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.73, bottom=0.15, wspace=0.75)
    for run in pick("LoRA", r=8):
        left.plot(run["log"]["step"], [100 * d for d in run["log"]["drift"]], color=c["dec"], linewidth=1.4,
                  alpha=0.8)
    left.set_xlabel("training step")
    left.set_ylabel("‖A − A₀‖ / ‖A₀‖, %")
    left.set_title("how far A moves (r = 8, 3 seeds)", fontsize=10.5, pad=6)
    full = accuracy(pick("full fine-tuning"))
    right.axvspan(full.min(), full.max(), color=c["line"], alpha=0.35, linewidth=0)
    right.axvline(full.mean(), color=c["muted"], linewidth=1.1, linestyle="--")
    for k, (_, group, extra) in enumerate(ONE_FACTOR):
        values = accuracy(pick(group, **extra))
        color = c["dec"] if k < 3 else c["att"]
        right.plot([values.min(), values.max()], [k, k], color=color, linewidth=3, alpha=0.45,
                   solid_capstyle="round")
        right.plot([values.mean()], [k], "o", color=color, markersize=7)
    right.set_yticks(range(len(ONE_FACTOR)), [label for label, *_ in ONE_FACTOR], fontsize=9.5)
    right.invert_yaxis()
    right.grid(axis="x")
    right.grid(axis="y", visible=False)
    right.set_xlabel("validation accuracy, %")
    right.set_title("what each factor is worth (r = 8)", fontsize=10.5, pad=6)
    _caption(fig, "Both factors learn",
             "left: A's distance from its random start; right: mean of 3 seeds and range; grey: full fine-tuning", c)
    return fig


def _similarity(key, name):
    return np.array(results()["analysis"]["similarity"][key][name])


def chart_subspace(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    keys = list(results()["analysis"]["similarity"])
    fig = plt.figure(figsize=(7.6, 3.7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], left=0.07, right=0.98, top=0.74, bottom=0.14,
                            wspace=0.3, hspace=0.55)
    for row, key in enumerate(SUBSPACE_EXAMPLES):
        ax = fig.add_subplot(grid[row, 0])
        phi = _similarity(key, "A_r8_r64")
        ax.imshow(phi, cmap=_sequential(c), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 7], ["1", "8"])
        ax.set_xticks([0, 15, 31, 63], ["1", "16", "32", "64"])
        ax.grid(False)
        layer, name = key.split(".")
        ax.set_title(f"layer {int(layer) + 1}, {name}: i of r = 8 against j of r = 64", fontsize=9.5, pad=4)
        if row == 1:
            ax.set_xlabel("j")
        ax.set_ylabel("i", rotation=0, labelpad=8)
    ax = fig.add_subplot(grid[:, 1])
    curves = {
        "rank 8 vs rank 64": ([np.diag(_similarity(k, "A_r8_r64")) for k in keys], c["dec"]),
        "two seeds, rank 64": ([_similarity(k, "A_seeds_r64") for k in keys], c["enc"]),
    }
    for label, (values, color) in curves.items():
        values = np.array(values)
        i = np.arange(1, values.shape[1] + 1)
        ax.fill_between(i, np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0), color=color,
                        alpha=0.18, linewidth=0)
        ax.plot(i, np.median(values, axis=0), color=color, linewidth=2, label=label)
    i = np.arange(1, 65)
    ax.plot(i, i / D, color=c["muted"], linewidth=1.4, linestyle="--", label="random directions, i / 768")
    ax.set_xscale("log")
    ax.set_xticks([1, 8, 64], ["1", "8", "64"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("i, the top directions compared")
    ax.set_ylabel("φ(i, i)")
    ax.set_title("all 24 matrices: median, quartiles", fontsize=9.5, pad=4)
    ax.legend(loc="upper right", fontsize=8.5)
    _caption(fig, "Do two LoRA runs find the same directions?",
             "Hu et al.'s similarity φ between the input directions of A, from 0, perpendicular, to 1 (violet), "
             "the same subspace", c)
    return fig


SUBSPACE_EXAMPLES = ["5.query", "5.value"]


def _sequential(c):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", [c["card"], c["att_tint"], c["att"]])


def chart_amplification(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    amp = results()["analysis"]["amplification"]
    keys = list(amp)
    sources = [("LoRA r = 8", c["dec"]), ("LoRA r = 64", c["att"]), ("full, top 8", c["enc"])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.72, bottom=0.2, wspace=0.32)
    for k, (source, color) in enumerate(sources):
        ratio = [amp[key][source]["W_in_update"] / amp[key][source]["W_random"] for key in keys]
        factor = [amp[key][source]["update"] / amp[key][source]["W_in_update"] for key in keys]
        jitter = np.linspace(-0.18, 0.18, len(keys))
        left.plot(k + jitter, ratio, "o", color=color, markersize=4, alpha=0.8)
        left.plot([k - 0.25, k + 0.25], [np.median(ratio)] * 2, color=color, linewidth=2.2)
        right.plot(k + jitter, factor, "o", color=color, markersize=4, alpha=0.8)
        right.plot([k - 0.25, k + 0.25], [np.median(factor)] * 2, color=color, linewidth=2.2)
    top = [amp[key]["LoRA r = 8"]["W_top"] / amp[key]["LoRA r = 8"]["W_random"] for key in keys]
    left.axhline(np.median(top), color=c["muted"], linewidth=1, linestyle="--")
    left.text(2.45, np.median(top), "W's own top 8", ha="right", va="bottom", fontsize=8.5, color=c["muted"])
    left.axhline(1, color=c["muted"], linewidth=1, linestyle=":")
    for ax in (left, right):
        ax.set_yscale("log")
        ax.set_xticks(range(len(sources)), [s.replace(", ", "\n") for s, _ in sources], fontsize=9)
        ax.minorticks_off()
        ax.set_xlim(-0.5, 2.5)
    left.set_title("‖UᵀWV‖ against random directions", fontsize=10, pad=6)
    right.set_title("amplification ‖ΔW‖ / ‖UᵀWV‖", fontsize=10, pad=6)
    _caption(fig, "Which directions of W the update works on",
             "U, V: the update's top singular directions; one dot per query or value matrix, line: median", c)
    return fig


def chart_spectra(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    spectra = results()["analysis"]["spectra"][SPECTRUM_EXAMPLE]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.15)
    floor = 1e-5
    for name, color, label in [("W", c["muted"], "W, pretrained"), ("full", c["enc"], "ΔW, full fine-tuning"),
                               ("LoRA r = 64", c["att"], "ΔW, LoRA r = 64"),
                               ("LoRA r = 8", c["dec"], "ΔW, LoRA r = 8")]:
        values = np.maximum(spectra[name], floor)
        ax.plot(np.arange(1, len(values) + 1), values, color=color, linewidth=1.9, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, None)
    ax.set_xticks([1, 8, 64, D], ["1", "8", "64", str(D)])
    ax.minorticks_off()
    ax.set_xlabel("i")
    ax.set_ylabel("σᵢ")
    ax.legend(loc="lower left", fontsize=9)
    layer, name = SPECTRUM_EXAMPLE.split(".")
    _caption(fig, "What a rank limit looks like",
             f"singular values of the {name} matrix of layer {int(layer) + 1} and of three updates to it; "
             "zeros are drawn at 10⁻⁵", c)
    return fig


SPECTRUM_EXAMPLE = "5.value"


# ---------------------------------------------------------------- after training

def _update(fig, x, y, side, role, thin=7):
    """A low-rank update drawn as its factors: a column B, a row A and their product."""
    fig.rect(x, y + thin + 4, thin, side, role=role, r=2)
    fig.rect(x + thin + 4, y, side, thin, role=role, r=2)
    fig.rect(x + thin + 4, y + thin + 4, side, side, role=role, tint=True, r=3)
    return x + thin + 4 + side


def merge() -> Figure:
    fig = Figure(
        760,
        312,
        alt="Two diagrams. Merge: the frozen matrix W0 plus the trained low-rank update gamma B A equals one matrix "
        "W of the same shape, so the merged model costs nothing extra. Add: two updates B1 A1 and B2 A2 sum to one "
        "product of B1 and B2 side by side with A1 and A2 stacked, an update of rank at most r1 plus r2",
    )
    fig.text(14, 28, "Merging an update, and adding two", kind="title")
    fig.text(14, 50, "both are exact matrix algebra; whether two added updates still work together is another "
             "question", kind="note")
    side, y = 84, 108
    fig.text(20, 88, "merge", kind="subtitle")
    fig.rect(20, y + 11, side, side, role="neutral", r=3)
    fig.text(20 + side / 2, y + 11 + side / 2 + 6, "W₀", kind="lead", anchor="middle")
    fig.text(116, y + 11 + side / 2 + 6, "+", kind="lead")
    right = _update(fig, 134, y, side, "dec")
    fig.text(right - side / 2, y + 11 + side / 2 + 5, "γBA", kind="block", role="dec", anchor="middle")
    fig.text(right + 10, y + 11 + side / 2 + 6, "=", kind="lead")
    fig.rect(right + 30, y + 11, side, side, role="neutral", r=3)
    fig.rect(right + 30, y + 11, side, side, role="dec", tint=True, r=3)
    fig.text(right + 30 + side / 2, y + 11 + side / 2 + 5, "W", kind="block", role="dec", anchor="middle")
    fig.text(62, y + side + 34, "frozen", kind="small", anchor="middle")
    fig.text(right - side / 2, y + side + 34, "trained", kind="small", anchor="middle")
    fig.text(right + 30 + side / 2, y + side + 34, "one matrix", kind="small", anchor="middle")
    fig.text(20, y + side + 62, "no extra layer, no extra time", kind="note")

    x0 = 410
    fig.text(x0, 88, "add", kind="subtitle")
    thin = 7
    fig.rect(x0, y + 2 * thin + 8, thin, side, role="dec", r=2)  # B1
    fig.rect(x0 + thin + 3, y + 2 * thin + 8, thin, side, role="att", r=2)  # B2, beside it
    fig.rect(x0 + 2 * thin + 8, y, side, thin, role="dec", r=2)  # A1
    fig.rect(x0 + 2 * thin + 8, y + thin + 3, side, thin, role="att", r=2)  # A2, under it
    fig.rect(x0 + 2 * thin + 8, y + 2 * thin + 8, side, side, role="att", tint=True, r=3)
    left_end = x0 + 2 * thin + 8 + side
    fig.text(left_end + 12, y + 2 * thin + 8 + side / 2 + 6, "=", kind="lead")
    a = _update(fig, left_end + 34, y + thin + 4, side - 22, "dec")
    fig.text(a + 8, y + 2 * thin + 8 + side / 2 + 6, "+", kind="lead")
    b = _update(fig, a + 26, y + thin + 4, side - 22, "att")
    fig.text(x0 + 2 * thin + 8 + side / 2, y + side + 46, "[B₁ B₂] [A₁; A₂]", kind="small", anchor="middle")
    fig.text(a - (side - 22) / 2, y + side + 46, "γB₁A₁", kind="small", anchor="middle")
    fig.text(b - (side - 22) / 2, y + side + 46, "γB₂A₂", kind="small", anchor="middle")
    fig.text(x0, y + side + 74, "one update of rank at most r₁ + r₂", kind="note")
    fig.text(14, 298, "Kept apart, the factors are a small file per task that can be swapped in and out of one "
             "base model.", kind="note")
    return fig


# ---------------------------------------------------------------- cover

def _rank_one_svg(x, y, size, u, v, weight):
    """A rank-one matrix u v^T as a grid of cells, red for positive, grey for negative."""
    n = len(u)
    cell = size / n
    out = []
    for i in range(n):
        for j in range(n):
            value = weight * u[i] * v[j]
            fill = cv.RED if value > 0 else cv.CHIP_INK
            opacity = min(abs(value), 1)
            out.append(f'<rect x="{x + j * cell:.1f}" y="{y + i * cell:.1f}" width="{cell - 1:.1f}" '
                       f'height="{cell - 1:.1f}" fill="{fill}" fill-opacity="{opacity:.2f}"/>')
    out.append(f'<rect x="{x - 3:.1f}" y="{y - 3:.1f}" width="{size + 5:.1f}" height="{size + 5:.1f}" rx="4" '
               f'fill="none" stroke="{cv.LINE}" stroke-width="1.4"/>')
    return "".join(out)


def _formula(x, y, parts, size=26):
    """One line of SVG text with superscripts and subscripts stacked on the same letter."""
    small = round(size * 0.62)
    spans, level = [], 0
    for text, kind in parts:
        if kind == "sup":
            spans.append(f'<tspan dy="{-0.42 * size - level:.1f}" font-size="{small}">{text}</tspan>')
            level = -0.42 * size
        elif kind == "sub":  # back under the superscript just written
            back = 0.56 * small
            spans.append(f'<tspan dx="{-back:.1f}" dy="{0.72 * size:.1f}" font-size="{small}">{text}</tspan>')
            level = 0.30 * size
        else:
            spans.append(f'<tspan dy="{-level:.1f}">{text}</tspan>')
            level = 0
    return (f'<text x="{x}" y="{y}" font-family="{cv.SANS}" font-size="{size}" font-weight="500" '
            f'fill="{cv.RED}" xml:space="preserve">{"".join(spans)}</text>')


def cover():
    n, size, gap = 10, 96, 36
    t = np.linspace(0, 1, n)
    layers = [  # three rank-one layers with falling weights, like the terms of an SVD
        (np.cos(np.pi * t), np.sin(np.pi * (t + 0.2)), 1.0),
        (np.cos(2 * np.pi * t), np.cos(2 * np.pi * (t - 0.15)), 0.62),
        (np.sin(3 * np.pi * t), np.cos(3 * np.pi * t), 0.38),
    ]
    total = sum(w * np.outer(u, v) for u, v, w in layers)
    x0, y0 = cv.MOTIF_X0 + 4, 162
    out = [cv.label(cv.MOTIF_X0, 132, "AN UPDATE OF RANK 3, TERM BY TERM")]
    cells = []
    x = x0
    peak = np.abs(total).max()
    for k in range(n):
        for j in range(n):
            value = total[k, j] / peak
            fill = cv.RED if value > 0 else cv.CHIP_INK
            cells.append(f'<rect x="{x + j * size / n:.1f}" y="{y0 + k * size / n:.1f}" '
                         f'width="{size / n - 1:.1f}" height="{size / n - 1:.1f}" fill="{fill}" '
                         f'fill-opacity="{min(abs(value), 1):.2f}"/>')
    out.append("".join(cells))
    out.append(f'<rect x="{x - 3:.1f}" y="{y0 - 3:.1f}" width="{size + 5:.1f}" height="{size + 5:.1f}" rx="4" '
               f'fill="none" stroke="{cv.LINE}" stroke-width="1.4"/>')
    out.append(cv.text(x + size / 2, y0 + size + 34, "ΔW", 21, cv.INK, cv.SANS, 600, "middle"))
    for k, (u, v, w) in enumerate(layers):
        sign = "=" if k == 0 else "+"
        x += size + gap
        out.append(cv.text(x - gap / 2, y0 + size / 2 + 10, sign, 28, cv.MUTED, cv.SANS, 400, "middle"))
        out.append(_rank_one_svg(x, y0, size, u, v, w / peak * 1.6))
        out.append(cv.text(x + size / 2, y0 + size + 34, f"σ{'₁₂₃'[k]} u{'₁₂₃'[k]} v{'₁₂₃'[k]}ᵀ", 21, cv.MUTED,
                           cv.SANS, 400, "middle"))
    cv.fits(x + size)
    out.append(cv.label(cv.MOTIF_X0, 392, "THE BEST RANK-r APPROXIMATION MISSES"))
    w = 282
    out.append(f'<rect x="{cv.MOTIF_X0}" y="412" width="{w}" height="54" rx="8" fill="{cv.RED_TINT}" '
               f'stroke="{cv.RED}" stroke-width="1.6"/>')
    out.append(_formula(cv.MOTIF_X0 + 24, 448, [("√( σ", ""), ("2", "sup"), ("r+1", "sub"), ("  + σ", ""),
                                                ("2", "sup"), ("r+2", "sub"), ("  + …  )", "")]))
    cv.fits(cv.MOTIF_X0 + w)
    out.append(cv.text(cv.MOTIF_X0, 500, "Eckart and Young, Psychometrika, 1936", 15, cv.FAINT, cv.MONO))
    return cv.frame("blog / machine learning", ["The algebra", "of LoRA"],
                    ["Why a weight update can be", "the product of two thin", "matrices, and when it can't"],
                    "".join(out), footer="LoRA, derived and measured")


COVER = cover

CHARTS = {
    "maps": chart_maps,
    "photo": chart_photo,
    "eckart-young": chart_eckart_young,
    "projection": chart_projection,
    "scale": chart_scale,
    "ranks": chart_ranks,
    "factors": chart_factors,
    "subspace": chart_subspace,
    "amplification": chart_amplification,
    "spectra": chart_spectra,
}

FIGURES = {
    "lora-graph": lora_graph,
    "merge": merge,
}


def _alt_numbers():
    """The numbers the alt texts of the experiment's charts quote."""
    out = {}
    first = {g: [pick(g, r=r, seed=0)[0]["log"]["update"][0] for r in ranks()] for g in ("LoRA", "rsLoRA")}
    out["first"] = first
    out["loss"] = {g: [np.mean([final_loss(x) for x in pick(g, r=r)]) for r in ranks()] for g in ("LoRA", "rsLoRA")}
    out["acc"] = {g: [accuracy(pick(g, r=r)).mean() for r in ranks()] for g in ("LoRA", "rsLoRA")}
    out["full"] = accuracy(pick("full fine-tuning")).mean()
    out["drift"] = np.mean([run["log"]["drift"][-1] for run in pick("LoRA", r=8)])
    out["factors"] = {label: accuracy(pick(group, **extra)).mean() for label, group, extra in ONE_FACTOR}
    keys = list(results()["analysis"]["similarity"])
    out["phi_r8"] = np.median([_similarity(k, "A_r8_r64")[0, 0] for k in keys])
    out["phi_seeds"] = np.median([_similarity(k, "A_seeds_r64")[0] for k in keys])
    amp = results()["analysis"]["amplification"]
    out["amp"] = {src: np.median([amp[k][src]["update"] / amp[k][src]["W_in_update"] for k in keys])
                  for src in ("LoRA r = 8", "LoRA r = 64", "full, top 8")}
    return out


def chart_alts():
    gray, _, s, _ = photo()
    n = _alt_numbers()
    rk = ranks()
    return {
        "scale": f"Two panels of the size of the LoRA update during training, for ranks {rk[0]} to {rk[-1]}. Under "
        f"the scale alpha over r the curves start far apart, from {n['first']['LoRA'][0]:.3f} at rank 1 down to "
        f"{n['first']['LoRA'][-1]:.3f} at rank 256, and the low ranks stay ahead. Under alpha over square root of "
        f"r all ranks start together, between {min(n['first']['rsLoRA']):.3f} and {max(n['first']['rsLoRA']):.3f}, "
        "and the high ranks end ahead",
        "ranks": "Training loss and validation accuracy against rank under the two scales. Under alpha over r the "
        f"final training loss is flat, about {n['loss']['LoRA'][-1]:.2f}, and accuracy stays between "
        f"{min(n['acc']['LoRA']):.1f}% and {max(n['acc']['LoRA']):.1f}%. Under alpha over square root of r the "
        f"loss falls from {n['loss']['rsLoRA'][0]:.2f} to {n['loss']['rsLoRA'][-1]:.2f} and accuracy rises to "
        f"{max(n['acc']['rsLoRA']):.1f}%, level with full fine-tuning at {n['full']:.1f}%",
        "factors": f"Left: the distance of A from its random start grows during training to about "
        f"{100 * n['drift']:.0f}% of its initial size. Right: validation accuracy at rank 8: "
        + "; ".join(f"{label} {v:.1f}%" for label, v in n["factors"].items()),
        "subspace": "Heatmaps of Hu et al.'s subspace similarity between the input directions of A in a rank-8 and "
        "a rank-64 run, for the query and value matrices of layer 6: the top direction of the rank-8 run is largely "
        "contained in the rank-64 run, the others much less. Right: the similarity of the top i directions, median "
        f"over 24 matrices: {n['phi_r8']:.2f} for the top direction between ranks 8 and 64, "
        f"{n['phi_seeds']:.2f} between two seeds at rank 64, both falling towards the random baseline i over 768",
        "amplification": "Two dot plots over the 24 adapted matrices. Left: how strongly W acts in the update's top "
        "directions compared with random directions: about 1.2 times for LoRA, 2.3 times for full fine-tuning, "
        "while W's own top directions carry about 28 times more. Right: the amplification factor, the size of the "
        f"update over the size of W in its directions: median {n['amp']['LoRA r = 8']:.1f} for LoRA rank 8, "
        f"{n['amp']['LoRA r = 64']:.1f} for rank 64, {n['amp']['full, top 8']:.1f} for full fine-tuning",
        "spectra": "Singular values on log axes for one value matrix of BERT and three updates to it. The pretrained "
        "matrix has a slowly falling spectrum. The rank-8 LoRA update has eight singular values and then drops to "
        "zero; the rank-64 update drops after 64; the full fine-tuning update falls gradually with no cliff",
        "maps": "Three 2×2 matrices applied to the unit circle. A full-rank matrix turns it into an ellipse whose "
        "semi-axes are the singular values; a nearly singular one squashes it into a thin ellipse; a rank-one "
        "matrix flattens it into a segment, so that pairs of points land on the same spot",
        "photo": "A grayscale photograph of the astronaut Eileen Collins reconstructed from the first 1, 4, 16 and 64 "
        "terms of its singular value decomposition, next to the original. Rank 1 is a blurred cross of stripes, "
        f"rank 16 is recognisable, rank 64 is close to the original; errors {relative_error(1):.0%}, "
        f"{relative_error(4):.0%}, {relative_error(16):.0%} and {relative_error(64):.0%}",
        "eckart-young": "Left: the singular values of the photograph on log axes, falling from "
        f"{s[0]:.0f} to {s[-1]:.4f}. Right: the error of the rank-k approximation. The measured error of the "
        "truncated SVD lies exactly on the curve computed from the remaining singular values, and keeping k "
        "random directions instead is far worse",
        "projection": "Heatmaps of A0-transposed A0 divided by r sigma squared for a random A0 with 48 columns and "
        "1, 4, 16, 64 and 1024 rows. With one row it is a rank-one pattern of stripes; as r grows the noise fades "
        "and the identity matrix, a bright diagonal, emerges",
    }
