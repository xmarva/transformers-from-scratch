"""Figures, charts and the cover of the post "Adapters and PEFT" on xmarva.github.io.

The block diagrams draw one encoder layer of BERT the same way for every method and add, in colour, what the method
adds; parameter counts are computed from bert-base-uncased's config. The charts read data/adapters.json, the results
of experiment.py on a Kaggle T4.

    uv run python tools/render.py --article adapters --out <site>/assets/posts/adapters
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Box, Figure

DATA = Path(__file__).parent / "data"
D, D_FF, LAYERS, HEADS = 768, 3072, 12, 12  # bert-base-uncased


# ---------------------------------------------------------------- one BERT layer, drawn the same way every time

Y = 176  # the row of the main path; everything below is placed relative to it
H = 34
BELOW = Y + 64  # the row for what a method adds under the layer
LAYER_H = 296


def _base(fig, lora=()):
    """The frozen layer: x, three projections, attention, output projection, add & norm, feed-forward, add & norm.
    Projections named in `lora` are drawn with their low-rank path B A beside them."""
    fig.box(14, Y - H / 2, 34, H, role="data", label="x", r=8)
    for k, name in enumerate(("W_q", "W_k", "W_v")):
        y = Y - 41 + k * 38
        if name in lora:
            fig.els.append(Box(78, y - 15, 28, 30, "neutral", name, mono=True, r=6, size=11))
            fig.els.append(Box(108, y - 15, 24, 30, "dec", "BA", mono=True, strong=True, r=6, size=11))
        else:
            fig.box(78, y - 15, 54, 30, role="neutral", label=name, r=7)
        fig.line((48, Y), (64, Y), (64, y), (76, y), width=1.2)
        fig.arrow((134, y), (150, y), width=1.2)
    fig.box(152, Y - 56, 90, 106, role="att", label="attention", r=9)
    fig.arrow((244, Y), (262, Y))
    fig.box(264, Y - H / 2, 50, H, role="neutral", label="W_o", r=7)
    fig.box(372, Y - H / 2, 56, H, role="neutral", label="+ norm", r=17)
    fig.box(452, Y - H / 2, 50, H, role="neutral", label="W_1", r=7)
    fig.box(518, Y - H / 2, 54, H, role="neutral", label="GELU", r=7)
    fig.box(588, Y - H / 2, 50, H, role="neutral", label="W_2", r=7)
    fig.box(674, Y - H / 2, 56, H, role="neutral", label="+ norm", r=17)
    fig.arrow((430, Y), (450, Y))
    fig.arrow((504, Y), (516, Y))
    fig.arrow((574, Y), (586, Y))
    fig.line((31, Y - H / 2), (31, Y - 86), (400, Y - 86), (400, Y - H / 2 - 2), width=1.1, dash=True)  # residuals
    fig.line((440, Y), (440, Y - 78), (702, Y - 78), (702, Y - H / 2 - 2), width=1.1, dash=True)
    fig.text(214, Y - 92, "residual", kind="small", anchor="middle")
    fig.text(572, Y - 84, "residual", kind="small", anchor="middle")


def _gap_arrows(fig, after_o=True, after_2=True):
    if after_o:
        fig.arrow((316, Y), (370, Y))
    if after_2:
        fig.arrow((640, Y), (672, Y))


def _layer_figure(title, note, alt, height=LAYER_H):
    fig = Figure(760, height, alt=alt)
    fig.text(14, 28, title, kind="title")
    fig.text(14, 50, note, kind="note")
    return fig


def _count(n):
    return f"{n:,}"


def _footnote(fig, text):
    fig.text(14, LAYER_H - 14, text, kind="note")


ADAPTER = 2 * D * 64 + 64 + D  # one bottleneck adapter of size 64 in bert-base


def map_sequential() -> Figure:
    fig = _layer_figure(
        "Sequential adapters: two per layer",
        f"Houlsby et al. 2019. Each adapter: {D} → 64 → {D}, {_count(ADAPTER)} parameters; "
        f"{_count(2 * LAYERS * ADAPTER)} in bert-base",
        f"One BERT layer with two adapters: one after the attention output projection W_o and one after the second "
        f"feed-forward matrix W_2, both before the residual addition and layer norm. {_count(2 * LAYERS * ADAPTER)} "
        f"parameters in all",
    )
    _base(fig)
    fig.arrow((316, Y), (322, Y))
    fig.els.append(Box(324, Y - 13, 44, 26, "dec", "A", mono=True, strong=True, r=6, size=12))
    fig.arrow((640, Y), (644, Y))
    fig.els.append(Box(646, Y - 13, 26, 26, "dec", "A", mono=True, strong=True, r=6, size=12))
    fig.text(346, Y + 40, "adapter", kind="label", role="dec", anchor="middle")
    fig.text(659, Y + 40, "adapter", kind="label", role="dec", anchor="middle")
    _footnote(fig, "grey is frozen; the adapters start as the identity and learn a correction")
    return fig


def map_parallel() -> Figure:
    fig = _layer_figure(
        "A parallel adapter beside the feed-forward block",
        f"He et al. 2022: reads the block's input, adds s · adapter(x) to its output; {_count(LAYERS * ADAPTER)} "
        "parameters",
        "One BERT layer with a parallel adapter: it takes the same input as the feed-forward block, and its output, "
        "scaled, is added to the block's output before the residual addition and layer norm",
    )
    _base(fig)
    _gap_arrows(fig)
    fig.els.append(Box(452, BELOW - 15, 186, 30, "dec", "adapter, scaled by s", strong=True, r=7, size=12))
    fig.line((440, Y + 2), (440, BELOW), (450, BELOW), role="dec", width=1.6)
    fig.line((640, BELOW), (656, BELOW), (656, Y + 2), role="dec", width=1.6)
    _footnote(fig, "the two paths add up: y = FFN(x) + s · W_up GELU(W_down x)")
    return fig


def map_lora() -> Figure:
    r = 8
    per = 2 * r * D
    fig = _layer_figure(
        "LoRA on the query and value projections",
        f"Hu et al. 2021: W x + (α / r) B A x with rank r = {r}; {_count(per)} parameters per matrix, "
        f"{_count(2 * LAYERS * per)} in all",
        f"One BERT layer where the query and value projections each get a low-rank path B A beside the frozen "
        f"matrix, {_count(per)} parameters each against {_count(D * D)} in the matrix",
    )
    _base(fig, lora=("W_q", "W_v"))
    _gap_arrows(fig)
    fig.text(14, BELOW + 4, "BA beside W_q and W_v: the same input goes through both, and the outputs are added",
             kind="small")
    _footnote(fig, f"the frozen {D}×{D} matrices are untouched; only the {_count(per)} numbers of each B and A train")
    return fig


def map_ia3() -> Figure:
    per = 2 * D + D_FF
    fig = _layer_figure(
        "(IA)³: three vectors that rescale",
        f"Liu et al. 2022: keys, values and feed-forward activations, scaled element-wise; {_count(per)} per layer",
        f"One BERT layer where the outputs of W_k and W_v and the activations after GELU are multiplied element by "
        f"element by learned vectors l_k, l_v and l_ff, which start at ones. {_count(LAYERS * per)} parameters",
    )
    _base(fig)
    _gap_arrows(fig)
    for x, y, name in ((142, Y - 3, "k"), (142, Y + 35, "v"), (580, Y, "ff")):  # on the arrows they rescale
        fig.els.append(Box(x - 10, y - 10, 20, 20, "dec", name, mono=True, strong=True, r=10, size=10))
    fig.text(14, BELOW + 4, "⊙ l_k, ⊙ l_v: every key and value, all heads; ⊙ l_ff: the 3,072 hidden units after "
             "GELU", kind="small")
    _footnote(fig, "a vector of ones changes nothing; training moves each entry up (amplify) or down (inhibit)")
    return fig


def map_prefix() -> Figure:
    length = 20
    per = 2 * length * D
    fig = _layer_figure(
        "Prefix tuning: extra keys and values in every layer",
        f"Li & Liang 2021: {length} learned key and value vectors per layer; {_count(per)} per layer, "
        f"{_count(LAYERS * per)} in all",
        f"One BERT layer whose attention also sees {length} extra keys and values that are free parameters of this "
        f"layer, as if {length} invisible tokens preceded the input",
    )
    _base(fig)
    _gap_arrows(fig)
    fig.els.append(Box(152, BELOW - 13, 90, 26, "dec", "P_k, P_v", mono=True, strong=True, r=6, size=12))
    fig.arrow((197, BELOW - 15), (197, Y + 52), role="dec")
    fig.text(252, BELOW + 4, "joined in front of the real keys and values", kind="small")
    _footnote(fig, "queries still come only from real tokens; the prefix is something to attend to, not a token")
    return fig


def map_prompt() -> Figure:
    length = 20
    fig = _layer_figure(
        "Prompt tuning: learned embeddings at the input",
        f"Lester et al. 2021: {length} vectors of size {D} placed among the input embeddings; {_count(length * D)} "
        "parameters",
        f"The input of the first BERT layer with {length} learned embeddings inserted after [CLS]; every layer above "
        "processes them like the embeddings of real tokens",
        height=200,
    )
    x = 14
    for token, hot in (("[CLS]", False), ("p_1", True), ("p_2", True), ("…", True), ("p_20", True), ("the", False),
                       ("film", False), ("is", False), ("tender", False), ("[SEP]", False)):
        x += fig.chip(x, 80, token, role="dec" if hot else "data", strong=hot, h=30) + 6
    fig.text(14, 144, "what layer 1 receives: the p_i are parameters, the other embeddings are looked up", kind="note")
    fig.text(14, 170, "nothing inside the layers changes; all 12 attend to the p_i as to any token", kind="note")
    return fig


def map_bitfit() -> Figure:
    biases = LAYERS * (3 * D + D + D + D_FF + D + D)  # q k v, attention output, its norm, W_1, W_2, the second norm
    fig = _layer_figure(
        "BitFit: only the biases",
        f"Ben Zaken et al. 2022: the + b of every linear layer and layer norm; {_count(biases)} in the encoder",
        f"One BERT layer in which only the bias vectors of the linear layers and layer norms are trained: "
        f"{_count(biases)} parameters in the encoder of bert-base",
    )
    _base(fig)
    _gap_arrows(fig)
    for x, w in ((78, 54), (264, 50), (372, 56), (452, 50), (588, 50), (674, 56)):
        fig.chip(x + w / 2 - 14, BELOW - 13, "+b", role="dec", strong=True, h=24, size=11)
    _footnote(fig, "a bias shifts a layer's output by the same vector for every token")
    return fig


# ---------------------------------------------------------------- inside a bottleneck adapter

def bottleneck() -> Figure:
    b = 64
    fig = Figure(
        760,
        250,
        alt=f"A bottleneck adapter: the hidden state h of size {D} is projected down to {b}, passed through GELU, "
        f"projected back up to {D} by a matrix that starts at zero, and added to h",
    )
    fig.text(14, 28, "Inside an adapter", kind="title")
    fig.text(14, 50, f"down to {b} numbers and back up: the bottleneck limits what the adapter can change", kind="note")
    fig.box(14, 104, 70, 40, role="data", label=f"h  ({D})", r=8)
    fig.line((84, 124), (110, 124), width=1.4)
    fig.box(112, 92, 120, 64, role="dec", label=f"W_down {b}×{D}", r=8)
    fig.arrow((234, 124), (262, 124), role="dec")
    fig.box(264, 108, 70, 32, role="dec", label="GELU", r=8)
    fig.arrow((336, 124), (364, 124), role="dec")
    fig.box(366, 92, 120, 64, role="dec", label=f"W_up {D}×{b}", r=8)
    fig.arrow((488, 124), (548, 124), role="dec")
    fig.box(550, 106, 44, 36, role="neutral", label="+", r=18)
    fig.arrow((596, 124), (632, 124))
    fig.box(634, 104, 112, 40, role="data", label="h + Δh", r=8)
    fig.line((49, 104), (49, 76), (572, 76), (572, 104), width=1.2, dash=True)
    fig.text(300, 70, "skip: h itself", kind="small", anchor="middle")
    fig.text(426, 180, "initialized to zero:", kind="small", anchor="middle")
    fig.text(426, 196, "the adapter starts as the identity", kind="small", anchor="middle")
    fig.text(14, 236, f"{2 * D * b + b + D:,} parameters: two matrices and two bias vectors", kind="count_hot")
    return fig


# ---------------------------------------------------------------- LoRA, to scale

def lora_shapes() -> Figure:
    r = 8
    px = 0.2  # pixels per row or column
    side = D * px
    thin = r * px * 4  # the rank-r strips drawn four times wider, or they would vanish
    fig = Figure(
        760,
        290,
        alt=f"LoRA drawn to scale for one {D}×{D} projection of bert-base. The frozen W has {D * D:,} entries. The "
        f"update ΔW = B A is just as large, but it is the product of a {D}×{r} matrix B and an {r}×{D} matrix A: "
        f"{2 * D * r:,} trained numbers",
    )
    fig.text(14, 28, "A full matrix and a low-rank update, to scale", kind="title")
    fig.text(14, 50, f"one {D}×{D} projection of bert-base; the strips are drawn four times wider than {r}",
             kind="note")
    top = 76
    fig.rect(14, top, side, side, role="neutral", r=3)
    fig.text(14 + side / 2, top + side / 2 + 5, "W", kind="block", role="neutral", anchor="middle")
    fig.text(14 + side / 2, top + side + 20, f"frozen: {D * D:,} numbers", kind="small", anchor="middle")
    fig.text(14 + side + 26, top + side / 2 + 7, "+", kind="lead")
    x0 = 14 + side + 56
    fig.rect(x0, top + thin + 6, thin, side, role="dec", r=2)  # B, a column
    fig.rect(x0 + thin + 6, top, side, thin, role="dec", r=2)  # A, a row
    fig.rect(x0 + thin + 6, top + thin + 6, side, side, role="dec", tint=True, r=3)  # their product, ΔW
    fig.text(x0 + thin + 6 + side / 2, top + thin + 6 + side / 2, "ΔW = B A", kind="block", role="dec",
             anchor="middle")
    fig.text(x0 + thin + 6 + side / 2, top + thin + 6 + side / 2 + 20, f"rank {r}", kind="small", anchor="middle")
    fig.text(x0 - 4, top + thin + 6 + side + 20, f"B: {D}×{r}", kind="small")
    fig.text(x0 + thin + 6 + side + 8, top + thin + 2, f"A: {r}×{D}", kind="small")
    fig.text(x0 + thin + 6 + side + 8, top + side / 2 + 20, f"trained: {2 * D * r:,}", kind="count_hot")
    fig.text(x0 + thin + 6 + side + 8, top + side / 2 + 40, f"{2 * D * r / (D * D):.1%} of W", kind="small")
    return fig


# ---------------------------------------------------------------- charts from the experiment

@cache
def results():
    return json.loads((DATA / "adapters.json").read_text(encoding="utf-8"))


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c, top=0.975):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, top - 0.075, note, ha="left", va="top", fontsize=10, color=c["muted"])


def summary():
    """Per method: trainable parameters besides the head, final accuracy over seeds (mean, min, max)."""
    rows = {}
    for run in results()["runs"]:
        rows.setdefault(run["method"], []).append(run)
    out = []
    for method, runs in rows.items():
        acc = [r["accuracy"][-1] for r in runs]
        out.append({
            "method": method,
            "added": runs[0]["trainable"] - runs[0]["head"],
            "trainable": runs[0]["trainable"],
            "total": runs[0]["total"],
            "mean": float(np.mean(acc)),
            "low": min(acc),
            "high": max(acc),
            "seconds": float(np.mean([r["seconds"] for r in runs])),
            "memory": float(np.mean([r["peak_memory"] for r in runs])),
        })
    return out


def chart_results(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    rows = sorted(summary(), key=lambda r: r["mean"])
    full = [r for r in rows if r["method"] == "full fine-tuning"][0]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.27, right=0.8, top=0.82, bottom=0.12)
    ax.axvline(100 * full["mean"], color=c["muted"], linestyle="--", linewidth=1.1)
    for k, r in enumerate(rows):
        color = c["muted"] if r is full else c["dec"]
        ax.plot([100 * r["low"], 100 * r["high"]], [k, k], color=color, linewidth=3, alpha=0.45,
                solid_capstyle="round")
        ax.plot([100 * r["mean"]], [k], "o", color=color, markersize=8)
        share = r["added"] / r["total"]
        added = "all" if r is full else ("0" if r["added"] == 0 else f"{r['added']:,}  ({share:.2%})")
        ax.text(1.02, k, added, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=c["ink"])
    ax.text(1.02, len(rows) - 0.2, "trained besides\nthe head", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=9, color=c["muted"])
    ax.set_yticks(range(len(rows)), [r["method"] for r in rows], fontsize=10)
    ax.set_xlabel("validation accuracy, %")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _caption(fig, "The same task, ten ways",
             f"bert-base, SST-2, {results()['train_size']:,} sentences, 3 epochs; dot: mean of 3 seeds, bar: range", c)
    return fig


def chart_spectrum(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    spectra = results()["delta_spectra"]
    ranks = np.arange(1, D + 1)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.15)
    curves = []
    for name, values in spectra.items():
        if name == "random":
            continue
        energy = np.cumsum(np.square(values)) / np.sum(np.square(values))
        curves.append(energy)
        ax.plot(ranks, 100 * energy, color=c["dec"], linewidth=0.8, alpha=0.25)
    random = np.cumsum(np.square(spectra["random"])) / np.sum(np.square(spectra["random"]))
    ax.plot(ranks, 100 * np.median(curves, axis=0), color=c["dec"], linewidth=2.2, label="full fine-tuning, median")
    ax.plot(ranks, 100 * random, color=c["muted"], linewidth=1.6, linestyle="--", label="a random 768×768 matrix")
    ax.axvline(8, color=c["ink"], linewidth=1, linestyle=":")
    ax.text(8.6, 96, "r = 8", fontsize=9.5, color=c["ink"], va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 768], ["1", "10", "100", "768"])
    ax.minorticks_off()
    ax.set_xlabel("number of singular directions kept")
    ax.set_ylabel("share of the update, %")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    _caption(fig, "How low is the rank of a real fine-tuning update?",
             "ΔW of the query and value matrices of all 12 layers after full fine-tuning: energy in the top directions",
             c)
    return fig


@cache
def llama():
    """Parameter count and shapes of Llama 3.1 8B, from the Hub."""
    from huggingface_hub import hf_hub_download, model_info

    repo = "unsloth/Meta-Llama-3.1-8B"
    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return model_info(repo).safetensors.total, config


def lora_params(config, r=16):
    """LoRA on every linear layer of every block: q, k, v, o and the three MLP matrices."""
    d, ff = config["hidden_size"], config["intermediate_size"]
    kv = config["num_key_value_heads"] * (d // config["num_attention_heads"])
    shapes = [(d, d), (d, kv), (d, kv), (d, d), (d, ff), (d, ff), (ff, d)]
    return config["num_hidden_layers"] * sum(r * (a + b) for a, b in shapes)


def chart_memory(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    n, config = llama()
    adapters = lora_params(config)
    gb = 1e9
    rows = [  # (name, weights, gradients, optimizer state), bytes
        ("full fine-tuning, Adam", 2 * n, 2 * n, 12 * n),
        ("LoRA, 16-bit base", 2 * n + 2 * adapters, 2 * adapters, 12 * adapters),
        ("QLoRA, 4-bit base", n * (4 + 0.127) / 8 + 2 * adapters, 2 * adapters, 12 * adapters),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.72, bottom=0.17)
    parts = [("weights", c["line"]), ("gradients", c["dec"]), ("optimizer state", c["att"])]
    for k, (_, *sizes) in enumerate(rows):
        left = 0
        for (label, color), size in zip(parts, sizes, strict=True):
            ax.barh(k, size / gb, left=left, height=0.55, color=color, label=label if k == 0 else None)
            left += size / gb
        ax.text(left + 1.5, k, f"{left:.0f} GB" if left >= 10 else f"{left:.1f} GB", va="center", fontsize=10,
                color=c["ink"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("GB, before activations")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=3, fontsize=9.5)
    _caption(fig, "What training an 8B model holds in memory",
             f"Llama 3.1 8B, {n / 1e9:.2f}B parameters; LoRA r = 16 on every linear layer, {adapters / 1e6:.0f}M "
             "parameters", c)
    return fig


A_SMALL = np.array([[1.0, 0.5], [-1.0, 2.0]])
S_SMALL = np.array([1.0, 2.0, -1.0])
T_SMALL = np.array([1.0, 0.5, -1.0, 2.0])


def chart_kronecker(theme):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    c = _style(theme)
    B = np.outer(S_SMALL, T_SMALL)
    K = np.kron(A_SMALL, B)
    cmap = LinearSegmentedColormap.from_list("signed", [c["dec"], c["card"], c["enc"]])
    limit = np.abs(K).max()
    fig = plt.figure(figsize=(7.6, 3.3))
    panels = [
        (0.03, 0.2, "A", A_SMALL, True),
        (0.24, 0.27, "B = s tᵀ  (rank 1)", B, True),
        (0.58, 0.39, "A ⊗ B", K, False),
    ]
    for x, w, title, matrix, numbers in panels:
        rows, cols = matrix.shape
        h = 0.62 * rows / 6
        ax = fig.add_axes((x, 0.12 + (0.62 - h) / 2, w, h))
        ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color=c["card"], linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, fontsize=10.5, color=c["ink"], loc="center")
        if numbers:
            for i in range(rows):
                for j in range(cols):
                    ax.text(j, i, f"{matrix[i, j]:g}", ha="center", va="center", fontsize=9, color=c["ink"])
        if matrix is K:
            for k in (cols // 2 - 0.5,):
                ax.axvline(k, color=c["ink"], linewidth=1.4)
            ax.axhline(rows // 2 - 0.5, color=c["ink"], linewidth=1.4)
    fig.text(0.235, 0.47, "⊗", fontsize=16, color=c["ink"], ha="center")
    fig.text(0.535, 0.47, "=", fontsize=16, color=c["ink"], ha="center")
    _caption(fig, "A Kronecker product: a big matrix from two small ones",
             "each block of A ⊗ B is B scaled by one entry of A; Compacter sums n such products", c)
    return fig


def chart_alts():
    rows = {r["method"]: r for r in summary()}
    full = rows["full fine-tuning"]
    lora = rows["LoRA, q and v, r = 8"]
    spectra = results()["delta_spectra"]
    tops = []
    for name, values in spectra.items():
        if name != "random":
            v = np.square(values)
            tops.append(v[:8].sum() / v.sum())
    noise = np.square(spectra["random"])
    random_top = noise[:8].sum() / noise.sum()
    n, config = llama()
    return {
        "memory": f"Memory to train Llama 3.1 8B before activations: about {16 * n / 1e9:.0f} GB for full "
        f"fine-tuning with Adam, about {(2 * n + 16 * lora_params(config)) / 1e9:.0f} GB with LoRA on a 16-bit base, "
        f"about {(n * 4.127 / 8 + 16 * lora_params(config)) / 1e9:.1f} GB with QLoRA's 4-bit base",
        "kronecker": "The Kronecker product of a 2×2 matrix A and a rank-one 3×4 matrix B: a 6×8 matrix made of "
        "four blocks, each a copy of B multiplied by one entry of A",
        "results": f"Validation accuracy on SST-2 against the number of trained parameters. Full fine-tuning reaches "
        f"{100 * full['mean']:.1f}%; LoRA with {lora['added']:,} parameters reaches {100 * lora['mean']:.1f}%; "
        + "; ".join(f"{m} {100 * r['mean']:.1f}%" for m, r in rows.items() if m not in ("full fine-tuning",)),
        "spectrum": f"Cumulative share of the squared singular values of the weight updates made by full "
        f"fine-tuning, for the query and value matrices of all 12 layers. The top 8 directions hold a median of "
        f"{100 * np.median(tops):.0f}% of the update, against {100 * random_top:.0f}% for a random matrix",
    }


# ---------------------------------------------------------------- cover

def cover():
    r = 8
    side = 214
    px = side / D
    thin = max(r * px * 3, 7)
    x0, y0 = cv.MOTIF_X0 + 4, 150
    out = [cv.label(cv.MOTIF_X0, 128, f"ONE {D}x{D} PROJECTION OF BERT")]
    out.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" rx="4" fill="{cv.CHIP_FILL}" '
               f'stroke="{cv.LINE}" stroke-width="1.6"/>')
    out.append(cv.text(x0 + side / 2, y0 + side / 2 + 10, "W", 34, cv.FAINT, cv.SANS, 600, "middle"))
    bx = x0 + side + 46
    out.append(cv.text(x0 + side + 18, y0 + side / 2 + 10, "+", 30, cv.MUTED, cv.SANS, 400, "middle"))
    out.append(f'<rect x="{bx}" y="{y0 + thin + 8}" width="{thin}" height="{side}" rx="2" fill="{cv.RED}"/>')
    out.append(f'<rect x="{bx + thin + 8}" y="{y0}" width="{side}" height="{thin}" rx="2" fill="{cv.RED}"/>')
    out.append(cv.text(bx + thin + 18 + side / 2, y0 + side / 2 + 18, "B A", 26, cv.RED, cv.MONO, 500, "middle"))
    cv.fits(bx + thin + 8 + side)
    out.append(cv.label(cv.MOTIF_X0, 450, "TRAINED NUMBERS"))
    for row, (name, value, kind) in enumerate([("full update", f"{D * D:,}", "plain"),
                                               (f"LoRA, rank {r}", f"{2 * D * r:,}", "hot")]):
        y = 466 + row * 52
        out.append(cv.text(cv.MOTIF_X0, y + 26, name, 15, cv.FAINT, cv.MONO))
        chip, w = cv.chip(cv.MOTIF_X0 + 190, y, value, size=17, kind=kind)
        cv.fits(cv.MOTIF_X0 + 190 + w)
        out.append(chip)
    return cv.frame("blog / machine learning", ["Adapters", "and LoRA"], ["Fine-tuning by changing", "almost nothing"],
                    "".join(out), footer="PEFT in PyTorch, measured")


COVER = cover

CHARTS = {
    "memory": chart_memory,
    "kronecker": chart_kronecker,
    "results": chart_results,
    "spectrum": chart_spectrum,
}

FIGURES = {
    "map-sequential": map_sequential,
    "map-parallel": map_parallel,
    "map-lora": map_lora,
    "map-ia3": map_ia3,
    "map-prefix": map_prefix,
    "map-prompt": map_prompt,
    "map-bitfit": map_bitfit,
    "bottleneck": bottleneck,
    "lora-shapes": lora_shapes,
}
