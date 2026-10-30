"""Figures, charts and the cover of the post "Hybrid attention" on xmarva.github.io.

Facts about real models come from their config.json files on the Hugging Face Hub, kept in data/configs/ as they
were when the post was written. Measurements come from data/speed.json (speed.py), run on a Kaggle T4, and
data/mqar.json (mqar.py), run partly on a T4 and partly on an Apple M2; mqar.py says which part each run covered. The
memory experiment of `chart_memory` is small enough to run here.

    uv run --group articles python tools/render.py --article hybrid_attention --out <site>/assets/posts/hybrid-attention
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from tools.figures import cover as cv
from tools.figures.lib import Figure

DATA = Path(__file__).parent / "data"
KiB, GiB = 2**10, 2**30


# ---------------------------------------------------------------- three ways to remember


def memories() -> Figure:
    fig = Figure(
        760,
        360,
        alt="Three ways for a layer to remember the tokens it has read. Softmax attention keeps the key and the "
        "value of every token, so its memory grows with the text, and a query is compared with every key. Linear "
        "attention keeps one matrix and adds the outer product of each key and value to it, so its memory has a "
        "fixed size. The delta rule keeps the same matrix but first reads what it holds under the new key and "
        "writes only the difference",
    )
    fig.text(14, 28, "What a layer keeps after reading t tokens", kind="title")
    fig.text(14, 50, "and how a query reads it", kind="note")
    panels = [("softmax attention", "att"), ("linear attention", "enc"), ("delta rule", "enc")]
    width = 236
    for p, (name, role) in enumerate(panels):
        x0 = 14 + p * (width + 12)
        fig.box(x0, 68, width, 280, role="plain", r=10)
        fig.text(x0 + 14, 92, name, kind="subtitle")
        if p == 0:
            rows = ["k1  v1", "k2  v2", "k3  v3", "…", "kt  vt"]
            for r, label in enumerate(rows):
                y = 108 + r * 30
                if label == "…":
                    fig.text(x0 + 78, y + 18, "…", kind="label", anchor="middle")
                    continue
                fig.box(x0 + 30, y, 96, 24, role=role, label=label)
                fig.line((x0 + 186, 186), (x0 + 128, y + 12), role="att", width=1)
            fig.box(x0 + 168, 172, 40, 28, role=role, label="q", strong=True)
            fig.text(x0 + 14, 280, "a row per token: grows with t", kind="small")
            fig.text(x0 + 14, 300, "q is compared with every key", kind="small")
            fig.text(x0 + 14, 326, "memory O(t), read O(t)", kind="mono_att")
        else:
            cell, gx, gy = 13, x0 + 34, 112
            # linear attention: every write lands on top of the others, so cells pile up everywhere; the delta
            # rule rewrites what one key points to, drawn as one row in red
            piled = {(1, 2), (2, 5), (3, 1), (4, 4), (5, 6), (6, 3), (0, 7), (7, 0), (2, 2), (5, 1)}
            for i in range(8):
                for j in range(8):
                    if p == 2 and i == 4:
                        fig.rect(gx + j * cell, gy + i * cell, cell - 2, cell - 2, role="dec", r=2)
                    elif p == 1 and (i, j) in piled:
                        fig.rect(gx + j * cell, gy + i * cell, cell - 2, cell - 2, role=role, r=2)
                    else:
                        fig.rect(gx + j * cell, gy + i * cell, cell - 2, cell - 2, role=role, tint=True, r=2)
            fig.text(gx + 4 * cell, gy - 6, "S, d × d", kind="small", anchor="middle")
            if p == 1:
                fig.text(gx + 8 * cell + 14, gy + 40, "S += k vᵀ", kind="mono")
                fig.text(x0 + 14, 280, "every pair is added on top", kind="small")
                fig.text(x0 + 14, 300, "of the others", kind="small")
            else:
                fig.text(gx + 8 * cell + 14, gy + 30, "read Sᵀk,", kind="mono")
                fig.text(gx + 8 * cell + 14, gy + 52, "write the", kind="mono")
                fig.text(gx + 8 * cell + 14, gy + 74, "difference", kind="mono")
                fig.text(x0 + 14, 280, "an old value under the same", kind="small")
                fig.text(x0 + 14, 300, "key is replaced, not piled on", kind="small")
            fig.text(x0 + 14, 326, "memory O(1), read O(1)", kind="mono")
    return fig


# ---------------------------------------------------------------- one step of the delta rule

def delta_step() -> Figure:
    # a 2-dimensional memory: what it returns for k before the write, and after writes of strength 0.5 and 1,
    # computed with the update S <- S + beta k (v - S^T k)^T
    k = np.array([1.0, 0.0])
    memory = np.array([[0.8, 1.6], [0.3, -0.2]])  # rows: key coordinates; S^T k is its first row
    target = np.array([2.6, 0.7])
    returned = {}
    for beta in (0.0, 0.5, 1.0):
        updated = memory + beta * np.outer(k, target - memory.T @ k)
        returned[beta] = updated.T @ k
    fig = Figure(
        760,
        320,
        alt="One write of the delta rule, drawn in the space of values. The memory returns an old value for the "
        "key; the new value is elsewhere; the error is the arrow between them. A write of strength 0.5 moves the "
        "memory's answer halfway along the error, a write of strength 1 lands on the new value exactly. Plain "
        "addition would instead add the new value to the old one",
    )
    fig.text(14, 28, "One write of the delta rule, in the space of values", kind="title")
    fig.text(14, 50, "the memory moves its answer for k towards v by a fraction β of the error", kind="note")
    x0, y0, scale = 120, 270, 120  # the origin of value space, and pixels per unit

    def at(point):
        return (x0 + scale * point[0], y0 - scale * point[1])

    fig.line((x0 - 20, y0), (x0 + 3.4 * scale, y0), role="muted", width=1)
    fig.line((x0, y0 + 14), (x0, y0 - 1.6 * scale), role="muted", width=1)
    old, half, full = at(returned[0.0]), at(returned[0.5]), at(returned[1.0])
    fig.arrow((x0, y0), old, role="muted", width=1.6)
    fig.arrow((x0, y0), full, role="data", width=1.8)
    fig.line(old, full, role="dec", width=1.6, dash=True)
    fig.rect(half[0] - 5, half[1] - 5, 10, 10, role="enc", r=5)
    fig.text(old[0] + 12, old[1] - 8, "Sᵀk: what the memory returns now", kind="label")
    fig.text(full[0] + 12, full[1] + 4, "v: the new value", kind="mono")
    fig.text(half[0] + 10, half[1] - 12, "β = 0.5", kind="mono")
    fig.text(full[0] + 12, full[1] + 26, "β = 1 lands here", kind="small")
    error_mid = ((old[0] + full[0]) / 2, (old[1] + full[1]) / 2)
    fig.text(error_mid[0] - 24, error_mid[1] + 46, "error v − Sᵀk", kind="mono_dec", anchor="end")
    fig.text(530, 120, "plain addition would return", kind="small")
    fig.text(530, 140, "old + new: a mixture of both", kind="small")
    return fig


# ---------------------------------------------------------------- the same product, in another order

def associativity() -> Figure:
    n, d = 32_768, 128
    fig = Figure(
        760,
        300,
        alt=f"The same product computed in two orders, for {n:,} tokens and {d}-dimensional heads. Multiplying the "
        f"queries by the keys first gives a {n:,} by {n:,} matrix of scores, {n * n:,} numbers, before the values "
        f"are applied. Multiplying the keys by the values first gives a {d} by {d} matrix, {d * d:,} numbers, that "
        f"every query then reads. Without the softmax between them, the order is free",
    )
    fig.text(14, 28, "One product, two orders", kind="title")
    fig.text(14, 50, f"n = {n:,} tokens, d = {d}: without a softmax in the middle, the brackets can move",
             kind="note")
    for row, (y, left, middle, size, role) in enumerate([
        (86, "(Q Kᵀ) V", f"Q Kᵀ: {n:,} × {n:,}", f"{n * n:,} numbers", "att"),
        (196, "Q (Kᵀ V)", f"Kᵀ V: {d} × {d}", f"{d * d:,} numbers", "enc"),
    ]):
        fig.text(14, y + 34, left, kind="mono")
        if row == 0:
            fig.rect(150, y, 16, 70, role="data", tint=True, r=2)
            fig.text(158, y + 86, "Q", kind="small", anchor="middle")
            fig.rect(176, y + 27, 70, 16, role="data", tint=True, r=2)
            fig.text(211, y + 86, "Kᵀ", kind="small", anchor="middle")
            fig.text(262, y + 40, "=", kind="mono", anchor="middle")
            fig.rect(280, y, 70, 70, role=role, r=2)
        else:
            fig.rect(150, y + 27, 70, 16, role="data", tint=True, r=2)
            fig.text(185, y + 86, "Kᵀ", kind="small", anchor="middle")
            fig.rect(230, y, 16, 70, role="data", tint=True, r=2)
            fig.text(238, y + 86, "V", kind="small", anchor="middle")
            fig.text(262, y + 40, "=", kind="mono", anchor="middle")
            fig.rect(299, y + 27, 16, 16, role=role, r=2)
        fig.text(378, y + 26, middle, kind="mono_att" if row == 0 else "mono")
        fig.text(378, y + 50, size, kind="small")
        fig.text(378, y + 70, "grows as n²" if row == 0 else "does not depend on n", kind="note")
    return fig


# ---------------------------------------------------------------- chunks

def chunks() -> Figure:
    fig = Figure(
        760,
        250,
        alt="How a recurrent layer is trained in parallel. The sequence is cut into chunks of 64 tokens. Inside a "
        "chunk, all positions are computed at once with matrix products; between chunks, only the d by d state is "
        "passed on, one step per chunk instead of one per token",
    )
    fig.text(14, 28, "Training a recurrence in parallel: chunks", kind="title")
    fig.text(14, 50, "inside a chunk, matrix products for all 64 positions; between chunks, only the state",
             kind="note")
    x = 30
    for c in range(4):
        fig.box(x, 90, 140, 56, role="enc")
        for t in range(8):
            fig.rect(x + 10 + t * 15, 104, 11, 28, role="enc", tint=True, r=2)
        fig.text(x + 70, 168, f"tokens {64 * c + 1}–{64 * (c + 1)}", kind="small", anchor="middle")
        if c < 3:
            fig.arrow((x + 144, 118), (x + 172, 118), role="enc")
            fig.text(x + 158, 108, "S", kind="mono", anchor="middle")
        x += 176
    fig.text(30 + 4 * 176 - 8, 124, "…", kind="label", anchor="end")
    fig.text(14, 210, "a sequence of 4,096 tokens: 64 steps from chunk to chunk instead of 4,096 from token to token",
             kind="note")
    fig.text(14, 232, "the production kernels are written this way: flash-linear-attention, in Triton", kind="small")
    return fig


# ---------------------------------------------------------------- real models, from their config.json


def config(repo):
    c = json.loads((DATA / "configs" / f"{repo.replace('/', '__')}.json").read_text())
    return {**c, **c.get("text_config", {})}


def layers(repo):
    """The token-mixing layers of a model, in order: "full" attention, a sliding "window", or a fixed "state"."""
    c = config(repo)
    n = c["num_hidden_layers"]
    if c.get("layer_types"):
        kinds = {"full_attention": "full", "sliding_attention": "window", "linear_attention": "state"}
        return [kinds[t] for t in c["layer_types"]]
    if "full_attention_interval" in c:
        return ["full" if (i + 1) % c["full_attention_interval"] == 0 else "state" for i in range(n)]
    if "linear_attn_config" in c:  # Kimi: 1-indexed lists of layers
        full = set(c["linear_attn_config"]["full_attn_layers"])
        return ["full" if i + 1 in full else "state" for i in range(n)]
    if "attn_type_list" in c:  # MiniMax: 1 is softmax attention, 0 lightning attention
        return ["full" if t == 1 else "state" for t in c["attn_type_list"]]
    if "attn_layer_period" in c:  # Jamba
        return ["full" if i % c["attn_layer_period"] == c["attn_layer_offset"] else "state" for i in range(n)]
    if "hybrid_override_pattern" in c:  # Nemotron: M is Mamba-2, * attention; MLP (-) and MoE (E) mix no tokens
        return [{"M": "state", "*": "full"}[s] for s in c["hybrid_override_pattern"] if s in "M*"]
    return ["full"] * n  # a plain Transformer


def kv_bytes(repo, tokens):
    """Keys and values one sequence keeps in bf16 after `tokens` tokens, in the attention layers only."""
    c = config(repo)
    if "kv_lora_rank" in c:  # MLA: engines cache one compressed latent per token instead of keys and values
        per_token = (c["kv_lora_rank"] + c["qk_rope_head_dim"]) * 2
    else:
        dim = c.get("head_dim") or c.get("attention_head_dim") or c["hidden_size"] // c["num_attention_heads"]
        per_token = 2 * c["num_key_value_heads"] * dim * 2
    kept = {"full": tokens, "window": min(tokens, c.get("sliding_window") or tokens), "state": 0}
    return sum(per_token * kept[kind] for kind in layers(repo))


def state_bytes(repo):
    """The fixed state one sequence keeps in the recurrent layers, in bf16; short-convolution buffers left out."""
    c = config(repo)
    if "linear_num_value_heads" in c:  # Gated DeltaNet: a k x v matrix per value head
        per_layer = c["linear_num_value_heads"] * c["linear_key_head_dim"] * c["linear_value_head_dim"]
    elif "linear_attn_config" in c:  # Kimi Delta Attention
        per_layer = c["linear_attn_config"]["num_heads"] * c["linear_attn_config"]["head_dim"] ** 2
    elif "attn_type_list" in c:  # lightning attention: the same heads as the softmax layers
        per_layer = c["num_attention_heads"] * c["head_dim"] ** 2
    elif "mamba_d_state" in c:  # Mamba: a state vector per inner channel
        per_layer = c["mamba_expand"] * c["hidden_size"] * c["mamba_d_state"]
    else:  # Mamba-2: a head_dim x state matrix per head
        per_layer = c["mamba_num_heads"] * c["mamba_head_dim"] * c["ssm_state_size"]
    return per_layer * layers(repo).count("state") * 2


PATTERNS = [
    ("a recurrent layer with a fixed state, and full attention", [
        ("Jamba", "ai21labs/Jamba-v0.1", "Mamba, March 2024"),
        ("MiniMax-Text-01", "MiniMaxAI/MiniMax-Text-01", "lightning attention, January 2025"),
        ("Nemotron-H-8B", "nvidia/Nemotron-H-8B-Base-8K", "Mamba-2, April 2025; MLP layers not drawn"),
        ("Qwen3-Next-80B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct", "Gated DeltaNet, September 2025"),
        ("Kimi Linear 48B-A3B", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "Kimi Delta Attention, October 2025"),
        ("Qwen3.5-397B-A17B", "Qwen/Qwen3.5-397B-A17B", "Gated DeltaNet, February 2026"),
        ("Kimi K3", "moonshotai/Kimi-K3", "Kimi Delta Attention, June 2026"),
    ]),
    ("a sliding window, and full attention", [
        ("gpt-oss-20b", "openai/gpt-oss-20b", "window of 128 tokens, August 2025"),
        ("Gemma 4 31B", "google/gemma-4-31B-it", "window of 1,024 tokens, March 2026"),
    ]),
    ("full attention in every layer", [
        ("MiniMax-M2", "MiniMaxAI/MiniMax-M2", "October 2025, after MiniMax-Text-01"),
    ]),
]


def _width(s, size=14):
    """Width in px of a subtitle (IBM Plex Sans SemiBold) on the figure's canvas."""
    from PIL import ImageFont

    import tfs.viz

    font = Path(tfs.viz.__file__).parent / "fonts" / "IBMPlexSans-SemiBold.ttf"
    return ImageFont.truetype(str(font), size).getlength(s)


def patterns() -> Figure:
    rows = [(name, layers(repo), note) for _, group in PATTERNS for name, repo, note in group]
    pitch = 7.8  # px per layer, the same for every model, so the length of a row is its depth
    height = 112 + sum(30 + 46 * len(group) for _, group in PATTERNS)
    fig = Figure(
        760,
        height,
        alt="The layers of real models, one mark per token-mixing layer, read from their config files. "
        + ". ".join(f"{name}: {kinds.count('full')} of {len(kinds)} layers are full attention"
                    + (f", {kinds.count('window')} use a sliding window" if "window" in kinds else "")
                    for name, kinds, _ in rows)
        + ". MiniMax-M2, the successor of MiniMax-Text-01, uses full attention in all of its layers",
    )
    fig.text(14, 28, "Where real models keep full attention", kind="title")
    fig.text(14, 50, "one mark per token-mixing layer, from the bottom of the model to the top; read from config.json",
             kind="note")
    legend = [("att", True, "full attention: a cache that grows"), ("att", False, "sliding window"),
              ("enc", False, "recurrent: a state of fixed size")]
    x = 14
    for role, strong, label in legend:
        fig.box(x, 66, 6, 16, role=role, r=1.5, strong=strong)
        fig.text(x + 14, 79, label, kind="label")
        x += 26 + 7.2 * len(label)
    y = 112
    for title, group in PATTERNS:
        fig.text(14, y, title, kind="label")
        fig.line((14, y + 8), (746, y + 8), role="muted", width=0.6)
        y += 30
        for name, repo, note in group:
            kinds = layers(repo)
            fig.text(14, y, name, kind="subtitle")
            fig.text(14 + _width(name) + 16, y, note, kind="note")
            fig.text(746, y, f"{kinds.count('full')} of {len(kinds)}", kind="mono_att", anchor="end")
            for i, kind in enumerate(kinds):
                role = "enc" if kind == "state" else "att"
                fig.box(14 + i * pitch, y + 9, pitch - 2.2, 18, role=role, r=1.5, strong=kind == "full")
            y += 46
    return fig


# ---------------------------------------------------------------- the memory experiment

D_STATE = 64


@cache
def memory_experiment():
    rng = np.random.default_rng(0)
    trials = 20

    def unit(rows):
        x = rng.standard_normal((rows, D_STATE))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def recall(keys, values, queries, candidates, targets):
        added = keys.T @ values
        edited = np.zeros((D_STATE, D_STATE))
        for k, v in zip(keys, values, strict=True):
            edited += np.outer(k, v - edited.T @ k)
        weights = np.exp(queries @ keys.T / 0.05)
        weights /= weights.sum(1, keepdims=True)
        out = {}
        for name, read in (("added", queries @ added), ("edited", queries @ edited), ("softmax", weights @ values)):
            out[name] = float(np.mean(np.argmax(read @ candidates.T, 1) == targets))
        return out

    distinct = []
    for n in [16, 32, 64, 96, 128, 192, 256, 384, 512]:
        scores = [recall(k, v, k, v, np.arange(n)) for k, v in ((unit(n), unit(n)) for _ in range(trials))]
        distinct.append({"pairs": n, **{m: float(np.mean([s[m] for s in scores])) for m in scores[0]}})
    overwritten = []
    n = 32
    for m in [1, 2, 4, 8, 16, 32]:
        scores = []
        for _ in range(trials):
            k, v, new = unit(n), unit(n), unit(m)
            keys, values = np.concatenate([k, k[:m]]), np.concatenate([v, new])
            scores.append(recall(keys, values, k[:m], values, np.arange(n, n + m)))
        overwritten.append({"rewritten": m, **{name: float(np.mean([s[name] for s in scores])) for name in scores[0]}})
    return {"distinct": distinct, "overwritten": overwritten, "stored": n}


def _style(theme):
    from tfs.viz.style import use_course_style

    return use_course_style(theme)


def _caption(fig, title, note, c):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=c["ink"])
    fig.text(0.012, 0.895, note, ha="left", va="top", fontsize=10, color=c["muted"])


def chart_memory(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    m = memory_experiment()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.17, wspace=0.3)
    styles = {"softmax": ("keep every pair (softmax)", c["att"]), "added": ("add, k vᵀ", c["muted"]),
              "edited": ("edit, delta rule", c["enc"])}
    pairs = [row["pairs"] for row in m["distinct"]]
    for name, (label, color) in styles.items():
        left.plot(pairs, [row[name] for row in m["distinct"]], color=color, marker="o", markersize=4, label=label)
    left.axvline(D_STATE, color=c["line"], linestyle="--", linewidth=1)
    left.text(D_STATE * 1.06, 0.06, f"d = {D_STATE}", fontsize=9.5, color=c["muted"])
    left.set_xscale("log", base=2)
    left.set_xticks(pairs[::2], [str(p) for p in pairs[::2]])
    left.minorticks_off()
    left.set_ylim(0, 1.05)
    left.set_title("distinct pairs stored", fontsize=11)
    left.set_xlabel("pairs written")
    left.set_ylabel("share recalled")
    rewritten = [row["rewritten"] for row in m["overwritten"]]
    for name, (label, color) in styles.items():
        right.plot(rewritten, [row[name] for row in m["overwritten"]], color=color, marker="o", markersize=4,
                   label=label)
    right.set_xscale("log", base=2)
    right.set_xticks(rewritten, [str(r) for r in rewritten])
    right.minorticks_off()
    right.set_ylim(0, 1.05)
    right.set_title(f"{m['stored']} pairs, then new values", fontsize=11)
    right.set_xlabel("keys given a new value")
    left.legend(loc="lower left", fontsize=9.5)
    _caption(fig, f"A {D_STATE} × {D_STATE} memory: how much it holds, and whether it can be corrected",
             "random unit keys and values; a value counts as recalled when it is the nearest to what the memory "
             "returns", c)
    return fig


MQAR_STYLE = {  # model -> (palette key, line style): attention violet, recurrent teal and amber, the hybrids rose
    "Transformer": ("att", "-"),
    "Hybrid (attention 2nd)": ("dec", "-"),
    "Hybrid (attention on top)": ("dec", "--"),
    "Gated DeltaNet": ("enc", "-"),
    "DeltaNet": ("data", "-"),
    "Linear attention": ("muted", "-"),
}


@cache
def mqar():
    """The better of the two learning rates for every model and number of pairs: {model: {pairs: accuracy}}, and
    the numbers of pairs every model was trained on."""
    data = json.loads((DATA / "mqar.json").read_text())
    best = {}
    for run in data["runs"]:
        row = best.setdefault(run["model"], {})
        row[run["pairs"]] = max(row.get(run["pairs"], 0.0), run["accuracy"])
    pairs = sorted(set.intersection(*(set(row) for row in best.values())))
    return data, best, pairs


def chart_mqar(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    data, best, pairs = mqar()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fig.subplots_adjust(left=0.09, right=0.66, top=0.8, bottom=0.14)
    for model, (key, style) in MQAR_STYLE.items():
        # the Transformer is drawn wide and underneath, so it still shows where a hybrid runs along the same values
        wide = model == "Transformer"
        ax.plot(pairs, [best[model][p] for p in pairs], color=c[key], linestyle=style, marker="o",
                markersize=8 if wide else 4.5, label=model, linewidth=6.0 if wide else 2.0, zorder=1 if wide else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(pairs, [str(p) for p in pairs])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("key-value pairs in the sequence")
    ax.set_ylabel("values recalled")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    _caption(fig, "Associative recall: four layers of width 64",
             f"{data['steps']:,} training steps of {data['batch']} fresh sequences; held-out accuracy at the better "
             "learning rate where two were tried", c)
    return fig


@cache
def speed():
    return json.loads((DATA / "speed.json").read_text())


def _length_axis(ax, lengths):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths, [f"{n // 1024}K" if n >= 1024 else str(n) for n in lengths])
    ax.minorticks_off()


def _log_y(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))


def chart_prefill(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = speed()
    rows = s["prefill"]
    lengths = [r["length"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.15)
    ax.plot(lengths, [r["softmax"] * 1e3 for r in rows], color=c["att"], marker="o", markersize=4,
            label="softmax attention: PyTorch's fused kernel, fp16")
    done = [r for r in rows if r["gated_delta"] is not None]
    ax.plot([r["length"] for r in done], [r["gated_delta"] * 1e3 for r in done], color=c["enc"], marker="o",
            markersize=4, label="Gated DeltaNet: chunks of 64, plain PyTorch, fp32")
    _length_axis(ax, lengths)
    _log_y(ax)
    ax.set_xlabel("tokens read at once")
    ax.set_ylabel("milliseconds")
    ax.legend(loc="upper left", fontsize=9.5)
    _caption(fig, "Reading a prompt: one layer, 16 heads of 128", f"{s['device']}; the best of several runs", c)
    return fig


def chart_decode(theme):
    import matplotlib.pyplot as plt

    c = _style(theme)
    s = speed()
    rows = s["decode"]
    lengths = [r["length"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.17, wspace=0.32)
    left.plot(lengths, [r["softmax"] * 1e3 for r in rows], color=c["att"], marker="o", markersize=4,
              label="softmax attention")
    left.plot(lengths, [r["gated_delta"] * 1e3 for r in rows], color=c["enc"], marker="o", markersize=4,
              label="Gated DeltaNet")
    right.plot(lengths, [r["kv_bytes"] / 2**20 for r in rows], color=c["att"], marker="o", markersize=4)
    right.plot(lengths, [r["state_bytes"] / 2**20 for r in rows], color=c["enc"], marker="o", markersize=4)
    for ax in (left, right):
        _length_axis(ax, lengths[::2])
        ax.set_xlabel("tokens before it")
    left.set_ylim(bottom=0)
    left.set_title("time per new token, ms", fontsize=11)
    left.legend(loc="upper left", fontsize=9.5)
    _log_y(right)
    right.set_title("what it reads, MiB", fontsize=11)
    _caption(fig, "Writing one token after a long context: one layer, 16 heads of 128",
             f"{s['device']}; keys and values in fp16, the state in fp32", c)
    return fig


KV_MODELS = [  # (name, repo, how its layers mix tokens)
    ("Qwen3-32B", "Qwen/Qwen3-32B", "full"),
    ("MiniMax-M2", "MiniMaxAI/MiniMax-M2", "full"),
    ("Qwen3-8B", "Qwen/Qwen3-8B", "full"),
    ("MiniMax-Text-01", "MiniMaxAI/MiniMax-Text-01", "state"),
    ("gpt-oss-20b", "openai/gpt-oss-20b", "window"),
    ("Qwen3-Next", "Qwen/Qwen3-Next-80B-A3B-Instruct", "state"),
    ("Jamba", "ai21labs/Jamba-v0.1", "state"),
    ("Kimi Linear", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "state"),
    ("Nemotron 3 Nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "state"),
]
KV_TOKENS = 128 * 1024


def chart_kv(theme):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    c = _style(theme)
    colors = {"full": c["att"], "window": c["data"], "state": c["enc"]}
    names = [name for name, _, _ in KV_MODELS]
    sizes = [kv_bytes(repo, KV_TOKENS) / GiB for _, repo, _ in KV_MODELS]
    states = [state_bytes(repo) / 2**20 for _, repo, kind in KV_MODELS if kind == "state"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.12)
    y = np.arange(len(names))
    ax.barh(y, sizes, height=0.62, color=[colors[kind] for _, _, kind in KV_MODELS])
    for row, size in zip(y, sizes, strict=True):
        ax.text(size + 0.5, row, f"{size:.2g} GiB", va="center", fontsize=9.5, color=c["muted"])
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(sizes) * 1.15)
    ax.set_xlabel("GiB of keys and values, bf16")
    ax.legend(handles=[Patch(color=c["att"], label="full attention in every layer"),
                       Patch(color=c["data"], label="sliding window + full attention"),
                       Patch(color=c["enc"], label="recurrent + full attention")],
              loc="lower right", fontsize=9.5)
    _caption(fig, "The cache of one sequence of 128K tokens",
             f"derived from each model's config.json; the fixed state of the recurrent layers, {min(states):.0f} to "
             f"{max(states):.0f} MiB in bf16, is left out", c)
    return fig


# ---------------------------------------------------------------- the cover

def cover():
    pattern = ["linear"] * 3 + ["full"]
    out = [cv.label(cv.MOTIF_X0, 128, "48 LAYERS, 12 OF THEM ATTENTION")]
    x0, y0, size, gap = cv.MOTIF_X0 + 4, 158, 34, 8
    for i in range(48):
        row, column = divmod(i, 12)
        kind = pattern[i % 4]
        x, y = x0 + column * (size + gap), y0 + row * (size + gap)
        if kind == "full":
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="6" fill="{cv.RED}"/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="6" fill="{cv.CHIP_FILL}" '
                       f'stroke="{cv.LINE}" stroke-width="1.4"/>')
    cv.fits(x0 + 12 * (size + gap) - gap)
    legend_y = y0 + 4 * (size + gap) + 30
    out.append(f'<rect x="{x0}" y="{legend_y - 14}" width="16" height="16" rx="4" fill="{cv.RED}"/>')
    out.append(cv.text(x0 + 26, legend_y, "attention: a cache that grows", 16, cv.MUTED, cv.MONO))
    out.append(f'<rect x="{x0}" y="{legend_y + 22}" width="16" height="16" rx="4" fill="{cv.CHIP_FILL}" '
               f'stroke="{cv.LINE}" stroke-width="1.4"/>')
    out.append(cv.text(x0 + 26, legend_y + 36, "linear: a state of fixed size", 16, cv.MUTED, cv.MONO))
    return cv.frame("blog / machine learning", ["Hybrid", "attention"], ["Why new models keep", "only a few attention",
                                                                            "layers"], "".join(out),
                    footer="Linear attention, the delta rule and the hybrids")


def chart_alts():
    m = memory_experiment()
    distinct = {row["pairs"]: row for row in m["distinct"]}
    last = m["overwritten"][-1]
    alts = {
        "memory": f"Left: share of {D_STATE}-dimensional key-value pairs recalled from a {D_STATE} by {D_STATE} "
        f"memory. Keeping every pair, as softmax attention does, recalls all of them at any number. Adding outer "
        f"products recalls {distinct[128]['added']:.0%} of 128 pairs and {distinct[512]['added']:.0%} of 512; "
        f"editing with the delta rule recalls {distinct[128]['edited']:.0%} and {distinct[512]['edited']:.0%}, "
        f"keeping the newest. Right: after {m['stored']} pairs, some keys get a new value; the delta rule returns the "
        f"new value {last['edited']:.0%} of the time, adding and softmax attention about half the time",
        "kv-memory": "Keys and values one sequence of 128K tokens keeps in bf16, derived from each model's config: "
        + ", ".join(f"{name} {kv_bytes(repo, KV_TOKENS) / GiB:.2g} GiB" for name, repo, _ in KV_MODELS)
        + ". Qwen3-32B, MiniMax-M2 and Qwen3-8B use full attention in every layer, gpt-oss-20b a sliding window in "
        "half of them, the others recurrent layers with full attention in a few",
    }
    s = speed()
    first, last = s["prefill"][0], s["prefill"][-1]
    alts["prefill"] = (
        f"Time for one layer with 16 heads of 128 to read a prompt, on a {s['device']}, on logarithmic axes. "
        f"Softmax attention takes {first['softmax'] * 1e3:.2f} ms for {first['length']} tokens and "
        f"{last['softmax'] * 1e3:,.0f} ms for {last['length']:,}; the chunked Gated DeltaNet in plain PyTorch "
        f"takes {first['gated_delta'] * 1e3:.1f} ms and {last['gated_delta'] * 1e3:.0f} ms. Attention is faster up "
        "to 8K tokens, the recurrent layer from 16K on"
    )
    first, last = s["decode"][0], s["decode"][-1]
    alts["decode"] = (
        f"Writing one token after a context of {first['length']} to {last['length']:,} tokens. Left: softmax "
        f"attention takes {first['softmax'] * 1e3:.2f} ms to {last['softmax'] * 1e3:.1f} ms, growing with the "
        f"context; Gated DeltaNet about {last['gated_delta'] * 1e3:.1f} ms at every length. Right: attention reads "
        f"{first['kv_bytes'] / 2**20:.0f} to {last['kv_bytes'] / 2**20:.0f} MiB of keys and values, the recurrent "
        f"layer a state of {last['state_bytes'] / 2**20:.0f} MiB"
    )
    data, best, pairs = mqar()
    alts["mqar"] = (
        f"Share of values recalled with {', '.join(map(str, pairs[:-1]))} and {pairs[-1]} key-value pairs, by six "
        f"models of four layers and width {data['d_model']}: "
        + "; ".join(f"{model} " + ", ".join(f"{best[model][p]:.0%}" for p in pairs) for model in MQAR_STYLE)
    )
    return alts


CHARTS = {
    "memory": chart_memory,
    "mqar": chart_mqar,
    "kv-memory": chart_kv,
    "prefill": chart_prefill,
    "decode": chart_decode,
}

COVER = cover

FIGURES = {
    "memories": memories,
    "delta-step": delta_step,
    "associativity": associativity,
    "chunks": chunks,
    "patterns": patterns,
}
