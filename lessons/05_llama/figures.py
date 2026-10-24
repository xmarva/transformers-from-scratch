"""Static figures and covers of lesson 5.

Facts about real models are read at render time: Llama 3.1 8B's config.json from the Hugging Face Hub (an ungated
copy of Meta's weights), our model's config.json from the course's repository. The RMSNorm and RoPE pictures
compute their numbers from the same formulas the lesson writes.
"""

import json
import math

from tools.figures import cover as cv
from tools.figures.lib import Figure

LLAMA = "unsloth/Meta-Llama-3.1-8B-Instruct"
OURS = "xmarva/tfs-transformer-en-ru"


def config(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: two configs side by side

def hero() -> Figure:
    ours, llama = config(OURS), config(LLAMA)
    rows = [
        ("layers", f"{ours['n_layers']} + {ours['n_layers']}", f"{llama['num_hidden_layers']}", False),
        ("width", f"{ours['d_model']}", f"{llama['hidden_size']:,}", False),
        ("attention heads", f"{ours['n_heads']}",
         f"{llama['num_attention_heads']}, sharing {llama['num_key_value_heads']} key/value heads", True),
        ("normalization", "LayerNorm, after each block", "RMSNorm, before each block", True),
        ("feed-forward", f"ReLU, {ours['d_ff']:,} wide", f"SwiGLU, {llama['intermediate_size']:,} wide", True),
        ("positions", "sinusoids added to the input", f"RoPE, rotation by position, base {llama['rope_theta']:,.0f}",
         True),
        ("biases", "in every layer", "none", True),
        ("vocabulary", f"{ours['vocab_size']:,}", f"{llama['vocab_size']:,}", False),
    ]
    fig = Figure(
        760,
        96 + len(rows) * 36 + 40,
        alt="Our 2017-style Transformer next to Llama 3.1 8B, from their config files: Llama is far bigger, and "
        "five design details differ: grouped key/value heads, RMSNorm before each block instead of LayerNorm after "
        "it, SwiGLU instead of ReLU, rotary position embeddings instead of added sinusoids, and no biases",
    )
    fig.text(14, 28, "Our Transformer and Llama 3.1 8B, read from their config files", kind="title")
    fig.text(14, 50, "highlighted: what changed in design, not in size", kind="note")
    fig.text(180, 82, "ours (the 2017 design)", kind="subtitle")
    fig.text(430, 82, "Llama 3.1 8B (2024)", kind="subtitle")
    for k, (name, a, b, design) in enumerate(rows):
        y = 96 + k * 36
        fig.text(14, y + 20, name, kind="label")
        fig.box(180, y, 236, 28, role="neutral" if not design else "plain", label=None)
        fig.text(190, y + 19, a, kind="body")
        fig.box(430, y, 316, 28, role="att" if design else "neutral", label=None)
        fig.text(440, y + 19, b, kind="body")
    fig.text(14, 96 + len(rows) * 36 + 22, "Llama keeps only the decoder; we keep the encoder too, and change both.",
             kind="note")
    return fig


# ---------------------------------------------------------------- the two blocks

def _block(fig, x, title, steps, note, skips):
    fig.text(x, 76, title, kind="subtitle")
    y = 96
    for k, (label, role) in enumerate(steps):
        fig.box(x, y, 300, 34, role=role, label=label)
        if k < len(steps) - 1:
            fig.arrow((x + 150, y + 35), (x + 150, y + 48))
        y += 50
    for k, (start, end) in enumerate(skips):  # the residual connections, on the right
        side = x + 312 + 10 * k
        y0, y1 = 96 + start * 50 + 17, 96 + end * 50 + 17
        fig.arrow((x + 302, y0), (side, y0), (side, y1), (x + 304, y1), role="data")
    fig.text(x, y + 6, note, kind="label")


def blocks() -> Figure:
    fig = Figure(
        760,
        470,
        alt="One encoder block in 2017 and in Llama's style. 2017: self-attention, add the input, LayerNorm, "
        "feed-forward with ReLU, add, LayerNorm: the normalization sits on the main path. Llama: RMSNorm, "
        "self-attention with rotary positions, add the input; RMSNorm, SwiGLU feed-forward, add: the main path "
        "from input to output carries only additions",
    )
    fig.text(14, 28, "One block, two arrangements", kind="title")
    _block(fig, 14, "2017: normalize after adding", [
        ("x", "data"),
        ("self-attention", "att"),
        ("add x", "neutral"),
        ("LayerNorm", "dec"),
        ("feed-forward, ReLU", "enc"),
        ("add", "neutral"),
        ("LayerNorm", "dec"),
    ], "every path to the output goes through a norm", [(0, 2), (3, 5)])
    _block(fig, 404, "Llama: normalize before, then add", [
        ("x", "data"),
        ("RMSNorm", "dec"),
        ("self-attention, RoPE on q and k", "att"),
        ("add x", "neutral"),
        ("RMSNorm", "dec"),
        ("feed-forward, SwiGLU", "enc"),
        ("add", "neutral"),
    ], "x reaches the output through additions only", [(0, 3), (3, 6)])
    return fig


# ---------------------------------------------------------------- LayerNorm and RMSNorm on one vector

VECTOR = [2.6, 3.1, 1.2, 2.9, 4.0, 1.8, 3.3, 2.4]  # a vector with a large common offset


def norms() -> Figure:
    n = len(VECTOR)
    mean = sum(VECTOR) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in VECTOR) / n)
    rms = math.sqrt(sum(v * v for v in VECTOR) / n)
    layer = [(v - mean) / std for v in VECTOR]
    rmsn = [v / rms for v in VECTOR]
    fig = Figure(
        760,
        320,
        alt=f"LayerNorm and RMSNorm on the same eight numbers. LayerNorm subtracts their mean, {mean:.2f}, and "
        f"divides by their standard deviation, {std:.2f}: the result is centred on zero. RMSNorm only divides by "
        f"the root mean square, {rms:.2f}: the pattern keeps its offset and only its scale changes",
    )
    fig.text(14, 28, "LayerNorm centres and scales; RMSNorm only scales", kind="title")
    panels = [("the input", VECTOR, "data"), (f"LayerNorm: (x − {mean:.2f}) / {std:.2f}", layer, "dec"),
              (f"RMSNorm: x / {rms:.2f}", rmsn, "enc")]
    top = max(max(abs(v) for v in values) for _, values, _ in panels)
    for k, (title, values, role) in enumerate(panels):
        x0 = 14 + k * 250
        fig.text(x0, 70, title, kind="label")
        zero = 190
        fig.line((x0, zero), (x0 + 230, zero), role="muted", width=1)
        for i, v in enumerate(values):
            h = 100 * v / top
            y = zero - h if h > 0 else zero
            fig.rect(x0 + 6 + i * 28, y, 20, abs(h), role=role, r=2)
        mean_here = sum(values) / n
        shown = "0.00" if abs(mean_here) < 0.005 else f"{mean_here:+.2f}".replace("-", "−")
        fig.text(x0, 262, f"mean {shown}", kind="small")
    fig.text(14, 300, "Same order and the same shape; RMSNorm skips the subtraction and the bias.", kind="note")
    return fig


# ---------------------------------------------------------------- SwiGLU

def swiglu() -> Figure:
    fig = Figure(
        760,
        300,
        alt="The feed-forward layer, two ways. 2017: x times W1, ReLU, times W2. SwiGLU: x goes through two "
        "matrices; one result passes SiLU and becomes a gate, multiplied element by element with the other, "
        "and the product goes through the third matrix. Three matrices instead of two, so each is two thirds as "
        "wide to keep the number of weights",
    )
    fig.text(14, 28, "Feed-forward: ReLU in 2017, a gated unit in Llama", kind="title")
    fig.text(14, 76, "2017", kind="subtitle")
    fig.box(14, 90, 70, 36, role="data", label="x")
    fig.box(120, 90, 130, 36, role="enc", label="× W₁")
    fig.box(286, 90, 100, 36, role="neutral", label="ReLU")
    fig.box(422, 90, 130, 36, role="enc", label="× W₂")
    for x0, x1 in [(86, 116), (252, 282), (388, 418)]:
        fig.arrow((x0, 108), (x1, 108))
    fig.text(570, 112, "2 matrices, d × 4d", kind="label")
    fig.text(14, 168, "SwiGLU", kind="subtitle")
    fig.box(14, 214, 70, 36, role="data", label="x")
    fig.box(120, 184, 130, 36, role="enc", label="× W_gate")
    fig.box(120, 244, 130, 36, role="enc", label="× W_up")
    fig.box(286, 184, 100, 36, role="neutral", label="SiLU")
    fig.box(422, 214, 60, 36, role="att", label="⊙")
    fig.box(518, 214, 130, 36, role="enc", label="× W_down")
    fig.arrow((86, 226), (116, 204))
    fig.arrow((86, 238), (116, 260))
    fig.arrow((252, 202), (282, 202))
    fig.arrow((388, 204), (418, 226))
    fig.arrow((252, 262), (418, 240))
    fig.arrow((484, 232), (514, 232))
    fig.text(286, 176, "the gate", kind="label")
    fig.text(560, 278, "3 matrices, d × ⅔·4d", kind="label")
    return fig


# ---------------------------------------------------------------- RoPE: rotation by position

def rope() -> Figure:
    theta = 0.5  # radians per position, one pair of dimensions
    q_angle, k_angle = math.radians(20), math.radians(70)
    fig = Figure(
        760,
        340,
        alt="Rotary position embedding on one pair of dimensions. The query of the token at position m is "
        "rotated by m times theta, the key of the token at position n by n times theta. Their dot product depends "
        "on the angle between them, which changes by (n − m) times theta: only the distance between the tokens "
        "counts, not where the pair sits in the sentence",
    )
    fig.text(14, 28, "RoPE turns position into an angle", kind="title")
    fig.text(14, 50, "one pair of dimensions of a query q and a key k; θ = 0.5 radian per position", kind="note")
    cases = [(0, 2), (3, 5), (6, 8)]
    for c, (m, n) in enumerate(cases):
        cx, cy, radius = 130 + c * 250, 190, 80
        fig.line((cx - radius - 10, cy), (cx + radius + 10, cy), role="muted", width=1)
        fig.line((cx, cy - radius - 10), (cx, cy + radius + 10), role="muted", width=1)
        qa = q_angle + m * theta
        ka = k_angle + n * theta
        q_end = (cx + radius * math.cos(qa), cy - radius * math.sin(qa))
        k_end = (cx + radius * 0.8 * math.cos(ka), cy - radius * 0.8 * math.sin(ka))
        fig.arrow((cx, cy), q_end, role="dec", width=2.2)
        fig.arrow((cx, cy), k_end, role="enc", width=2.2)
        fig.text(q_end[0] + 8 * math.cos(qa), q_end[1] - 8 * math.sin(qa) + 4, "q", kind="mono_dec",
                 anchor="middle")
        fig.text(k_end[0] + 10 * math.cos(ka), k_end[1] - 10 * math.sin(ka) + 4, "k", kind="mono",
                 role="enc", anchor="middle")
        fig.text(cx - 80, 92, f"q at {m}, k at {n}", kind="mono")
        between = math.degrees((ka - qa) % (2 * math.pi))
        fig.text(cx - 80, 300, f"angle between: {between:.0f}°", kind="label")
    fig.text(14, 326, "Shift both tokens along the sentence and the angle between them stays: attention sees n − m.",
             kind="note")
    return fig


# ---------------------------------------------------------------- the ablation ladder

LADDER = [("2017", "lesson 2's design"), ("+ pre-norm", "norm before blocks"), ("+ RMSNorm", "not LayerNorm"),
          ("+ SwiGLU", "not ReLU"), ("+ RoPE", "not sinusoids")]


def ladder() -> Figure:
    fig = Figure(
        760,
        260,
        alt="The ablation ladder: five models, each one change away from the previous one: the 2017 design, then "
        "pre-norm, then RMSNorm, then SwiGLU, then RoPE. All are trained on the same data with the same budget and "
        "measured the same way; the 2017 design is trained twice, with two seeds, to see how much the scores move "
        "by chance alone",
    )
    fig.text(14, 28, "One change at a time, the same budget for every model", kind="title")
    for k, (name, what) in enumerate(LADDER):
        x = 14 + k * 150
        y = 160 - k * 22
        fig.box(x, y, 136, 40, role="att" if k == len(LADDER) - 1 else "plain", label=name)
        fig.text(x, y + 58, what, kind="small")
        if k:
            fig.arrow((x - 12, y + 30), (x - 2, y + 22))
    fig.text(14, 248, "plus the 2017 design with another seed: the noise floor", kind="note")
    return fig


# ---------------------------------------------------------------- the KV cache

def kv_cache() -> Figure:
    fig = Figure(
        760,
        300,
        alt="Decoding the fifth token. Without a cache the decoder recomputes keys and values for all five "
        "positions at every step. With a cache, the keys and values of the first four positions are kept from the "
        "earlier steps, and only the new token's are computed",
    )
    fig.text(14, 28, "Writing token 5: recompute everything, or keep what you had", kind="title")
    for r, (title, cached) in enumerate([("without a cache", 0), ("with a KV cache", 4)]):
        y = 80 + r * 110
        fig.text(14, y + 22, title, kind="label")
        for i in range(5):
            x = 170 + i * 74
            is_new = i >= cached
            fig.box(x, y, 64, 34, role="att" if is_new else "neutral", label=f"k,v {i + 1}", strong=is_new)
        fig.text(560, y + 22, f"computed now: {5 - cached}", kind="count_hot" if cached else "count")
    fig.text(14, 280, "Every step adds one row; without the cache, step t redoes t rows.", kind="note")
    return fig


# ---------------------------------------------------------------- covers

def _cover(lang):
    english = lang == "en"
    ours, llama = config(OURS), config(LLAMA)
    pairs = [
        ("LayerNorm, after", "RMSNorm, before"),
        ("ReLU", "SwiGLU"),
        ("sinusoids", f"RoPE, base {llama['rope_theta']:,.0f}"),
        ("biases", "no biases"),
    ] if english else [
        ("LayerNorm после", "RMSNorm до"),
        ("ReLU", "SwiGLU"),
        ("синусоиды", f"RoPE, база {llama['rope_theta']:,.0f}".replace(",", " ")),
        ("смещения", "без смещений"),
    ]
    out = [cv.label(cv.MOTIF_X0, 128, "2017" if english else "2017"),
           cv.label(cv.MOTIF_X0 + 270, 128, "LLAMA 3.1" if english else "LLAMA 3.1")]
    for k, (old, new) in enumerate(pairs):
        y = 150 + k * 66
        chip, w = cv.chip(cv.MOTIF_X0, y, old, size=17)
        out.append(chip)
        out.append(cv.text(cv.MOTIF_X0 + 236, y + 27, "→", 22, cv.MUTED))
        chip, w2 = cv.chip(cv.MOTIF_X0 + 270, y, new, size=17, kind="hot" if k < 3 else "warm")
        cv.fits(cv.MOTIF_X0 + 270 + w2)
        assert cv.MOTIF_X0 + w < cv.MOTIF_X0 + 228, old
        out.append(chip)
    layers = f"{ours['n_layers']}+{ours['n_layers']} layers, {ours['d_model']} wide"
    out.append(cv.label(cv.MOTIF_X0, 440, ("OUR MODEL: " + layers).upper() if english else
                        f"НАША МОДЕЛЬ: {ours['n_layers']}+{ours['n_layers']} СЛОЯ"))
    title = ["What Llama", "changed"] if english else ["Что изменила", "Llama"]
    hook = ["Which of the changes", "help a small translator"] if english else ["Какие из изменений", "помогут маленькому переводчику"]
    return cv.frame("05_llama.ipynb", title, hook, "".join(out))


FIGURES = {
    "hero": hero,
    "blocks": blocks,
    "norms": norms,
    "swiglu": swiglu,
    "rope": rope,
    "ladder": ladder,
    "kv-cache": kv_cache,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
