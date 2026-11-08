"""Static figures and cover of the second course's lesson 6, modern LLMs."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

QWEN = "Qwen/Qwen3-0.6B"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: where each change acts

HERO_ALT = ("The decoder block of a modern LLM, with each change since Llama marked where it acts. Grouped-query "
            "attention shares one key-value head among several query heads and shrinks the KV cache. A sliding "
            "window limits how far back a local layer can look. QK-norm normalises queries and keys before their "
            "dot product, and soft-capping squashes the scores with a tanh. The rest of the block is Llama's: "
            "RMSNorm before each part, rotary positions, a SwiGLU feed-forward network")


def hero() -> Figure:
    fig = Figure(760, 352, alt=HERO_ALT)
    fig.text(14, 28, "Where the changes since Llama act in a decoder block", kind="title")
    fig.text(14, 50, "two of them save memory while the model generates, two keep training stable", kind="note")
    steps = ["projections", "QK-norm", "RoPE", "q·k / √d", "soft-cap", "mask", "softmax · V"]
    knobs = {0, 1, 4, 5}
    x, y, w, gap = 14, 84, 94, 10
    centers = []
    for k, step in enumerate(steps):
        fig.box(x, y, w, 40, role="att" if k in knobs else "neutral", label=step, strong=k in knobs)
        centers.append(x + w / 2)
        if k < len(steps) - 1:
            fig.arrow((x + w, y + 20), (x + w + gap, y + 20))
        x += w + gap
    cards = [
        (0, "grouped queries", ["fewer key-value heads", "a smaller KV cache", "nearly every model"],
         "memory", "enc"),
        (1, "QK-norm", ["RMSNorm on q and k", "scores stay bounded", "OLMo 2, Gemma 3, Qwen3"],
         "stability", "dec"),
        (4, "soft-cap", ["50 · tanh(s / 50)", "a ceiling on scores", "Gemma 2"], "stability", "dec"),
        (5, "sliding window", ["local layers see only", "the last W tokens", "Mistral, Gemma, gpt-oss"],
         "memory", "enc"),
    ]
    cw, cgap, top = 178, 6, 168
    for k, (step, title, lines, tag, role) in enumerate(cards):
        cx = 14 + k * (cw + cgap)
        fig.line((centers[step], y + 41), (centers[step], 146), (cx + cw / 2, 146), (cx + cw / 2, top - 1),
                 role="att")
        fig.box(cx, top, cw, 118, role="plain")
        fig.text(cx + 10, top + 24, title, kind="subtitle")
        for j, line in enumerate(lines):
            fig.text(cx + 10, top + 46 + j * 19, line, kind="note")
        fig.text(cx + 10, top + 106, tag, kind="label", role=role)
    fig.text(14, 316, "in most models the rest is Llama's: RMSNorm before attention and before the feed-forward",
             kind="note")
    fig.text(14, 336, "network, rotary positions, a SwiGLU feed-forward network, residual connections, almost no biases",
             kind="note")
    return fig


# ---------------------------------------------------------------- GPT-2's block against a modern one

def skeleton() -> Figure:
    fig = Figure(
        760,
        472,
        alt="Two decoder blocks side by side. GPT-2's, from lesson 4: LayerNorm, attention with learned position "
        "embeddings added to the input, LayerNorm, a GELU feed-forward network, biases everywhere. The modern "
        "block: RMSNorm, attention with rotary positions applied to queries and keys inside it, RMSNorm, a SwiGLU "
        "feed-forward network with three matrices, no biases. The places where this lesson's changes act are "
        "marked: the key-value heads, the mask, the queries and keys, the scores",
    )
    fig.text(14, 28, "The decoder block of GPT-2 and of a 2025 model", kind="title")
    fig.text(14, 50, "the same residual stream and causal mask; almost everything inside has been replaced",
             kind="note")
    left = [("tokens + learned position vectors", "data"), ("LayerNorm: mean, scale, bias", "neutral"),
            ("attention: a key and value per head", "att"), ("+ residual", "plain"),
            ("LayerNorm", "neutral"), ("feed-forward: GELU, 2 matrices", "enc"), ("+ residual", "plain"),
            ("final LayerNorm, tied output", "neutral")]
    right = [("tokens, no position vectors", "data"), ("RMSNorm: scale only", "neutral"),
             ("attention: RoPE, grouped queries", "att"), ("+ residual", "plain"),
             ("RMSNorm", "neutral"), ("feed-forward: SwiGLU, 3 matrices", "enc"), ("+ residual", "plain"),
             ("final RMSNorm, output", "neutral")]
    for x, title, rows in [(14, "GPT-2, 2019 (lesson 4)", left), (400, "Qwen3, 2025 (this lesson)", right)]:
        fig.text(x, 84, title, kind="subtitle")
        for k, (label, role) in enumerate(rows):
            y = 96 + k * 44
            fig.box(x, y, 300, 30, role=role, label=label, strong=role == "att" and x > 100)
            if k < len(rows) - 1:
                fig.arrow((x + 150, y + 31), (x + 150, y + 43))
        fig.text(x, 456, "biases in every layer" if x < 100 else "no biases; QK-norm inside attention",
                 kind="label", role=None if x < 100 else "att")
    fig.text(330, 199, "→", kind="lead", anchor="middle")
    fig.text(330, 331, "→", kind="lead", anchor="middle")
    return fig


# ---------------------------------------------------------------- MHA, GQA, MQA

def gqa() -> Figure:
    fig = Figure(
        760,
        330,
        alt="Three ways to give query heads their keys and values, with eight query heads. Multi-head attention: "
        "each query head has its own key-value head, eight in all. Grouped-query attention: four groups of two "
        "query heads, each group sharing one key-value head, four in all. Multi-query attention: all eight query "
        "heads share one key-value head. Below each, the KV cache per token, in proportion: eight, four and one "
        "units",
    )
    fig.text(14, 28, "Multi-head, grouped-query and multi-query attention with eight query heads", kind="title")
    fig.text(14, 50, "the cache stores only the key-value heads, so it shrinks with them", kind="note")
    panels = [("multi-head (MHA)", 8, "each query head has its own"),
              ("grouped-query (GQA)", 4, "two query heads per group"),
              ("multi-query (MQA)", 1, "all eight share one")]
    q_w, q_gap = 22, 4
    for p, (title, kv, note) in enumerate(panels):
        x0 = 14 + p * 250
        fig.text(x0, 86, title, kind="subtitle")
        fig.text(x0, 104, note, kind="note")
        q_centers = []
        for h in range(8):
            x = x0 + h * (q_w + q_gap)
            fig.box(x, 118, q_w, 26, role="att", label=None)
            q_centers.append(x + q_w / 2)
        fig.text(x0 + 210, 136, "q", kind="label", role="att")
        group = 8 // kv
        span = 8 * q_w + 7 * q_gap
        kv_w = (span - (kv - 1) * q_gap) / kv
        for g in range(kv):
            x = x0 + g * (kv_w + q_gap)
            fig.box(x, 196, kv_w, 26, role="dec", label=None)
            for h in range(g * group, (g + 1) * group):
                fig.line((q_centers[h], 145), (x + kv_w / 2, 195), role="muted", width=1.2)
        fig.text(x0 + 210, 214, "k, v", kind="label", role="dec")
        fig.text(x0, 256, "KV cache per token", kind="label")
        fig.rect(x0, 266, span * kv / 8, 16, role="dec")
        fig.text(x0, 304, f"{kv} key-value head{'s' if kv > 1 else ''}, {8 // kv}× smaller" if kv < 8
                 else "8 key-value heads", kind="note")
    return fig


# ---------------------------------------------------------------- sliding windows

def window() -> Figure:
    fig = Figure(
        760,
        402,
        alt="Sliding-window attention. Left: the causal mask of global attention, a full lower triangle, against "
        "the mask of a local layer with a window of four, a band along the diagonal. Middle: through three local "
        "layers, a token reaches four, then seven, then ten tokens back, because each layer reads the previous "
        "layer's summaries. Right: the rolling buffer, four slots, where token i is written into slot i mod 4 "
        "over the token that left the window",
    )
    fig.text(14, 28, "The mask, the reach and the cache of a window of four", kind="title")
    fig.text(14, 50, "each token sees itself and the three before it; information still travels further", kind="note")
    cell = 13
    for p, (title, w) in enumerate([("global", 99), ("local, W = 4", 4)]):
        x0, y0 = 14 + p * 130, 96
        fig.text(x0, 86, title, kind="subtitle")
        for i in range(8):
            for j in range(8):
                seen = j <= i and i - j < w
                fig.rect(x0 + j * cell, y0 + i * cell, cell - 2, cell - 2, role="att", tint=not seen)
        fig.text(x0, y0 + 8 * cell + 18, "rows: queries", kind="small")
        fig.text(x0, y0 + 8 * cell + 34, "columns: keys", kind="small")

    x0, y0 = 290, 96
    fig.text(x0, 86, "how far token 11 reaches", kind="subtitle")
    rows = [("layer 3", 11, 11), ("layer 2", 8, 11), ("layer 1", 5, 11), ("input", 2, 11)]
    for r, (name, lo, hi) in enumerate(rows):
        y = y0 + r * 26
        fig.text(x0, y + 15, name, kind="small")
        for t in range(12):
            on = lo <= t <= hi
            fig.rect(x0 + 62 + t * 15, y + 4, 13, 13, role="att", tint=not on)
        if r:
            fig.text(x0 + 62 + 12 * 15 + 6, y + 15, f"{hi - lo + 1} tokens", kind="small")
    fig.text(x0, y0 + 4 * 26 + 18, "W − 1 more tokens with every layer", kind="note")

    x0, y0 = 14, 268
    fig.text(x0, y0, "the rolling buffer puts token i into slot i mod 4", kind="subtitle")
    slots = [8, 9, 6, 7]
    for s, token in enumerate(slots):
        fig.box(x0 + s * 66, y0 + 14, 58, 34, role="data", label=f"t{token}")
        fig.text(x0 + s * 66 + 29, y0 + 66, f"slot {s}", kind="small", anchor="middle")
    fig.arrow((x0 + 2 * 66 + 29, y0 + 104), (x0 + 2 * 66 + 29, y0 + 76), role="dec")
    fig.text(x0 + 2 * 66 + 50, y0 + 104, "token 10 overwrites token 6, which has just left the window",
             kind="note", role="dec")
    fig.text(300, y0 + 26, "after token 9 the buffer holds tokens 6 to 9, out of order;", kind="note")
    fig.text(300, y0 + 46, "the order doesn't matter, positions are already rotated into", kind="note")
    fig.text(300, y0 + 66, "the keys, and the cache never grows past W", kind="note")
    return fig


# ---------------------------------------------------------------- LoRA

def lora() -> Figure:
    fig = Figure(
        760,
        300,
        alt="LoRA. The input x goes through the frozen weight matrix W, d_out by d_in, and in parallel through two "
        "small trained matrices: A, which maps it down to r numbers, and B, which maps them back up to d_out. The "
        "two outputs are added, the low-rank path scaled by alpha over r. B starts at zero, so at the start the "
        "model is unchanged. After training, B times A is added into W and the extra path disappears",
    )
    fig.text(14, 28, "LoRA adds a thin trained detour to a frozen matrix", kind="title")
    fig.text(14, 50, "W' = W + (α / r) · B A, with r far smaller than the matrix's sides", kind="note")
    fig.box(14, 132, 80, 40, role="data", label="x")
    fig.arrow((95, 152), (139, 110))
    fig.arrow((95, 152), (139, 210))
    fig.box(140, 80, 300, 60, role="neutral", label="W, frozen: d_out × d_in")
    fig.box(140, 190, 120, 40, role="enc", label="A: r × d_in", strong=True)
    fig.arrow((261, 210), (299, 210))
    fig.box(300, 190, 140, 40, role="enc", label="B: d_out × r", strong=True)
    fig.arrow((441, 210), (489, 210))
    fig.box(490, 190, 70, 40, role="plain", label="× α / r")
    fig.arrow((441, 110), (599, 140))
    fig.arrow((561, 210), (599, 166))
    fig.box(600, 132, 60, 40, role="plain", label="+")
    fig.arrow((661, 152), (689, 152))
    fig.box(690, 132, 56, 40, role="data", label="y")
    fig.text(140, 252, "A starts random and B at zero, so at step 0 the model is exactly the original", kind="note")
    fig.text(140, 274, "after training, B A is added into W once, and the detour costs nothing at inference",
             kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(QWEN)
    rows = [
        ("published", "April 2025, Alibaba's Qwen team; eight models, Apache 2.0"),
        ("type", "decoder-only; six dense models and two mixtures of experts"),
        ("training", "36 trillion tokens in 119 languages; thinking and non-thinking in one model"),
        ("attention", f"grouped queries: {c['num_attention_heads']} query heads, {c['num_key_value_heads']} "
                      f"key-value heads of {c['head_dim']} (0.6B); QK-norm"),
        ("positions", f"rotary, base {c['rope_theta']:,}; no sliding window"),
        ("normalisation", "pre-norm RMSNorm, and RMSNorm on every query and key head"),
        ("activation", f"SwiGLU, {c['intermediate_size']:,} wide in the 0.6B; no biases anywhere"),
        ("vocabulary", f"byte-level BPE, {c['vocab_size']:,} rows; small models tie input and output"),
        ("sizes", "0.6B, 1.7B, 4B, 8B, 14B, 32B dense; 30B-A3B and 235B-A22B experts"),
        ("good at", "running locally, narrow jobs after LoRA, reasoning in thinking mode"),
        ("not for", "facts it can't look up; the 0.6B has the knowledge of a 0.6B"),
    ]
    return arch.passport(
        "Qwen3", rows,
        alt="Qwen3's passport. Published April 2025 by Alibaba's Qwen team, eight models under Apache 2.0. Type: "
        "decoder-only, six dense models and two mixtures of experts. Training: 36 trillion tokens in 119 "
        "languages, thinking and non-thinking modes in one model. Attention: grouped queries, 16 query heads and 8 "
        "key-value heads of 128 in the 0.6B model, with QK-norm. Positions: rotary with a base of one million, no "
        "sliding window. Normalisation: pre-norm RMSNorm, and RMSNorm on every query and key head. Activation: "
        "SwiGLU, 3,072 wide in the 0.6B model, no biases. Vocabulary: byte-level BPE with 151,936 rows; small "
        "models tie input and output embeddings. Sizes: 0.6B, 1.7B, 4B, 8B, 14B and 32B dense, 30B-A3B and "
        "235B-A22B mixtures of experts. Good at: running locally, narrow jobs after LoRA, reasoning in thinking "
        "mode. Not for: facts it can't look up",
    )


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "A VOICE COMMAND")]
    svg, spans, right = cv.chip_row(["wake me up", "at five am", "this week"], cv.MOTIF_X0, 172, size=17)
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 262, "JSON FROM A 0.6B MODEL WITH LORA"))
    svg, spans, right = cv.chip_row(["alarm_set"], cv.MOTIF_X0, 284, size=17, kinds={0: "hot"})
    cv.fits(right)
    out.append(svg)
    svg, spans, right = cv.chip_row(["time: five am", "date: this week"], cv.MOTIF_X0, 330, size=17)
    cv.fits(right)
    out.append(svg)
    out.append(cv.text(cv.MOTIF_X0, 410, "16 query heads, 8 key-value heads, QK-norm", 18, cv.MUTED, cv.SANS))
    return cv.frame("a06_modern.ipynb", ["Modern LLMs"], ["GQA, sliding windows, QK-norm,", "and Qwen3 from scratch"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "skeleton": skeleton,
    "gqa": gqa,
    "window": window,
    "lora": lora,
    "passport": passport,
    "family": lambda: arch.family(6),
}
COVERS = {"en": _cover}
