"""Static figures and cover of the second course's lesson 9, long context."""

import json
import math

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
LLAMA31 = "unsloth/Meta-Llama-3.1-8B"  # a public copy of Meta's config


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


def pair_turns(c):
    """Turns each rotary pair makes over the trained context, fastest pair first."""
    head = c["hidden_size"] // c["num_attention_heads"]
    trained = c["max_position_embeddings"]
    return [trained * c.get("rope_theta", 10000.0) ** (-2 * i / head) / (2 * math.pi) for i in range(head // 2)], trained


def yarn_bands(c, fast=32, slow=1):
    """The pairs YaRN's code leaves alone, blends and stretches (the floor and ceiling are its own)."""
    head = c["hidden_size"] // c["num_attention_heads"]
    trained = c["max_position_embeddings"]
    base = c.get("rope_theta", 10000.0)  # TinyLlama leaves it out: Llama's default

    def pair_making(turns):
        return head * math.log(trained / (turns * 2 * math.pi)) / (2 * math.log(base))

    first = max(math.floor(pair_making(fast)), 0)
    last = min(math.ceil(pair_making(slow)), head - 1)
    return ["keep" if i <= first else "stretch" if i >= last else "blend" for i in range(head // 2)]


def arc_points(cx, cy, r, fraction, steps=48):
    n = max(2, round(steps * fraction))
    return [(cx + r * math.sin(2 * math.pi * fraction * k / (n - 1)),
             cy - r * math.cos(2 * math.pi * fraction * k / (n - 1))) for k in range(n)]


def turns_label(t):
    return f"{t:.0f}" if t >= 10 else f"{t:.1f}" if t >= 1 else f"{t:.2f}"


# ---------------------------------------------------------------- hero: the clocks

HERO_ALT = ("TinyLlama's 32 rotary pairs drawn as dials, fastest first, each showing how far the pair turned over the "
            "2,048 positions of training. The fast pairs on the left turned hundreds of times, so every angle is "
            "familiar to the model. The slow pairs on the right never completed one turn: past position 2,048 they "
            "point in directions the model has never seen. Colors mark what YaRN does to each pair when it stretches "
            "the context eight times: the fast pairs are left alone, the slowest are stretched in full, and those in "
            "between are blended")

ROLE = {"keep": "enc", "blend": "att", "stretch": "dec"}


def hero() -> Figure:
    c = config(TINYLLAMA)
    turns, trained = pair_turns(c)
    bands = yarn_bands(c)
    fig = Figure(760, 330, alt=HERO_ALT)
    fig.text(14, 28, f"TinyLlama's {len(turns)} rotary pairs over its {trained:,} training positions", kind="title")
    fig.text(14, 50, "each dial is one pair of dimensions; the arc is how far it turned during training", kind="note")
    r, step = 15, 46.5
    for i, t in enumerate(turns):
        row, col = divmod(i, 16)
        cx, cy = 30 + col * step, 100 + row * 96
        fig.line(*arc_points(cx, cy, r, 1.0), role="neutral", width=1.2)
        fig.line(*arc_points(cx, cy, r, min(t, 1.0)), role=ROLE[bands[i]], width=3.6)
        fig.line((cx, cy), arc_points(cx, cy, r - 3, t % 1.0 if t >= 1 else t)[-1], role="neutral", width=1.4)
        fig.text(cx, cy + r + 18, turns_label(t), kind="small", anchor="middle")
    fig.text(14, 270, "number under a dial: turns in training", kind="small")
    never = sum(t < 1 for t in turns)
    fig.text(746, 270, f"{never} slowest pairs never finished a turn", kind="small", anchor="end")
    x = 14
    for band, words in [("keep", "YaRN leaves it alone"), ("blend", "YaRN blends"), ("stretch", "YaRN stretches 8×")]:
        fig.rect(x, 292, 22, 6, role=ROLE[band])
        fig.text(x + 30, 299, words, kind="note")
        x += 210
    return fig


# ---------------------------------------------------------------- FlashAttention's tiles

FLASH_ALT = ("How FlashAttention walks the attention matrix of a causal model. The queries are cut into blocks of rows "
             "and the keys and values into blocks of columns. For one block of queries the kernel loads one block of "
             "keys and values at a time into the GPU's fast on-chip memory, computes that tile of scores, folds it into "
             "a running maximum, a running sum and a running output, and moves on; the full matrix of scores never "
             "exists. Tiles above the diagonal are skipped, because a causal model's queries never look ahead")


def flash() -> Figure:
    fig = Figure(760, 360, alt=FLASH_ALT)
    fig.text(14, 28, "FlashAttention never builds the matrix of scores", kind="title")
    fig.text(14, 50, "one block of queries against one block of keys at a time, in on-chip memory", kind="note")
    n, cell, x0, y0 = 6, 34, 70, 86
    fig.text(x0 + n * cell / 2, y0 - 10, "keys and values, in blocks", kind="small", anchor="middle")
    fig.text(x0 - 12, y0 + 4, "queries, in blocks", kind="small", anchor="end", upright=True)
    current = 3
    for row in range(n):
        for col in range(n):
            x, y = x0 + col * cell, y0 + row * cell
            if col > row:
                fig.rect(x + 2, y + 2, cell - 4, cell - 4, role="neutral", tint=True)
            elif row == current and col < current:
                fig.rect(x + 2, y + 2, cell - 4, cell - 4, role="enc", tint=True)
            elif row == current and col == current:
                fig.rect(x + 2, y + 2, cell - 4, cell - 4, role="att")
            else:
                fig.rect(x + 2, y + 2, cell - 4, cell - 4, role="data", tint=True)
    fig.text(x0 + n * cell + 10, y0 + current * cell + 22, "← the block in work", kind="small")
    key = [("enc", True, "folded in"), ("att", False, "in work"), ("data", True, "other rows"),
           ("neutral", True, "never read")]
    for k, (role, tint, words) in enumerate(key):
        kx, ky = x0 - 30 + (k % 2) * 150, y0 + n * cell + 18 + (k // 2) * 22
        fig.rect(kx, ky, 12, 12, role=role, tint=tint)
        fig.text(kx + 20, ky + 10, words, kind="small")
    bx = 410
    fig.text(bx, 96, "for each tile of keys and values:", kind="subtitle")
    steps = [
        ("scores = Q·Kᵀ / √d", "a 128 × 128 tile, not n × n"),
        ("m = max(m, row max)", "the running maximum of each row"),
        ("ℓ = ℓ·e^(m_old − m) + Σ e^(s − m)", "the running sum, rescaled"),
        ("O = O·e^(m_old − m) + e^(s − m)·V", "the running output, rescaled"),
    ]
    for k, (formula, words) in enumerate(steps):
        y = 116 + k * 50
        fig.box(bx, y, 336, 26, role="att" if k == 0 else "plain", label=None)
        fig.text(bx + 10, y + 18, formula, kind="mono")
        fig.text(bx + 10, y + 42, words, kind="small")
    fig.text(bx, 336, "after the last tile, O / ℓ = softmax(QKᵀ/√d)·V", kind="note")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("Llama 3.1 8B's passport. Published July 2024 by Meta under the Llama 3.1 Community License. Type: "
                "decoder-only, 8 billion parameters. Attention: 32 query heads sharing 8 key-value heads. Positions: "
                "rotary with base 500,000, stretched 8 times from the 8,192 tokens of pretraining by Meta's own "
                "frequency-dependent scaling. Context: 131,072 tokens. Vocabulary: 128,256 tokens. Good at: long "
                "documents at a size one GPU can serve. Not for: tasks that need the whole window to be read reliably")


def passport() -> Figure:
    c = config(LLAMA31)
    s = c["rope_scaling"]
    rows = [
        ("published", "July 2024, Meta; Llama 3.1 Community License"),
        ("type", "decoder-only, 8B parameters"),
        ("attention", f"{c['num_attention_heads']} query heads sharing {c['num_key_value_heads']} key-value heads"),
        ("positions", f"rotary, base {c['rope_theta']:,.0f}"),
        ("stretch", f"{s['rope_type']}: ×{s['factor']:.0f} from {s['original_max_position_embeddings']:,} tokens, "
                    f"by frequency"),
        ("context", f"{c['max_position_embeddings']:,} tokens"),
        ("vocabulary", f"{c['vocab_size']:,} tokens"),
        ("good at", "long documents at a size one GPU can serve"),
        ("not for", "trusting that every part of the window gets read"),
    ]
    return arch.passport("Llama 3.1 8B", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    c = config(TINYLLAMA)
    turns, trained = pair_turns(c)
    out = [cv.label(cv.MOTIF_X0, 150, f"TURNS OF EACH ROTARY PAIR IN {trained:,} TOKENS")]
    r, step = 22, 64
    for i, t in enumerate(turns):
        row, col = divmod(i, 8)
        cx, cy = cv.MOTIF_X0 + r + col * step, 200 + row * 76
        ring = " ".join(f"{x:.1f},{y:.1f}" for x, y in arc_points(cx, cy, r, 1.0))
        out.append(f'<polyline points="{ring}" fill="none" stroke="{cv.LINE}" stroke-width="1.4"/>')
        arc = " ".join(f"{x:.1f},{y:.1f}" for x, y in arc_points(cx, cy, r, min(t, 1.0)))
        color = cv.RED if t < 1 else cv.MUTED
        out.append(f'<polyline points="{arc}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
    cv.fits(cv.MOTIF_X0 + 2 * r + 7 * step)
    never = sum(t < 1 for t in turns)
    out.append(cv.label(cv.MOTIF_X0, 520, f"RED: THE {never} PAIRS THAT NEVER FINISHED A TURN"))
    return cv.frame("a09_long.ipynb", ["Long context"], ["RoPE scaling, YaRN", "and FlashAttention"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "flash": flash,
    "passport": passport,
    "family": lambda: arch.family(9),
}
COVERS = {"en": _cover}
