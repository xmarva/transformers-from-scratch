"""Static figures and cover of the second course's lesson 10, Mamba and hybrids."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

MAMBA = "state-spaces/mamba-130m-hf"
TOKENS = ["Amundsen", "reached", "the", "Pole", "with", "fifty", "-two", "dogs"]


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: a cache that grows, a state that doesn't

HERO_ALT = ("The same eight tokens read by attention and by Mamba. Attention keeps a key and a value for every token, so "
            "its cache grows by one slot per token and never shrinks. Mamba writes every token into one state of fixed "
            "size, 1,536 channels of 16 numbers per layer in mamba-130m, and the state is the same size after eight "
            "tokens or eight million")


def hero() -> Figure:
    c = config(MAMBA)
    fig = Figure(760, 372, alt=HERO_ALT)
    fig.text(14, 28, "Attention keeps every token; Mamba keeps a state", kind="title")
    fig.text(14, 50, "the same eight tokens, read by each", kind="note")
    x0 = 120
    xs = []
    x = x0
    for t in TOKENS:
        xs.append(x)
        x += fig.chip_width(t) + 8
    fig.text(14, 98, "attention", kind="subtitle")
    for t, x in zip(TOKENS, xs):
        fig.chip(x, 76, t, role="data")
    for k, x in enumerate(xs):
        w = fig.chip_width(TOKENS[k])
        fig.rect(x, 116, w, 10, role="enc", tint=False)
        fig.rect(x, 130, w, 10, role="dec", tint=False)
    fig.text(xs[-1] + fig.chip_width(TOKENS[-1]) + 12, 136, "+ 1 key, 1 value", kind="small")
    fig.text(x0, 164, "the cache: one key and one value per token, per layer, kept to the end", kind="note")
    fig.text(14, 226, "Mamba", kind="subtitle")
    for t, x in zip(TOKENS, xs):
        fig.chip(x, 204, t, role="data")
    state_x, state_y, state_w = 300, 270, 210
    for k, x in enumerate(xs):
        cx = x + fig.chip_width(TOKENS[k]) / 2
        fig.arrow((cx, 236), (state_x + state_w * (k + 0.5) / len(xs), state_y - 2), role="att", width=1.2)
    fig.box(state_x, state_y, state_w, 40, role="att", label=None, strong=True)
    fig.text(state_x + state_w / 2, state_y + 25, f"state: {c['intermediate_size']:,} × {c['state_size']}",
             kind="mono", anchor="middle")
    fig.text(state_x + state_w + 14, state_y + 18, "rewritten at every token,", kind="small")
    fig.text(state_x + state_w + 14, state_y + 34, "the same size forever", kind="small")
    fig.text(14, 352, "per layer of mamba-130m; a token is read, folded into the state, and forgotten", kind="note")
    return fig


# ---------------------------------------------------------------- the Mamba block

BLOCK_ALT = ("The Mamba block of mamba-130m with the shape of every tensor for L tokens. The input, L × 768, is widened by "
             "in_proj to L × 3,072 and split into a stream u and a gate z, L × 1,536 each. The stream goes through a "
             "causal convolution of width 4 and SiLU. x_proj turns it into a step seed of 48 numbers and B and C, 16 "
             "numbers each, per token; dt_proj and softplus give a step Δ for each of the 1,536 channels. The selective "
             "scan runs 1,536 channels with a state of 16 each, using A (1,536 × 16) and the skip D. Its output is "
             "multiplied by SiLU of the gate and out_proj narrows it back to L × 768")


def block() -> Figure:
    c = config(MAMBA)
    d, e, n, r, k = (c["hidden_size"], c["intermediate_size"], c["state_size"], c["time_step_rank"],
                     c["conv_kernel"])
    fig = Figure(760, 470, alt=BLOCK_ALT)
    fig.text(14, 28, "One Mamba block of mamba-130m, with every shape", kind="title")
    fig.text(14, 50, f"L tokens; D = {d}, E = {e:,}, N = {n}, R = {r}", kind="note")
    fig.box(260, 66, 200, 32, role="data", label=f"x   L × {d}")
    fig.arrow((360, 99), (360, 117))
    fig.box(260, 118, 200, 32, role="neutral", label=f"in_proj → L × {2 * e:,}")
    fig.arrow((330, 151), (200, 177))
    fig.arrow((390, 151), (560, 177))
    fig.text(196, 172, "u", kind="mono", anchor="end")
    fig.text(566, 172, "z, the gate", kind="mono")
    fig.box(110, 178, 180, 32, role="neutral", label=f"conv1d, width {k}, SiLU")
    fig.arrow((200, 211), (200, 229))
    fig.box(110, 230, 180, 32, role="neutral", label=f"x_proj → {r} + {n} + {n}")
    fig.arrow((150, 263), (150, 283))
    fig.box(20, 284, 240, 32, role="neutral", label=f"dt_proj, softplus → Δ  L × {e:,}")
    fig.arrow((250, 263), (330, 335))
    fig.text(318, 300, f"B, C   L × {n}", kind="mono")
    fig.arrow((140, 317), (250, 336))
    fig.box(210, 337, 260, 40, role="att", label=None, strong=True)
    fig.text(340, 362, f"selective scan   A {e:,} × {n}, D", kind="mono", anchor="middle")
    fig.line((109, 220), (8, 220), (8, 357), (209, 357), dash=True)
    fig.text(16, 380, "u also enters the scan", kind="small")
    fig.box(520, 284, 140, 32, role="neutral", label="SiLU")
    fig.arrow((590, 211), (590, 283))
    fig.line((560, 178), (590, 178), (590, 211))
    fig.arrow((471, 357), (560, 357))
    fig.arrow((590, 317), (590, 340))
    fig.box(560, 341, 60, 32, role="plain", label="⊗")
    fig.arrow((590, 374), (590, 398))
    fig.box(500, 399, 180, 32, role="neutral", label=f"out_proj → L × {d}")
    fig.text(14, 456, "no attention and no MLP: 24 of these blocks, with RMSNorm and a residual around each, are the model",
             kind="note")
    return fig


# ---------------------------------------------------------------- the parallel scan

SCAN_ALT = ("The Hillis–Steele scan on eight positions. In the first round every position combines with its left "
            "neighbour, in the second with the position two to its left, in the third with the position four to its "
            "left; after three rounds, the logarithm of eight, every position holds the combined effect of all the "
            "steps before it, and each round is one elementwise operation over the whole sequence")


def scan() -> Figure:
    fig = Figure(760, 330, alt=SCAN_ALT)
    fig.text(14, 28, "The parallel scan: log₂ L rounds instead of L steps", kind="title")
    fig.text(14, 50, "each step is a pair (ā, w); two steps compose into one: (ā₂ā₁, ā₂w₁ + w₂)", kind="note")
    x0, step, w, h = 150, 74, 56, 28
    rows = [("steps", 78), ("round 1, shift 1", 140), ("round 2, shift 2", 202), ("round 3, shift 4", 264)]
    for k, (label, y) in enumerate(rows):
        fig.text(14, y + 19, label, kind="label")
        for i in range(8):
            first = max(0, i - 2 ** k + 1)
            name = f"{i + 1}" if first == i else f"{first + 1}–{i + 1}"
            role = "att" if (k == 3 or first == 0) and k > 0 else "plain"
            fig.box(x0 + i * step, y, w, h, role=role if k > 0 else "data", label=name)
        if k > 0:
            shift = 2 ** (k - 1)
            for i in range(shift, 8):
                fig.arrow((x0 + (i - shift) * step + w / 2, rows[k - 1][1] + h),
                          (x0 + i * step + w / 2 - 6, y), role="muted", width=1.0)
    fig.text(14, 318, "violet: positions that already hold everything from the start", kind="note")
    return fig


# ---------------------------------------------------------------- passport

PASSPORT_ALT = ("Mamba's passport. Published December 2023 by Albert Gu of Carnegie Mellon University and Tri Dao of "
                "Princeton University, code and weights under Apache 2.0. Type: a selective state space model, with no "
                "attention and no MLP blocks. Sizes: 130 million to 2.8 billion parameters. Training: 300 billion "
                "tokens of the Pile, context 2,048. State: per layer, 1,536 channels of 16 numbers in mamba-130m, plus "
                "a window of three inputs. Positions: none; order comes from the recurrence. Generation: the same "
                "memory and time for every token. Good at: long sequences at linear cost, fast generation, DNA and "
                "audio. Not for: recalling long spans exactly; pair it with attention")


def passport() -> Figure:
    c = config(MAMBA)
    rows = [
        ("published", "December 2023, Albert Gu (CMU) and Tri Dao (Princeton); Apache 2.0"),
        ("type", "selective state space model; no attention, no MLP"),
        ("sizes", "130M to 2.8B parameters"),
        ("training", "300B tokens of the Pile, context 2,048"),
        ("state", f"per layer {c['intermediate_size']:,} × {c['state_size']} in mamba-130m, plus "
                  f"{c['conv_kernel'] - 1} inputs"),
        ("positions", "none: order comes from the recurrence"),
        ("generation", "the same memory and time for every token"),
        ("good at", "long sequences at linear cost; DNA, audio"),
        ("not for", "recalling long spans exactly; pair it with attention"),
    ]
    return arch.passport("Mamba", rows, alt=PASSPORT_ALT)


# ---------------------------------------------------------------- cover

def _cover():
    c = config(MAMBA)
    out = [cv.label(cv.MOTIF_X0, 150, "ONE STATE, REWRITTEN AT EVERY TOKEN")]
    cols, rows, cell, gap = 16, 8, 26, 6
    for r in range(rows):
        for col in range(cols):
            x = cv.MOTIF_X0 + col * (cell + gap)
            y = 176 + r * (cell + gap)
            fade = (col + r * 3) % 7
            color = cv.RED if fade == 0 else cv.LINE
            fill = cv.RED if fade == 0 else "none"
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="{fill}" '
                       f'stroke="{color}" stroke-width="1.4"/>')
    cv.fits(cv.MOTIF_X0 + cols * (cell + gap))
    out.append(cv.label(cv.MOTIF_X0, 470, f"{c['intermediate_size']:,} CHANNELS × {c['state_size']} PER LAYER, "
                                          "AT ANY LENGTH"))
    return cv.frame("a10_mamba.ipynb", ["Beyond attention"], ["Mamba, state space models", "and hybrids"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "block": block,
    "scan": scan,
    "passport": passport,
    "family": lambda: arch.family(10),
}
COVERS = {"en": _cover}
