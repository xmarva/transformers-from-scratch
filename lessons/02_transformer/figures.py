"""Static figures and covers of lesson 2.

Nothing here is typed in by hand when it can be read from a source: the real models' shapes come from their
config.json files on the Hugging Face Hub, our model's shape from the settings in lesson.py, and the tokens of the
"shifted right" figure from the course tokenizer of lesson 1.
"""

import json
import re
from pathlib import Path

from tools.figures import cover as cv
from tools.figures.lib import MONO_ADVANCE, Figure

LESSON = Path(__file__).with_name("lesson.py")

EN_EXAMPLE = "The researcher said she was exhausted."
RU_EXAMPLE = "Исследовательница сказала, что вымоталась."


def setting(name):
    """A number set in lesson.py, e.g. D_MODEL = 256, so a figure can't disagree with the lesson."""
    match = re.search(rf"^{name} = ([\d_]+)", LESSON.read_text(encoding="utf-8"), re.M)
    return int(match.group(1).replace("_", ""))


def _config(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json"))


# ---------------------------------------------------------------- hero: real models, to scale

BLUEPRINTS = [  # name, year, repository; ours comes from the lesson's settings
    ("opus-mt-en-ru", "2020", "Helsinki-NLP/opus-mt-en-ru"),
    ("GPT-2", "2019", "openai-community/gpt2"),
    ("Llama 3.1 8B", "2024", "unsloth/Meta-Llama-3.1-8B"),
    ("DeepSeek-V3", "2024", "deepseek-ai/DeepSeek-V3"),
]


def blueprints():
    """(name, year, d_model, encoder layers, decoder layers), ours first."""
    out = [("Ours", "lesson 2", setting("D_MODEL"), setting("N_LAYERS"), setting("N_LAYERS"))]
    for name, year, repo in BLUEPRINTS:
        config = _config(repo)
        d_model = config.get("hidden_size") or config.get("d_model") or config.get("n_embd")
        if "encoder_layers" in config:
            out.append((name, year, d_model, config["encoder_layers"], config["decoder_layers"]))
        else:
            out.append((name, year, d_model, 0, config.get("num_hidden_layers") or config.get("n_layer")))
    return out


def hero() -> Figure:
    models = blueprints()
    widest = max(d for _, _, d, _, _ in models)
    tallest = max(max(e, d) for _, _, _, e, d in models)
    scale, bar, gap_y = 236 / widest, 3.0, 1.2
    fig = Figure(
        760,
        420,
        alt="Five Transformers drawn to scale, width proportional to d_model and height to the number of layers: "
        + "; ".join(
            f"{name} {f'{e} + {d}' if e else d} layers × {w}" for name, _, w, e, d in models
        ),
    )
    fig.text(14, 28, "One block, stacked: four real models and ours, to scale", kind="title")
    fig.text(14, 50, "width: d_model · height: number of layers · teal encoder, rose decoder", kind="note")
    base = 72 + tallest * (bar + gap_y)
    x = 14
    for name, year, width, enc, dec in models:
        stack_w = width * scale
        total_w = 2 * stack_w + 4 if enc else stack_w
        shape = f"{enc}+{dec} × {width}" if enc else f"{dec} × {width}"
        label_w = max(len(name) * 14 * 0.56, len(shape) * 13 * MONO_ADVANCE)
        slot = max(total_w, label_w)
        x0 = x + (slot - total_w) / 2
        stacks = [("enc", enc, x0), ("dec", dec, x0 + stack_w + 4)] if enc else [("dec", dec, x0)]
        for role, layers, sx in stacks:
            for k in range(layers):
                fig.rect(sx, base - (k + 1) * (bar + gap_y) + gap_y, max(stack_w, 2), bar, role=role)
        center = x + slot / 2
        fig.text(center, base + 26, name, kind="subtitle", anchor="middle")
        fig.text(center, base + 46, shape, kind="count", anchor="middle")
        fig.text(center, base + 64, year, kind="small", anchor="middle")
        x += slot + 24
    assert x - 24 <= 746, f"hero is {x - 24:.0f} px wide"
    return fig


# ---------------------------------------------------------------- the architecture

def architecture() -> Figure:
    n, d, h, d_ff = setting("N_LAYERS"), setting("D_MODEL"), setting("N_HEADS"), setting("D_FF")
    fig = Figure(
        760,
        700,
        alt=f"The encoder–decoder Transformer. The encoder (left, {n} layers) embeds the English tokens, adds "
        "positional encodings and applies self-attention and a feed-forward network, each followed by add & "
        f"norm. The decoder (right, {n} layers) embeds the Russian tokens shifted right and applies masked "
        "self-attention, cross-attention to the encoder's output and a feed-forward network. The output is "
        "multiplied by the transposed embedding matrix and passed through softmax",
    )
    fig.text(14, 28, "The Transformer of 2017, at our size", kind="title")
    fig.text(14, 682, f"d_model = {d} · h = {h} heads · d_ff = {d_ff} · N = {n} layers per stack", kind="note")
    w, bh = 210, 30
    ex, dx = 90, 470  # left edges of the encoder and decoder columns
    ecx, dcx = ex + w / 2, dx + w / 2

    def stage(x, y, label, role="neutral"):
        fig.box(x, y, w, bh, role=role, label=label, r=7)

    def up(cx, y_from, y_to, role="muted"):
        fig.arrow((cx, y_from), (cx, y_to + 2), role=role)

    def residual(x, y_from, y_to):
        """The skip connection: branches off the main path below a sub-layer, goes around it into add & norm."""
        fig.arrow((x + w / 2, y_from), (x - 14, y_from), (x - 14, y_to + bh / 2), (x - 2, y_to + bh / 2))

    # inputs
    fig.chip(ecx - Figure.chip_width("English ids") / 2, 616, "English ids")
    fig.chip(dcx - Figure.chip_width("Russian ids, shifted right") / 2, 616, "Russian ids, shifted right")
    for x, cx in [(ex, ecx), (dx, dcx)]:
        stage(x, 560, "embedding · √d_model")
        stage(x, 512, "+ positional encoding")
        up(cx, 614, 590)
        up(cx, 560, 542)
    # encoder layer: container, then its sub-layers bottom to top
    fig.box(ex - 26, 236, w + 52, 250, role="enc", r=12)
    fig.text(ex - 22, 256, f"× {n}", kind="mono", role="enc")
    up(ecx, 512, 444)
    stage(ex, 414, "self-attention", role="att")
    stage(ex, 370, "add & norm")
    stage(ex, 312, "feed-forward")
    stage(ex, 268, "add & norm")
    up(ecx, 414, 400)
    up(ecx, 370, 342)
    up(ecx, 312, 298)
    residual(ex, 456, 370)
    residual(ex, 356, 268)
    # decoder layer
    fig.box(dx - 26, 110, w + 52, 376, role="dec", r=12)
    fig.text(dx + w + 22, 130, f"× {n}", kind="mono", role="dec", anchor="end")
    up(dcx, 512, 444)
    stage(dx, 414, "masked self-attention", role="att")
    stage(dx, 370, "add & norm")
    stage(dx, 312, "cross-attention", role="att")
    stage(dx, 268, "add & norm")
    stage(dx, 194, "feed-forward")
    stage(dx, 150, "add & norm")
    for y_from, y_to in [(414, 400), (370, 342), (312, 298), (268, 224), (194, 180)]:
        up(dcx, y_from, y_to)
    residual(dx, 456, 370)
    residual(dx, 356, 268)
    residual(dx, 238, 150)
    # the encoder's output, "memory", feeds every decoder layer's cross-attention
    fig.arrow((ecx, 268), (ecx, 200), (ex + w + 70, 200), (ex + w + 70, 327), (dx - 2, 327), role="att")
    fig.text(ex + w + 76, 192, "memory", kind="label")
    # output
    stage(dx, 66, "× Eᵀ: the same matrix E", role="data")
    up(dcx, 110, 96)
    fig.text(dcx, 46, "softmax → probabilities of the next token", kind="note", anchor="middle")
    up(dcx, 66, 52)
    return fig


# ---------------------------------------------------------------- why attention: agreement

def _word_spans(words, x, gap, size=13, pad=9):
    spans = []
    for word in words:
        w = len(word) * size * MONO_ADVANCE + 2 * pad
        spans.append((x, w))
        x += w + gap
    return spans


def agreement() -> Figure:
    en = ["The", "researcher", "said", "she", "was", "exhausted."]
    ru = [("Исследователь", "ница", ""), ("сказал", "а", ","), ("что", "", ""), ("вымотал", "а", "сь.")]
    she = en.index("she")
    fig = Figure(
        540,
        300,
        alt="To translate 'The researcher said she was exhausted.' into Russian, three words take feminine forms "
        "because of 'she': Исследовательница, сказала, вымоталась. The decoder writing each of them has to look "
        "back at 'she' in the English sentence",
    )
    fig.text(14, 28, "One “she”, three feminine forms", kind="title")
    fig.text(14, 82, "English", kind="label")
    fig.text(14, 232, "Russian", kind="label")
    en_spans = _word_spans(en, 110, 8)
    for i, ((x, w), word) in enumerate(zip(en_spans, en)):
        fig.box(x, 60, w, 32, role="att" if i == she else "plain", label=word, r=7)
        fig.els[-1].mono = True
        fig.els[-1].strong = i == she
    ru_words = ["".join(parts) for parts in ru]
    ru_spans = _word_spans(ru_words, 110, 10)
    for (x, w), (stem, marker, tail) in zip(ru_spans, ru):
        fig.box(x, 210, w, 32, role="plain", r=7)
        tx = x + 9
        fig.text(tx, 231, stem, kind="mono")
        if marker:
            fig.text(tx + len(stem) * 13 * MONO_ADVANCE, 231, marker, kind="mono_dec")
        if tail:
            fig.text(tx + (len(stem) + len(marker)) * 13 * MONO_ADVANCE, 231, tail, kind="mono")
    # lexical alignment, grey: researcher -> Исследовательница, said -> сказала, exhausted -> вымоталась
    for en_i, ru_i in [(1, 0), (2, 1), (5, 3)]:
        (ax, aw), (bx, bw) = en_spans[en_i], ru_spans[ru_i]
        fig.line((ax + aw / 2, 94), (bx + bw / 2, 208), role="muted", width=1.2, dash=True)
    # agreement: every feminine form depends on "she"
    sx = en_spans[she][0] + en_spans[she][1] / 2
    for ru_i in [0, 1, 3]:
        bx, bw = ru_spans[ru_i]
        fig.line((sx, 94), (bx + bw / 2, 208), role="att", width=2.2)
    fig.text(14, 280, "violet: “she” decides these endings · dashed: word alignment", kind="note")
    return fig


# ---------------------------------------------------------------- attention, step by step

def attention() -> Figure:
    fig = Figure(
        760,
        272,
        alt="Scaled dot-product attention: queries Q times transposed keys give a score for every pair of "
        "positions; the scores are divided by the square root of d_k, the mask is added, softmax turns every "
        "row into weights that sum to 1, and the weights mix the values V into the output",
    )
    fig.text(14, 28, "Attention, step by step", kind="title")
    cell = 16

    def matrix(x, y, rows, cols, label, shape, role="att", tint=True, shade=None):
        for r in range(rows):
            for c in range(cols):
                strong = shade is not None and shade(r, c)
                fig.rect(x + c * (cell + 2), y + r * (cell + 2), cell, cell, role=role, tint=tint and not strong, r=2)
        width = cols * (cell + 2) - 2
        fig.text(x + width / 2, y - 12, label, kind="mono_att", anchor="middle")
        fig.text(x + width / 2, y + rows * (cell + 2) + 16, shape, kind="small", anchor="middle")
        return x + width

    y = 96
    end = matrix(24, y, 4, 3, "Q", "(T_q, d_k)")
    fig.text(end + 14, y + 40, "×", kind="subtitle")
    end = matrix(end + 34, y + 18, 3, 4, "Kᵀ", "(d_k, T_k)")
    fig.arrow((end + 10, y + 36), (end + 40, y + 36))
    sx = end + 50
    end = matrix(sx, y, 4, 4, "scores", "(T_q, T_k)")
    fig.arrow((end + 10, y + 36), (end + 110, y + 36))
    fig.text(end + 60, y + 24, "÷ √d_k, + M", kind="mono", anchor="middle")
    fig.text(end + 60, y + 60, "softmax", kind="mono", anchor="middle")
    wx = end + 120
    end = matrix(wx, y, 4, 4, "weights", "rows sum to 1", shade=lambda r, c: c == min(r, 3) or (r, c) == (3, 1))
    fig.text(end + 14, y + 40, "×", kind="subtitle")
    end = matrix(end + 34, y, 4, 3, "V", "(T_k, d_v)")
    fig.arrow((end + 10, y + 36), (end + 40, y + 36))
    matrix(end + 50, y, 4, 3, "output", "(T_q, d_v)", role="data")
    fig.text(14, 236, "Each row of the output is a mix of the rows of V. The weights say how much of each to take, and", kind="note")
    fig.text(14, 256, "they come from how well that position's query matches every key.", kind="note")
    return fig


# ---------------------------------------------------------------- heads

def heads() -> Figure:
    d, h = setting("D_MODEL"), setting("N_HEADS")
    d_k = d // h
    fig = Figure(
        760,
        300,
        alt=f"Multi-head attention: the {d} numbers of every position are split into {h} heads of {d_k}; each "
        "head runs attention on its own slice, and the results are concatenated and mixed by W^O",
    )
    fig.text(14, 28, f"{h} heads: split, attend, concatenate", kind="title")
    x0, total = 40, 560
    seg = total / h
    fig.text(14, 54, "the queries after one projection x W^Q; keys and values are split the same way", kind="note")
    for i in range(h):
        fig.box(x0 + i * seg + 2, 66, seg - 4, 30, role="att", label=f"head {i + 1}: {d_k}", r=6)
        cx = x0 + i * seg + seg / 2
        fig.arrow((cx, 98), (cx, 126))
        fig.box(cx - 54, 128, 108, 30, role="neutral", label="attention", r=6)
        fig.arrow((cx, 160), (cx, 188))
    fig.text(x0 + total + 14, 86, f"(B, T, {d})", kind="small")
    fig.text(x0 + total + 14, 148, f"(B, {h}, T, {d_k})", kind="small")
    for i in range(h):
        fig.rect(x0 + i * seg + 2, 190, seg - 4, 26, role="att", tint=True, r=4)
    fig.text(x0 + total / 2, 208, "concatenate", kind="mono", anchor="middle")
    fig.text(x0 + total + 14, 208, f"(B, T, {d})", kind="small")
    fig.arrow((x0 + total / 2, 218), (x0 + total / 2, 236))
    fig.box(x0 + total / 2 - 70, 238, 140, 30, role="neutral", label="W^O: mix the heads", r=6)
    fig.text(14, 290, "One matrix multiplication computes all heads' queries at once; the split is only a reshape.", kind="note")
    return fig


# ---------------------------------------------------------------- shifted right

def shift_right() -> Figure:
    tokenizer = _tokenizer()
    target = tokenizer.encode(RU_EXAMPLE).tokens
    decoder_input = ["<bos>", *target]
    expected = [*target, "<eos>"]
    size, gap = 12, 4
    fig = Figure(
        760,
        236,
        alt="Teacher forcing: the decoder reads <bos> and the Russian tokens, and at every position it is trained "
        "to predict the next token, so the target is the same sequence shifted left by one, ending with <eos>",
    )
    fig.text(14, 28, "The decoder reads the translation shifted right", kind="title")
    fig.text(14, 58, "decoder reads", kind="label")
    fig.text(14, 146, "and predicts", kind="label")
    x0 = 14
    widths = [max(Figure.chip_width(a, size), Figure.chip_width(b, size)) for a, b in zip(decoder_input, expected)]
    x = x0
    for i, (a, b) in enumerate(zip(decoder_input, expected)):
        w = widths[i]
        fig.box(x, 66, w, 28, role="dec", label=a, r=6)
        fig.els[-1].mono, fig.els[-1].size = True, size
        fig.box(x, 154, w, 28, role="data", label=b, r=6)
        fig.els[-1].mono, fig.els[-1].size = True, size
        fig.arrow((x + w / 2, 96), (x + w / 2, 128))
        x += w + gap
    assert x <= 750, f"tokens reach x={x:.0f}"
    fig.text(14, 216, "At every position the decoder sees the tokens up to that point and must predict the next one.", kind="note")
    return fig


# ---------------------------------------------------------------- attention as a soft lookup

LOOKUP_QUERY = [1.0, 2.0]  # the worked example of section 4: one query, three keys
LOOKUP_KEYS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def _softmax_weights(query, keys):
    import math

    d_k = len(query)
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(d_k) for key in keys]
    exps = [math.exp(score) for score in scores]
    return scores, [e / sum(exps) for e in exps]


def lookup() -> Figure:
    scores, weights = _softmax_weights(LOOKUP_QUERY, LOOKUP_KEYS)
    fig = Figure(
        760,
        330,
        alt="A Python dictionary returns the value of the one key that matches the query exactly. Attention "
        "compares the query with every key, turns the scores into weights "
        + ", ".join(f"{w:.2f}" for w in weights)
        + " and returns the weighted mix of all the values",
    )
    fig.text(14, 28, "A dictionary lookup, and attention", kind="title")
    # left: a dict
    fig.text(14, 62, "dict: one exact match, one value", kind="subtitle")
    fig.text(14, 94, "query", kind="label")
    fig.chip(80, 76, '"she"', role="att", strong=True)
    keys = ['"researcher"', '"said"', '"she"']
    fig.text(14, 136, "key", kind="label")
    fig.text(200, 136, "value", kind="label")
    for i, key in enumerate(keys):
        y = 148 + i * 40
        hit = i == 2
        fig.chip(14, y, key, role="att" if hit else "plain", strong=False)
        fig.chip(200, y, f"value {i + 1}", role="data" if hit else "plain")
        fig.arrow((14 + Figure.chip_width(key) + 4, y + 15), (196, y + 15), role="att" if hit else "muted")
    fig.text(14, 300, "result: value 3, and nothing else", kind="note")
    # right: attention
    x0 = 390
    fig.text(x0, 62, "attention: every key matches a little", kind="subtitle")
    fig.text(x0, 94, "query", kind="label")
    fig.chip(x0 + 66, 76, "[1, 2]", role="att", strong=True)
    fig.text(x0, 136, "key", kind="label")
    fig.text(x0 + 86, 136, "q·k / √2", kind="label")
    fig.text(x0 + 170, 136, "weight", kind="label")
    fig.text(x0 + 300, 136, "value", kind="label")
    for i, (key, score, weight) in enumerate(zip(LOOKUP_KEYS, scores, weights)):
        y = 148 + i * 40
        fig.chip(x0, y, f"[{key[0]:g}, {key[1]:g}]", role="plain")
        fig.text(x0 + 86, y + 20, f"{score:.2f}", kind="count")
        fig.rect(x0 + 170, y + 6, 110 * weight, 18, role="att", r=3)
        fig.text(x0 + 176 + 110 * weight, y + 20, f"{weight:.2f}", kind="count")
        fig.chip(x0 + 300, y, f"v{i + 1}", role="data")
    mix = " + ".join(f"{w:.2f}·v{i + 1}" for i, w in enumerate(weights))
    fig.text(x0, 300, f"result: {mix}", kind="note")
    return fig


# ---------------------------------------------------------------- queries, keys and values

def _grid(fig, x, y, n_rows, n_cols, cell, role, hot_row=None, tint=True):
    for r in range(n_rows):
        for k in range(n_cols):
            fig.rect(x + k * cell, y + r * cell, cell - 2, cell - 2, role=role, tint=tint and r != hot_row, r=2)
    return x + n_cols * cell - 2, y + n_rows * cell - 2


def qkv() -> Figure:
    tokens = _tokenizer().encode(EN_EXAMPLE).tokens
    she = tokens.index("▁she")
    cell = 17
    fig = Figure(
        760,
        350,
        alt="Every token's vector x is multiplied by three learned matrices, W^Q, W^K and W^V, which gives its "
        "query, its key and its value. For the whole sentence the rows of X become the rows of Q, K and V",
    )
    fig.text(14, 28, "Three roles for every token", kind="title")
    top = 100
    for r, token in enumerate(tokens):
        fig.text(104, top + r * cell + 12, token, kind="small", anchor="end")
    x_right, bottom = _grid(fig, 112, top, len(tokens), 8, cell, "data", hot_row=she)
    fig.text(112, bottom + 22, "X: one row per token", kind="label")
    columns = [("Q", "W^Q", "what she looks for"), ("K", "W^K", "how she can be found"), ("V", "W^V", "what she hands over")]
    x_center = (112 + x_right) / 2
    centers = []
    for i, (name, matrix, role_text) in enumerate(columns):
        gx = 330 + i * 140
        gx_right, _ = _grid(fig, gx, top, len(tokens), 4, cell, "att", hot_row=she)
        fig.text(gx, bottom + 22, f"{name} = X {matrix}", kind="mono_att")
        fig.text(gx, bottom + 42, role_text, kind="small")
        centers.append((gx + gx_right) / 2)
    # one input, three projections: a bracket from X into Q, K and V
    fig.line((x_center, top - 6), (x_center, 66), (centers[-1], 66), role="muted", width=1.4)
    for cx in centers:
        fig.arrow((cx, 66), (cx, top - 6))
    fig.text(14, bottom + 74, "The same vector, three learned views of it. The highlighted row is the token she.",
             kind="note")
    return fig


# ---------------------------------------------------------------- self-attention and cross-attention

def self_cross() -> Figure:
    tokenizer = _tokenizer()
    en = tokenizer.encode(EN_EXAMPLE).tokens
    ru = ["<bos>", *tokenizer.encode(RU_EXAMPLE).tokens[:-1]]  # what the decoder reads
    she, said_ru = en.index("▁she"), ru.index("▁сказала")
    cell = 19
    fig = Figure(
        760,
        444,
        alt="Two grids of attention weights. Self-attention in the encoder: every English token, as a query, "
        "looks at every English token. Cross-attention in the decoder: every Russian token looks at every English "
        "token; the cell where the query of ▁сказала meets the key of ▁she is highlighted",
    )
    fig.text(14, 28, "One mechanism, two uses", kind="title")
    panels = [
        (96, en, en, "self-attention, in the encoder", "English looks at English", she, en.index("▁rese")),
        (480, ru, en, "cross-attention, in the decoder", "Russian looks at English", said_ru, she),
    ]
    for gx, rows, cols, title, subtitle, hot_row, hot_col in panels:
        fig.text(gx - 82, 56, title, kind="subtitle")
        fig.text(gx - 82, 76, subtitle, kind="note")
        top = 96
        for r, token in enumerate(rows):
            fig.text(gx - 8, top + r * cell + 13, token, kind="small", anchor="end")
            for k in range(len(cols)):
                strong = r == hot_row and k == hot_col
                role = "att" if r == hot_row else "neutral"
                fig.rect(gx + k * cell, top + r * cell, cell - 2, cell - 2, role=role, tint=not strong, r=2)
        bottom = top + len(rows) * cell
        for k, token in enumerate(cols):
            fig.text(gx + k * cell + 13, bottom + 4, token, kind="small", anchor="end", upright=True)
        fig.text(gx + len(cols) * cell + 8, top + 12, "keys →", kind="small")
        fig.text(gx + len(cols) * cell + 8, top + 30, "queries ↓", kind="small")
    fig.text(14, 408, "Each row is one query and the weights it will give to every key, the columns. The solid cell",
             kind="note")
    fig.text(14, 428, "is one query meeting one key: she and the first piece of researcher; сказала and she.",
             kind="note")
    return fig


# ---------------------------------------------------------------- add & norm

def add_norm() -> Figure:
    fig = Figure(
        760,
        210,
        alt="One sub-layer in the 2017 Transformer: the input x goes through the sub-layer (attention or the "
        "feed-forward network) and dropout, is added back to x by the residual connection, and the sum is "
        "normalized: LayerNorm(x + Dropout(Sublayer(x)))",
    )
    fig.text(14, 28, "Every sub-layer, wrapped the same way", kind="title")
    y, h = 112, 34
    fig.chip(20, y + 2, "x", role="data")
    fig.arrow((52, y + h / 2), (96, y + h / 2))
    fig.box(98, y, 200, h, role="att", label="sub-layer: attention or FFN", r=7)
    fig.arrow((300, y + h / 2), (330, y + h / 2))
    fig.box(332, y, 92, h, role="neutral", label="dropout", r=7)
    fig.arrow((426, y + h / 2), (456, y + h / 2))
    fig.box(458, y, 34, h, role="neutral", label="+", r=17)
    fig.arrow((494, y + h / 2), (524, y + h / 2))
    fig.box(526, y, 120, h, role="neutral", label="LayerNorm", r=7)
    fig.arrow((648, y + h / 2), (684, y + h / 2))
    fig.chip(686, y + 2, "out", role="data")
    fig.arrow((36, y), (36, 70), (475, 70), (475, y - 2), role="enc", width=2)
    fig.text(120, 62, "residual connection: x skips the sub-layer and is added back", kind="note")
    fig.text(14, 188, "LayerNorm(x + Dropout(Sublayer(x)))", kind="mono")
    return fig


# ---------------------------------------------------------------- covers

COVER = {
    "en": dict(
        tgt=[("Issledovatel'", "nitsa", ""), ("skazal", "a", ""), (",", "", ""), ("chto", "", ""), ("vymotal", "a", "s'")],
        size=15,
        tlabel="RUSSIAN, TRANSLITERATED",
        title=["The Transformer,", "built by hand"],
        hook=["One «she», three feminine", "forms in Russian"],
        legend="feminine marker",
        source="SOURCE",
    ),
    "ru": dict(
        tgt=[("Исследователь", "ница", ""), ("сказал", "а", ""), (",", "", ""), ("что", "", ""), ("вымотал", "а", "сь")],
        size=16,
        tlabel="ПЕРЕВОД",
        title=["Трансформер", "своими руками"],
        hook=["Одно «she» и три", "формы женского рода"],
        legend="показатель рода",
        source="ОРИГИНАЛ",
    ),
}


def _cover(lang):
    c = COVER[lang]
    src = ["The", "researcher", "said", "she", "was", "exhausted"]
    she, feminine = 3, [0, 1, 4]
    out = [cv.label(cv.MOTIF_X0, 128, c["source"])]
    srow, sb, send = cv.chip_row(src, cv.MOTIF_X0, 142, 16, gap=7, kinds={she: "warm"})
    tokens = [t if t[1] else t[0] for t in c["tgt"]]
    trow, tb, tend = cv.chip_row(tokens, cv.MOTIF_X0, 432, c["size"], gap=6)
    cv.fits(send)
    cv.fits(tend)
    # word alignment (grey): researcher, said, exhausted; then agreement with "she" (red), drawn on top
    for t, s_ in [(0, 1), (1, 2), (4, 5)]:
        x1, x2 = tb[t][0] + tb[t][1] / 2, sb[s_][0] + sb[s_][1] / 2
        out.append(f'<line x1="{x1}" y1="432" x2="{x2}" y2="182" stroke="#8A8E98" stroke-opacity="0.7" stroke-width="1.6"/>')
    sx = sb[she][0] + sb[she][1] / 2
    for t in feminine:
        tx = tb[t][0] + tb[t][1] / 2
        out.append(f'<line x1="{tx}" y1="432" x2="{sx}" y2="182" stroke="{cv.RED}" stroke-width="2.6"/>')
    out += [srow, trow, cv.label(cv.MOTIF_X0, 506, c["tlabel"])]
    legend_x = cv.MOTIF_X1 - cv.chip_w(c["legend"], 15, 0) - 18
    out.append(f'<rect x="{legend_x}" y="495" width="10" height="10" rx="2" fill="{cv.RED}"/>')
    out.append(cv.text(legend_x + 18, 505, c["legend"], 15, cv.RED, cv.MONO, 500))
    return cv.frame("02_transformer.ipynb", c["title"], c["hook"], "".join(out))


FIGURES = {
    "hero": hero,
    "architecture": architecture,
    "agreement": agreement,
    "lookup": lookup,
    "qkv": qkv,
    "attention": attention,
    "self-cross": self_cross,
    "heads": heads,
    "add-norm": add_norm,
    "shift-right": shift_right,
}
COVERS = {"en": lambda: _cover("en"), "ru": lambda: _cover("ru")}
