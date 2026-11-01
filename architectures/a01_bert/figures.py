"""Static figures and covers of the second course's lesson 1, BERT.

BERT's sizes are read at render time from its config.json on the Hugging Face Hub; the windows figure computes its
windows and their owners with the same rule as the lesson's code.
"""

import json

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

UNCASED, CASED = "google-bert/bert-base-uncased", "google-bert/bert-base-cased"
WINDOW, STRIDE = 510, 384  # as in the lesson


def config(repo):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: Google's query read two ways

def hero() -> Figure:
    fig = Figure(
        760,
        290,
        alt="The query 2019 brazil traveler to usa need a visa, read two ways. As a bag of keywords, brazil, "
        "traveler, usa and visa match pages for Americans going to Brazil. Read as a sentence, the word to links "
        "traveler and usa: a Brazilian going to the US",
    )
    fig.text(14, 28, "Google's example: one query, two readings", kind="title")
    fig.text(14, 50, "the search engine before and after BERT, October 2019", kind="note")
    query = ["2019", "brazil", "traveler", "to", "usa", "need", "a", "visa"]
    fig.row(query, 14, 66, role="neutral")
    fig.text(14, 132, "as keywords", kind="subtitle")
    fig.row(["brazil", "traveler", "usa", "visa"], 14, 146, role="data")
    fig.box(14, 196, 340, 34, role="dec", label="pages for U.S. citizens going to Brazil")
    fig.text(14, 252, "the little word to is dropped as noise", kind="note")
    fig.text(400, 132, "as a sentence", kind="subtitle")
    spans = fig.row(["brazil", "traveler", "to", "usa"], 400, 146, role="data",
                    strong=lambda i, t: t == "to")
    (tx, tw), (ux, uw) = spans[1], spans[3]
    fig.line((tx + tw / 2, 176), (tx + tw / 2, 184), (ux + uw / 2, 184), (ux + uw / 2, 176), role="att")
    fig.box(400, 196, 300, 34, role="enc", label="a Brazilian travelling to the U.S.")
    fig.text(400, 252, "to sets the direction of the trip", kind="note")
    return fig


# ---------------------------------------------------------------- three ways to learn from raw text

def _sentence(fig, y, label, note, visible, target_text="bank", target=1):
    words = ["the", "bank", "of", "the", "river", "was", "steep"]
    words[target] = target_text
    fig.text(14, y + 20, label, kind="subtitle")
    x = 150
    for i, word in enumerate(words):
        if i == target:
            role, strong = "att", True
        else:
            role, strong = (visible(i), False)
        w = fig.chip(x, y, word, role=role, strong=strong)
        x += w + 6
    fig.text(150, y + 50, note, kind="note")


def readers() -> Figure:
    fig = Figure(
        760,
        350,
        alt="Three ways to learn from raw text, predicting the word bank in the bank of the river was steep. GPT "
        "predicts each word from the words before it and sees only the. ELMo trains a left-to-right and a "
        "right-to-left model separately and joins their outputs. BERT hides the word behind a mask and predicts it "
        "from both sides at once, in every layer",
    )
    fig.text(14, 28, "Predicting one word from raw text, three ways", kind="title")
    fig.text(14, 50, "teal: what the model may use to predict the highlighted word", kind="note")
    _sentence(fig, 72, "GPT", "left to right only: the word that decides the meaning comes too late",
              lambda i: "enc" if i < 1 else "neutral")
    _sentence(fig, 152, "ELMo", "one model reads each way; the two halves meet only at the output",
              lambda i: "enc" if i < 1 else "dec")
    _sentence(fig, 232, "BERT", "the word is hidden, and every layer reads both sides of the gap",
              lambda i: "enc", target_text="[MASK]")
    fig.text(14, 330, "ELMo: teal is its left-to-right model, rose its right-to-left one", kind="note")
    return fig


# ---------------------------------------------------------------- what goes in

def inputs() -> Figure:
    tokens = ["[CLS]", "the", "bank", "raised", "rates", "[SEP]", "river", "bank", "[SEP]"]
    segments = ["A"] * 6 + ["B"] * 3
    fig = Figure(
        760,
        300,
        alt="BERT's input for two sentences: each token's vector is the sum of a word embedding, a learned "
        "position embedding for positions 0 to 511, and a segment embedding A or B, followed by LayerNorm",
    )
    fig.text(14, 28, "Each token goes in as the sum of three learned vectors", kind="title")
    fig.text(14, 50, "then LayerNorm and dropout; [CLS] opens the input, [SEP] closes each sentence", kind="note")
    labels = [("word", 72), ("+ position", 118), ("+ segment", 164)]
    for text, y in labels:
        fig.text(14, y + 20, text, kind="label")
    x = 120
    for i, token in enumerate(tokens):
        w = max(fig.chip_width(token), 52)
        fig.box(x, 72, w, 30, role="data", label=token)
        fig.box(x, 118, w, 30, role="att", label=str(i))
        fig.box(x, 164, w, 30, role="enc" if segments[i] == "A" else "dec", label=segments[i])
        x += w + 6
    fig.text(14, 232, "= input", kind="label")
    fig.box(120, 212, x - 126, 30, role="neutral", label="the sum, LayerNorm, dropout: what layer 1 reads")
    fig.text(120, 272, "each box is a learned vector of 768 numbers; position 0 to 511, segment A or B",
             kind="note")
    return fig


# ---------------------------------------------------------------- one layer

def layer() -> Figure:
    steps = [
        ("x, 768 numbers per token", "data"),
        ("self-attention, 12 heads, no causal mask", "att"),
        ("dropout, add x", "neutral"),
        ("LayerNorm", "dec"),
        ("feed-forward 768 → 3072 → 768, GELU", "enc"),
        ("dropout, add", "neutral"),
        ("LayerNorm", "dec"),
    ]
    fig = Figure(
        760,
        420,
        alt="One BERT layer: self-attention, dropout, add the input, LayerNorm; then a feed-forward network 768 to "
        "3072 to 768 with GELU, dropout, add, LayerNorm. The norm sits after each addition, on the main path",
    )
    fig.text(14, 28, "One of BERT's twelve layers", kind="title")
    fig.text(14, 50, "post-norm, as in the 2017 Transformer: LayerNorm after each addition", kind="note")
    y = 72
    for k, (label, role) in enumerate(steps):
        fig.box(14, y, 360, 34, role=role, label=label)
        if k < len(steps) - 1:
            fig.arrow((194, y + 35), (194, y + 48))
        y += 50
    for k, (start, end) in enumerate([(0, 2), (3, 5)]):
        side = 392 + 12 * k
        y0, y1 = 72 + start * 50 + 17, 72 + end * 50 + 17
        fig.arrow((376, y0), (side, y0), (side, y1), (378, y1), role="data")
    fig.text(440, 120, "the residual path: x is added back", kind="note")
    fig.text(440, 140, "after attention and after the", kind="note")
    fig.text(440, 160, "feed-forward network", kind="note")
    fig.text(440, 220, "GELU(x) = x · Φ(x): a ReLU", kind="note")
    fig.text(440, 240, "with a smooth corner", kind="note")
    fig.text(440, 300, "the same layer, twelve times;", kind="note")
    fig.text(440, 320, "every one has its own weights", kind="note")
    return fig


# ---------------------------------------------------------------- the 80/10/10 rule

def masking() -> Figure:
    fig = Figure(
        760,
        300,
        alt="The 80/10/10 rule. Of all tokens, 15% are picked for prediction. Of those, 80% are replaced with the "
        "MASK token, 10% with a random word, and 10% are left unchanged; the model must predict the original word "
        "at all of them",
    )
    fig.text(14, 28, "How BERT hides words: 15% picked, then 80 / 10 / 10", kind="title")
    fig.text(14, 50, "highlighted: the positions picked for prediction; the loss looks only at them", kind="note")
    words = ["the", "bank", "raised", "its", "rates", "again", "after", "the", "river", "flooded"]
    picked = {1: "[MASK]", 4: "piano", 8: "river"}
    fig.text(14, 92, "original", kind="label")
    fig.row(words, 110, 72, role="neutral")
    shown = [picked.get(i, w) for i, w in enumerate(words)]
    fig.text(14, 152, "input", kind="label")
    fig.row(shown, 110, 132, role="data", strong=lambda i, t: i in picked)
    rows = [
        ("80%", "[MASK]", "bank", "replaced with the mask token", "att"),
        ("10%", "piano", "rates", "replaced with a random word", "dec"),
        ("10%", "river", "river", "left as it is, but still predicted", "enc"),
    ]
    for k, (share, shown_as, answer, note, role) in enumerate(rows):
        y = 190 + k * 34
        fig.text(14, y + 19, share, kind="count_hot")
        fig.box(110, y, 90, 28, role=role, label=shown_as)
        fig.text(214, y + 19, f"→ predict {answer!r}: {note}", kind="body")
    return fig


# ---------------------------------------------------------------- windows over a long document

def windows_of(n):
    if n <= WINDOW:
        return [(0, n)]
    starts = list(range(0, n - WINDOW + 1, STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    return [(s, s + WINDOW) for s in starts]


def windows() -> Figure:
    n = 1100
    spans = windows_of(n)
    owner, depth = [0] * n, [-1] * n
    for k, (start, end) in enumerate(spans):
        for i in range(start, end):
            room = min(i - start, end - 1 - i)
            if room > depth[i]:
                owner[i], depth[i] = k, room
    fig = Figure(
        760,
        290,
        alt=f"A document of {n:,} tokens cut into {len(spans)} overlapping windows of {WINDOW} tokens with a stride "
        f"of {STRIDE}. Each token's label is taken from the window where it is furthest from the window's edge, so "
        "every token is judged with context on both sides",
    )
    fig.text(14, 28, f"A {n:,}-token document, read in windows of {WINDOW}", kind="title")
    fig.text(14, 50, f"a window starts every {STRIDE} tokens; a token keeps the prediction of the window "
             "it sits deepest in", kind="note")
    x0, scale = 20, 720 / n
    roles = ["enc", "att", "dec"]
    for k, (start, end) in enumerate(spans):
        y = 72 + k * 40
        fig.box(x0 + start * scale, y, (end - start) * scale, 30, role=roles[k % 3],
                label=f"window {k + 1}: tokens {start}–{end - 1}")
    y = 72 + len(spans) * 40 + 14
    fig.text(14, y, "whose prediction each token keeps", kind="label")
    runs, begin = [], 0
    for i in range(1, n + 1):
        if i == n or owner[i] != owner[begin]:
            runs.append((begin, i, owner[begin]))
            begin = i
    for start, end, k in runs:
        fig.rect(x0 + start * scale, y + 10, (end - start) * scale, 22, role=roles[k % 3], tint=True)
    for start, end, _ in runs:
        fig.text(x0 + (start + end) / 2 * scale, y + 56, f"{start}–{end - 1}", kind="small", anchor="middle")
    return fig


# ---------------------------------------------------------------- the passport and the family tree

def passport() -> Figure:
    base, cased = config(UNCASED), config(CASED)
    rows = [
        ("published", "October 2018, Google (Devlin, Chang, Lee, Toutanova)"),
        ("type", "encoder only: reads the whole input at once, generates nothing"),
        ("pre-training", "masked language model (15% of tokens) + next sentence prediction"),
        ("attention", f"bidirectional, full; {base['num_attention_heads']} heads per layer in base"),
        ("positions", f"learned, at most {base['max_position_embeddings']} tokens"),
        ("normalisation", "LayerNorm after each sublayer (post-norm)"),
        ("activation", f"{base['hidden_act'].upper()} in the feed-forward network, {base['intermediate_size']:,} wide"),
        ("vocabulary", f"WordPiece: {base['vocab_size']:,} uncased, {cased['vocab_size']:,} cased"),
        ("sizes", f"base: {base['num_hidden_layers']} layers, {base['hidden_size']} wide, 110M; "
                  "large: 24 layers, 1024 wide, 340M"),
        ("good at", "classification, tagging tokens, search: fast, small, runs on a CPU"),
        ("not for", "generating text, inputs over 512 tokens without windows"),
    ]
    return arch.passport(
        "BERT", rows,
        alt="BERT's passport. Published October 2018 by Google. Type: encoder. Pre-training objective: masked "
        "language modelling and next sentence prediction. Attention: bidirectional, full. Positions: learned, up to "
        "512. Normalisation: LayerNorm after each sublayer. Activation: GELU. Sizes: base 110 million parameters, 12 "
        "layers, width 768; large 340 million, 24 layers, width 1024. Best at: classification, tagging, retrieval",
    )


# ---------------------------------------------------------------- cover

def _cover():
    words = ["the", "bank", "of", "the", "[MASK]", "was", "steep"]
    svg, spans, right = cv.chip_row(words, cv.MOTIF_X0, 250, size=19, kinds={4: "hot"})
    cv.fits(right)
    out = [cv.label(cv.MOTIF_X0, 150, "MASKED LANGUAGE MODEL"), svg]
    mask_x, mask_w = spans[4]
    target = mask_x + mask_w / 2
    for k, (x, w) in enumerate(spans):
        if k == 4:
            continue
        mid = x + w / 2
        lift = 30 + 10 * abs(k - 4)
        out.append(f'<path d="M {mid:.1f} 248 Q {(mid + target) / 2:.1f} {248 - lift} {target:.1f} 246" '
                   f'fill="none" stroke="{cv.RED}" stroke-width="1.6" opacity="0.8"/>')
    out.append(cv.label(cv.MOTIF_X0, 340, "BOTH SIDES OF THE GAP, IN EVERY LAYER"))
    out.append(cv.text(cv.MOTIF_X0, 400, "bert-base: 12 layers, 110M parameters", 18, cv.MUTED, cv.MONO))
    return cv.frame("a01_bert.ipynb", ["BERT from", "scratch"], ["What a model learns", "by filling in blanks"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "readers": readers,
    "inputs": inputs,
    "layer": layer,
    "masking": masking,
    "windows": windows,
    "passport": passport,
    "family": lambda: arch.family(1),
}
COVERS = {"en": _cover}
