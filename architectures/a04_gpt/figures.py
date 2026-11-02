"""Static figures and cover of the second course's lesson 4, GPT."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

GPT2 = "openai-community/gpt2"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: a game record in, a board inside

HERO_ALT = ("A chess GPT reads a game record one character at a time, a semicolon, 1, a dot, e, 4, a space, e, 5 "
            "and so on, and predicts the next character. Nothing in the input says where the pieces stand. A linear "
            "probe, one matrix, reads the residual stream after a middle layer and recovers what stands on each of "
            "the 64 squares")


def hero() -> Figure:
    fig = Figure(760, 326, alt=HERO_ALT)
    fig.text(14, 28, "A game record goes in; is there a board inside?", kind="title")
    fig.text(14, 50, "the model only ever predicts the next character; a linear probe reads its residual stream",
             kind="note")
    chars = [";", "1", ".", "e", "4", "␣", "e", "5", "␣", "2", "."]
    spans = fig.row(chars, 14, 84, role="data", size=13, h=30)
    right = spans[-1][0] + spans[-1][1]
    fig.text(14, 134, "the input: 32 possible characters, no board", kind="note")
    fig.arrow((right / 2 + 7, 147), (right / 2 + 7, 163))
    fig.box(14, 164, right - 14, 44, role="dec", label="GPT: 8 decoder layers")
    fig.arrow((right / 2 + 7, 209), (right / 2 + 7, 225))
    fig.chip(right / 2 - 8, 226, "N", role="dec", strong=True)
    fig.text(14, 282, "the output: the next character, here", kind="note")
    fig.text(14, 300, "the first letter of White's move", kind="note")

    fig.arrow((right + 1, 186), (435, 186), role="att")
    fig.box(436, 164, 116, 44, role="att", label="linear probe")
    fig.arrow((553, 186), (583, 186), role="att")
    board = ["rnbqkbnr", "pppp.ppp", "........", "....p...", "....P...", "........", "PPPP.PPP", "RNBQKBNR"]
    x0, y0, cell = 584, 114, 20
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            fig.rect(x0 + c * cell, y0 + r * cell, cell, cell, role="neutral", tint=(r + c) % 2 == 0, r=0)
            if piece != ".":
                kind = "mono" if piece.isupper() else "small"
                fig.text(x0 + c * cell + cell / 2, y0 + r * cell + 14, piece, kind=kind, anchor="middle")
    fig.text(x0, y0 + 8 * cell + 24, "what stands on each", kind="note")
    fig.text(x0, y0 + 8 * cell + 42, "of the 64 squares", kind="note")
    return fig


# ---------------------------------------------------------------- the objective: shifted targets, causal mask

WORDS = ["The", "game", "ended", "in", "a", "draw"]


def objective() -> Figure:
    fig = Figure(
        760,
        380,
        alt="Next-token prediction. The input is the text, the targets are the same text shifted one position left. "
        "The causal mask is a lower-triangular grid: each position attends to itself and every earlier position, "
        "never to later ones, so one forward pass makes a prediction at every position, each from the left "
        "context only",
    )
    fig.text(14, 28, "Next-token prediction: the text is its own label", kind="title")
    fig.text(14, 50, "one pass over the text trains a prediction at every position", kind="note")
    inputs, targets = WORDS[:-1], WORDS[1:]
    fig.text(14, 112, "input", kind="label")
    fig.text(14, 232, "target", kind="label")
    x, step = 84, 64
    for k in range(len(inputs)):
        left = x + k * step
        fig.box(left, 92, 56, 30, role="data", label=inputs[k])
        fig.box(left, 212, 56, 30, role="dec", label=targets[k], strong=k == len(inputs) - 1)
        fig.arrow((left + 28, 123), (left + 28, 211))
    fig.text(84, 280, "the model at position t sees tokens 1…t", kind="note")
    fig.text(84, 300, "and is scored on token t + 1", kind="note")

    gx, gy, cell = 456, 92, 40
    fig.text(gx, 80, "causal mask: who sees whom", kind="subtitle")
    for r in range(len(inputs)):
        fig.text(gx - 8, gy + r * cell + 20, inputs[r], kind="small", anchor="end")
        for c in range(len(inputs)):
            fig.rect(gx + c * cell + 1, gy + r * cell + 1, cell - 2, cell - 2, role="att", tint=c > r, r=3)
    for c in range(len(inputs)):
        fig.text(gx + c * cell + cell / 2, gy + len(inputs) * cell + 16, inputs[c], kind="small", anchor="middle")
    fig.text(gx, gy + len(inputs) * cell + 44, "a row predicts, its columns are what it", kind="note")
    fig.text(gx, gy + len(inputs) * cell + 62, "may look at; pale cells are masked out", kind="note")
    return fig


# ---------------------------------------------------------------- post-norm against pre-norm

def prenorm() -> Figure:
    fig = Figure(
        760,
        470,
        alt="Post-norm against pre-norm. In BERT's post-norm layer the LayerNorm sits on the main path, after each "
        "addition, so every layer's output is renormalised. In GPT-2's pre-norm layer the LayerNorm sits on the "
        "branch, before attention and before the feed-forward network; the main path, the residual stream, runs "
        "from the embeddings to the final LayerNorm with nothing on it but additions",
    )
    fig.text(14, 28, "Where the LayerNorm sits: BERT against GPT-2", kind="title")
    fig.text(14, 50, "pre-norm leaves the residual stream with nothing on it but additions", kind="note")

    fig.text(14, 84, "BERT: post-norm", kind="subtitle")
    steps = [("x", "data"), ("self-attention", "att"), ("add x", "neutral"), ("LayerNorm", "dec"),
             ("feed-forward", "enc"), ("add", "neutral"), ("LayerNorm", "dec")]
    y = 98
    for k, (label, role) in enumerate(steps):
        fig.box(14, y, 250, 32, role=role, label=label)
        if k < len(steps) - 1:
            fig.arrow((139, y + 33), (139, y + 45))
        y += 46
    for k, (start, end) in enumerate([(0, 2), (3, 5)]):
        side = 282 + 12 * k
        y0, y1 = 98 + start * 46 + 16, 98 + end * 46 + 16
        fig.arrow((266, y0), (side, y0), (side, y1), (268, y1), role="data")
    fig.text(14, 444, "every output passes through a LayerNorm", kind="note")

    sx = 420  # the residual stream of the pre-norm layer
    fig.text(380, 84, "GPT-2: pre-norm", kind="subtitle")
    fig.box(380, 98, 366, 32, role="data", label="x: the residual stream")
    fig.line((sx, 131), (sx, 380), role="data", width=2.4)
    for k, (first, second) in enumerate([("LayerNorm", "self-attention, causal"), ("LayerNorm", "feed-forward")]):
        top = 148 + k * 112
        fig.arrow((sx, top + 15), (479, top + 15))
        fig.box(480, top, 266, 30, role="dec", label=first)
        fig.arrow((613, top + 31), (613, top + 41))
        fig.box(480, top + 42, 266, 30, role="att" if k == 0 else "enc", label=second)
        plus = top + 86
        fig.arrow((613, top + 73), (613, plus), (sx + 15, plus))
        fig.box(sx - 14, plus - 14, 28, 28, role="neutral", label="+")
    fig.arrow((sx, 381), (sx, 393))
    fig.box(380, 394, 366, 32, role="dec", label="final LayerNorm, after the last layer")
    fig.text(380, 444, "only additions on the stream; one norm at the end", kind="note")
    return fig


# ---------------------------------------------------------------- the KV cache

CACHE_WORDS = ["The", "game", "ended", "in", "a"]


def kvcache() -> Figure:
    fig = Figure(
        760,
        330,
        alt="Generation without and with a KV cache. Without it, every step runs the whole sequence through the "
        "model again, so step t costs work proportional to t and the steps get slower. With it, every layer stores "
        "the keys and values of the tokens seen so far; each step runs only the new token, computes its query, key "
        "and value, appends the key and value to the cache and attends over it",
    )
    fig.text(14, 28, "Generating without and with a KV cache", kind="title")
    fig.text(14, 50, "highlighted: the tokens the model runs at each step", kind="note")
    columns = [(14, "without a cache", "the whole text, every step"), (400, "with a KV cache",
                                                                       "only the new token")]
    for x, name, note in columns:
        fig.text(x, 86, name, kind="subtitle")
        fig.text(x, 104, note, kind="note")
    for step in range(1, len(CACHE_WORDS)):
        y = 118 + (step - 1) * 44
        tokens = CACHE_WORDS[:step + 1]
        fig.row(tokens, 14, y, role="dec", strong=lambda i, t: True, size=12, h=28)
        fig.row(tokens, 400, y, role="dec", strong=lambda i, t, s=step: i == s, size=12, h=28)
    fig.text(14, 312, "step t reruns t tokens: total work grows as t²", kind="note")
    fig.text(400, 296, "earlier tokens' keys and values are read", kind="note")
    fig.text(400, 314, "from the cache, never recomputed", kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(GPT2)
    rows = [
        ("published", "February 2019, OpenAI; the full model only in November 2019"),
        ("type", "decoder only: generates text, one token at a time"),
        ("pre-training", "next-token prediction on WebText, 40 GB of web pages"),
        ("attention", f"causal, full; {c['n_head']} heads per layer in the smallest model"),
        ("positions", f"learned, at most {c['n_positions']:,} tokens"),
        ("normalisation", "LayerNorm before each sublayer (pre-norm), one more at the end"),
        ("activation", f"GELU, tanh approximation; feed-forward {4 * c['n_embd']:,} wide"),
        ("vocabulary", f"byte-level BPE, {c['vocab_size']:,} tokens; output tied to the input"),
        ("sizes", "124M (12 layers), 355M (24), 774M (36), 1.5B (48)"),
        ("good at", "a laboratory animal: interpretability, teaching, small baselines"),
        ("not for", "facts, instructions, long inputs: anything a modern LLM does"),
    ]
    return arch.passport(
        "GPT-2", rows,
        alt="GPT-2's passport. Published February 2019 by OpenAI, the full model released in November 2019. Type: "
        "decoder only. Pre-training: next-token prediction on WebText, 40 GB of web pages. Attention: causal, 12 "
        "heads per layer in the smallest model. Positions: learned, up to 1,024 tokens. Normalisation: pre-norm "
        "LayerNorm and a final one. Activation: GELU, tanh approximation. Vocabulary: byte-level BPE, 50,257 "
        "tokens, output tied to the input embeddings. Sizes: 124 million, 355 million, 774 million and 1.5 billion "
        "parameters. Good at: interpretability research, teaching, small baselines. Not for: facts, instructions, "
        "long inputs",
    )


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "A GAME, READ AS TEXT")]
    svg, spans, right = cv.chip_row([";1.e4", "e5", "2.Nf3", "N"], cv.MOTIF_X0, 172, size=19,
                                    kinds={3: "hot"})
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 268, "AND INSIDE THE MODEL, A BOARD"))
    board = chess_board_after(["e4", "e5", "Nf3"])
    size, x0, y0 = 26, cv.MOTIF_X0, 288
    for r in range(8):
        for c in range(8):
            fill = "#26272C" if (r + c) % 2 == 1 else "#3A3C43"  # dark and light squares
            out.append(f'<rect x="{x0 + c * size}" y="{y0 + r * size}" width="{size}" height="{size}" '
                       f'fill="{fill}"/>')
            piece = board[r][c]
            if piece != ".":
                color = cv.INK if piece.isupper() else cv.MUTED  # White in capitals, Black in small letters
                out.append(cv.text(x0 + c * size + size / 2, y0 + r * size + size * 0.72, piece, 15,
                                   color, cv.MONO, weight=600, anchor="middle"))
    for col, row, width in [(1, 0, 1.4), (2, 2, 2.6)]:  # the knight on b8 and the square it goes to, c6
        out.append(f'<rect x="{x0 + col * size + 1}" y="{y0 + row * size + 1}" width="{size - 2}" '
                   f'height="{size - 2}" fill="none" stroke="{cv.RED}" stroke-width="{width}"/>')
    note = x0 + 8 * size + 28
    out.append(cv.text(note, y0 + 3 * size + 12, "to write a legal knight move,", 18, cv.MUTED, cv.SANS))
    out.append(cv.text(note, y0 + 3 * size + 38, "it has to know where", 18, cv.MUTED, cv.SANS))
    out.append(cv.text(note, y0 + 3 * size + 64, "the knights stand", 18, cv.MUTED, cv.SANS))
    cv.fits(note + 250)
    return cv.frame("a04_gpt.ipynb", ["GPT from", "scratch"], ["A model that learns", "chess from text"],
                    "".join(out), footer=arch.COURSE)


def chess_board_after(moves):
    rows = [list("rnbqkbnr"), list("pppppppp")] + [list("........") for _ in range(4)] + \
        [list("PPPPPPPP"), list("RNBQKBNR")]
    squares = {"e4": ((6, 4), (4, 4)), "e5": ((1, 4), (3, 4)), "Nf3": ((7, 6), (5, 5))}
    for move in moves:
        (r0, c0), (r1, c1) = squares[move]
        rows[r1][c1], rows[r0][c0] = rows[r0][c0], "."
    return rows


FIGURES = {
    "hero": hero,
    "objective": objective,
    "prenorm": prenorm,
    "kvcache": kvcache,
    "passport": passport,
    "family": lambda: arch.family(4),
}
COVERS = {"en": _cover}
