"""Static figures and cover of the second course's lesson 5, T5 and BART."""

import json

from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

T5_SMALL = "google-t5/t5-small"


def config(repo):
    with open(hf_hub_download(repo, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- hero: a question in, a query out, checked by running it

HERO_ALT = ("A question and the database's schema go into T5's encoder, which reads all of it at once. The decoder "
            "writes an SQL query one token at a time, looking back at the encoder's output through cross-attention. "
            "The query is then run on the SQLite database, and its result is compared with the result of the "
            "correct query")


def hero() -> Figure:
    fig = Figure(760, 336, alt=HERO_ALT)
    fig.text(14, 28, "A question goes in, a query comes out, and SQLite says if it is right", kind="title")
    fig.text(14, 50, "the encoder reads the question and the schema; the decoder writes the query", kind="note")
    fig.text(14, 86, "question", kind="subtitle")
    fig.box(14, 96, 330, 34, role="data", label="How many singers are younger than 30?")
    fig.text(14, 156, "schema", kind="subtitle")
    fig.box(14, 166, 330, 34, role="data", label="singer : Singer_ID , Name , Age , …")
    fig.arrow((345, 148), (377, 148))
    fig.box(378, 96, 150, 104, role="enc", label="encoder")
    fig.text(453, 88, "reads both ways", kind="note", anchor="middle")
    fig.arrow((529, 148), (561, 148), role="att")
    fig.box(562, 96, 184, 104, role="dec", label="decoder")
    fig.text(654, 88, "writes left to right", kind="note", anchor="middle")
    fig.text(14, 262, "query", kind="subtitle")
    fig.box(14, 272, 450, 34, role="dec", label="SELECT count(*) FROM singer WHERE Age < 30", strong=True)
    fig.arrow((654, 201), (654, 236), (240, 236), (240, 270))
    fig.arrow((465, 289), (497, 289))
    fig.box(498, 272, 118, 34, role="neutral", label="SQLite")
    fig.arrow((617, 289), (649, 289))
    fig.box(650, 272, 96, 34, role="enc", label="[(2,)] ✓")
    fig.text(498, 326, "the same rows as the correct query?", kind="note")
    return fig


# ---------------------------------------------------------------- the encoder–decoder

def encdec() -> Figure:
    fig = Figure(
        760,
        412,
        alt="The encoder–decoder. On the left the encoder: a stack of layers with self-attention that sees the whole "
        "input in both directions and a feed-forward network; it runs once and turns the input into a sequence of "
        "vectors. On the right the decoder: in every layer causal self-attention over what it has written so far, "
        "then cross-attention, whose queries come from the decoder and whose keys and values come from the "
        "encoder's output, then a feed-forward network. Only the decoder runs once per generated token",
    )
    fig.text(14, 28, "Two stacks and a bridge: the encoder–decoder", kind="title")
    fig.text(14, 50, "the encoder runs once; the decoder runs once per token it writes and reads the encoder "
             "in every layer", kind="note")
    fig.text(14, 86, "encoder × 6 layers (t5-small)", kind="subtitle")
    fig.box(14, 98, 300, 34, role="data", label="input: translate English to German: …")
    fig.arrow((164, 133), (164, 149))
    fig.box(14, 150, 300, 160, role="neutral")
    fig.box(30, 164, 268, 34, role="att", label="self-attention, both directions")
    fig.arrow((164, 199), (164, 213))
    fig.box(30, 214, 268, 34, role="enc", label="feed-forward")
    fig.text(30, 278, "norm before each part, residual", kind="note")
    fig.text(30, 296, "around it, as in GPT-2", kind="note")
    fig.arrow((164, 311), (164, 327))
    fig.box(14, 328, 300, 34, role="enc", label="H: one vector per input token", strong=True)

    fig.text(420, 86, "decoder × 6 layers", kind="subtitle")
    fig.box(420, 98, 326, 34, role="data", label="written so far: <pad> Die Wachen")
    fig.arrow((583, 133), (583, 149))
    fig.box(420, 150, 326, 202, role="neutral")
    fig.box(436, 164, 294, 34, role="dec", label="self-attention, causal")
    fig.arrow((583, 199), (583, 213))
    fig.box(436, 214, 294, 34, role="att", label="cross-attention: Q from here, K, V from H", strong=True)
    fig.arrow((583, 249), (583, 263))
    fig.box(436, 264, 294, 34, role="enc", label="feed-forward")
    fig.text(436, 330, "then one more norm and the vocabulary", kind="note")
    fig.arrow((583, 353), (583, 369))
    fig.box(420, 370, 326, 30, role="dec", label="next token: kommen")
    fig.arrow((315, 345), (360, 345), (360, 231), (434, 231), role="att", width=2.2)
    fig.text(370, 292, "H", kind="label", role="att")
    return fig


# ---------------------------------------------------------------- two ways to corrupt a text

SENTENCE = ["The", "guards", "arrive", "at", "the", "museum", "at", "seven", "in", "the", "morning", "."]


def objectives() -> Figure:
    fig = Figure(
        760,
        346,
        alt="Two ways to pre-train an encoder–decoder on raw text. T5's span corruption replaces spans of the "
        "sentence with sentinel tokens, extra_id_0 and extra_id_1, and the target lists only the dropped spans, "
        "each after its sentinel, so the target is short. BART's text infilling replaces each span with a single "
        "mask token, which can stand for any number of words, and the target is the whole original sentence",
    )
    fig.text(14, 28, "Pre-training by repair: T5 and BART corrupt text differently", kind="title")
    fig.text(14, 50, "the encoder reads the damaged text; the decoder writes what is needed to undo the damage",
             kind="note")
    fig.text(14, 88, "T5: span corruption", kind="subtitle")
    fig.text(14, 112, "encoder reads", kind="label")
    t5_in = ["The", "guards", "<extra_id_0>", "the", "museum", "at", "<extra_id_1>", "the", "morning", "."]
    fig.row(t5_in, 130, 96, role="data", strong=lambda i, t: t.startswith("<"), size=11, h=26)
    fig.text(14, 150, "decoder writes", kind="label")
    t5_out = ["<extra_id_0>", "arrive", "at", "<extra_id_1>", "seven", "in", "<extra_id_2>"]
    fig.row(t5_out, 130, 134, role="dec", strong=lambda i, t: t.startswith("<"), size=11, h=26)
    fig.text(130, 182, "only the missing spans: a short target", kind="note")

    fig.text(14, 222, "BART: text infilling", kind="subtitle")
    fig.text(14, 246, "encoder reads", kind="label")
    bart_in = ["The", "guards", "<mask>", "the", "museum", "at", "<mask>", "the", "morning", "."]
    fig.row(bart_in, 130, 230, role="data", strong=lambda i, t: t.startswith("<"), size=11, h=26)
    fig.text(14, 284, "decoder writes", kind="label")
    fig.row(SENTENCE, 130, 268, role="dec", size=11, h=26)
    fig.text(130, 316, "the whole sentence back; one <mask> may hide any number of words, even none", kind="note")
    return fig


# ---------------------------------------------------------------- the vocabulary's rows

def vocab() -> Figure:
    c = config(T5_SMALL)
    fig = Figure(
        760,
        250,
        alt="The rows of t5-small's embedding matrix, not to scale. Rows 0 to 31,999 are the SentencePiece "
        "vocabulary, from the padding, end and unknown tokens to the last piece. Rows 32,000 to 32,099 are the 100 "
        "sentinels, extra_id_99 down to extra_id_0. Rows 32,100 to 32,127 are 28 rows that no token uses, left over "
        "because the matrix was rounded up to a multiple of 128. The new token for less-than takes row 32,100",
    )
    fig.text(14, 28, "Where a new token can go: the rows of T5's embedding matrix", kind="title")
    fig.text(14, 50, f"{c['vocab_size']:,} rows, of which the tokenizer uses 32,100; not to scale", kind="note")
    y = 84
    fig.box(14, y, 420, 44, role="data", label="32,000 SentencePiece pieces")
    fig.box(440, y, 170, 44, role="att", label="100 sentinels")
    fig.box(616, y, 130, 44, role="neutral", label="28 unused")
    fig.text(14, 156, "rows 0–31,999: <pad>, </s>, <unk>, then pieces of English, German, French, Romanian",
             kind="note")
    fig.text(14, 180, "rows 32,000–32,099: the sentinels, <extra_id_99> down to <extra_id_0>", kind="note",
             role="att")
    fig.text(14, 204, "rows 32,100–32,127: no token uses them; the matrix was rounded up to 251 × 128 rows",
             kind="note")
    fig.text(14, 232, "the new token '<' takes row 32,100, the first free one", kind="label", role="dec")
    return fig


# ---------------------------------------------------------------- two ways to grade a query

def execmatch() -> Figure:
    fig = Figure(
        760,
        372,
        alt="Two ways to grade a predicted query, on an illustrative table of four people aged 41, 29, 35 and 52. The "
        "reference is AGE greater than 34. A prediction of AGE at least 35 is a different string, so exact "
        "matching calls it wrong, but it returns the same three people and is right. A prediction of AGE greater "
        "than 30 also returns the same three people on this table, so execution calls it right although it is "
        "wrong; a second table with a person aged 32 tells the two apart, which is what a test suite does",
    )
    fig.text(14, 28, "Grading a query: compare the text, or run it", kind="title")
    fig.text(14, 50, "an illustration: the question is \"Who is older than 34?\" and the table has four people",
             kind="note")
    fig.text(14, 86, "People", kind="subtitle")
    people = [("Ana", 41), ("Ben", 29), ("Chen", 35), ("Dana", 52)]
    for k, (name, age) in enumerate(people):
        fig.box(14, 98 + k * 36, 92, 30, role="data", label=name)
        fig.box(110, 98 + k * 36, 52, 30, role="data", label=str(age))
    rows = [("reference", "AGE > 34", "—", "Ana, Chen, Dana", "—"),
            ("prediction 1", "AGE >= 35", "✗ different text", "Ana, Chen, Dana", "✓ right"),
            ("prediction 2", "AGE > 30", "✗ different text", "Ana, Chen, Dana", "✓ but wrong")]
    fig.text(196, 86, "WHERE …", kind="subtitle")
    fig.text(330, 86, "exact match", kind="subtitle")
    fig.text(468, 86, "rows it returns", kind="subtitle")
    fig.text(640, 86, "execution", kind="subtitle")
    for k, (who, where, exact, got, run) in enumerate(rows):
        y = 98 + k * 52
        fig.text(196, y + 12, who, kind="note")
        fig.box(196, y + 18, 120, 28, role="dec" if k else "neutral", label=where)
        if k:
            fig.text(330, y + 37, exact, kind="label", role="dec")
            fig.text(640, y + 37, run, kind="label", role="enc" if k == 1 else "dec")
        fig.text(468, y + 37, got, kind="label")
    fig.text(14, 286, "prediction 1 is the same query written differently: exact match is too strict", kind="note")
    fig.text(14, 308, "prediction 2 only agrees on this table: add Eli, aged 32, and the rows differ", kind="note")
    fig.text(14, 330, "Spider's official metric runs every query on a suite of such tables", kind="note")
    return fig


# ---------------------------------------------------------------- what an encoder–decoder costs

def cost() -> Figure:
    fig = Figure(
        760,
        330,
        alt="What an encoder–decoder costs against a decoder-only model. The decoder-only model is one stack with P "
        "parameters, and every token, of the input and of the output, passes through it. The encoder–decoder has "
        "two stacks of P parameters each: input tokens pass through the encoder, output tokens through the "
        "decoder. Each token meets P parameters in both cases, so the compute is about the same, but the "
        "encoder–decoder has to keep twice the parameters in memory",
    )
    fig.text(14, 28, "The same work per token, twice the weights", kind="title")
    fig.text(14, 50, "every token passes through one stack of P parameters, in either design", kind="note")
    fig.text(14, 86, "decoder-only: P parameters", kind="subtitle")
    tokens_in = ["question", "schema"]
    tokens_out = ["SELECT", "…"]
    fig.row(tokens_in + tokens_out, 14, 98, role="data", size=11, h=26)
    fig.box(14, 140, 300, 70, role="dec", label="one stack, P")
    fig.arrow((164, 125), (164, 139))
    fig.text(14, 236, "input and output tokens both pass through P", kind="note")
    fig.text(14, 256, "memory: P", kind="label")

    fig.text(400, 86, "encoder–decoder: 2P parameters", kind="subtitle")
    fig.row(tokens_in, 400, 98, role="data", size=11, h=26)
    fig.row(tokens_out, 590, 98, role="data", size=11, h=26)
    fig.box(400, 140, 160, 70, role="enc", label="encoder, P")
    fig.box(586, 140, 160, 70, role="dec", label="decoder, P")
    fig.arrow((470, 125), (470, 139))
    fig.arrow((640, 125), (640, 139))
    fig.arrow((561, 175), (585, 175), role="att")
    fig.text(400, 236, "input tokens through the encoder, output tokens", kind="note")
    fig.text(400, 256, "through the decoder: each still meets P", kind="note")
    fig.text(400, 276, "memory: 2P", kind="label", role="dec")
    fig.text(14, 312, "compute per token: the same; weights to hold: twice as many (Wang et al., 2022)",
             kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(T5_SMALL)
    rows = [
        ("published", "October 2019, Google; the paper ran 67 pages of experiments"),
        ("type", "encoder–decoder: reads the input whole, writes the output token by token"),
        ("pre-training", "span corruption on C4, then (v1.0) a mixture of supervised tasks"),
        ("attention", "bidirectional in the encoder; causal plus cross-attention in the decoder"),
        ("positions", f"no position embeddings: a learned bias per head, {c['relative_attention_num_buckets']} "
                      "log-spaced buckets"),
        ("normalisation", "pre-norm, scale only: no mean, no bias (RMSNorm)"),
        ("activation", f"ReLU in v1.0, gated GELU from v1.1; feed-forward {c['d_ff']:,} wide in t5-small"),
        ("vocabulary", f"SentencePiece, {c['vocab_size']:,} rows; no '<', '{{' or '}}'"),
        ("sizes", "60M (small), 220M (base), 770M (large), 3B, 11B"),
        ("good at", "input → output tasks: translation, extraction, SQL, a text encoder"),
        ("not for", "open-ended chat, long free generation, in-context learning"),
    ]
    return arch.passport(
        "T5", rows,
        alt="T5's passport. Published October 2019 by Google. Type: encoder–decoder. Pre-training: span corruption "
        "on C4, and in version 1.0 a mixture of supervised tasks. Attention: bidirectional in the encoder, causal "
        "self-attention and cross-attention in the decoder. Positions: no position embeddings, a learned bias per "
        "head over 32 log-spaced buckets of distance. Normalisation: pre-norm with scale only, no mean and no bias. "
        "Activation: ReLU in version 1.0, gated GELU from version 1.1. Vocabulary: SentencePiece, 32,128 rows, "
        "without the characters less-than and curly braces. Sizes: 60 million, 220 million, 770 million, 3 and 11 "
        "billion parameters. Good at: input-to-output tasks such as translation, extraction, SQL, and as a text "
        "encoder. Not for: open-ended chat, long free generation, in-context learning",
    )


# ---------------------------------------------------------------- cover

def _cover():
    out = [cv.label(cv.MOTIF_X0, 150, "A QUESTION, READ WHOLE")]
    svg, spans, right = cv.chip_row(["How many singers", "are younger than 30?"], cv.MOTIF_X0, 172, size=17)
    cv.fits(right)
    out.append(svg)
    out.append(cv.label(cv.MOTIF_X0, 262, "A QUERY, WRITTEN TOKEN BY TOKEN"))
    svg, spans, right = cv.chip_row(["SELECT", "count(*)", "FROM", "singer"], cv.MOTIF_X0, 284, size=17)
    cv.fits(right)
    out.append(svg)
    svg, spans, right = cv.chip_row(["WHERE", "Age", "<", "30"], cv.MOTIF_X0, 330, size=17, kinds={2: "hot"})
    cv.fits(right)
    out.append(svg)
    out.append(cv.text(cv.MOTIF_X0, 410, "checked by running it on SQLite", 18, cv.MUTED, cv.SANS))
    return cv.frame("a05_t5.ipynb", ["T5 and BART"], ["The encoder–decoder,", "and a model that writes SQL"],
                    "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "encdec": encdec,
    "objectives": objectives,
    "vocab": vocab,
    "execmatch": execmatch,
    "cost": cost,
    "passport": passport,
    "family": lambda: arch.family(5),
}
COVERS = {"en": _cover}
