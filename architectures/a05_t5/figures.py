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
    "passport": passport,
    "family": lambda: arch.family(5),
}
COVERS = {"en": _cover}
