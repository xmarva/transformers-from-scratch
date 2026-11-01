"""Static figures and cover of the second course's lesson 3, text embeddings and search."""

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

# ---------------------------------------------------------------- hero: three ways to score a pair

W = 236  # one column of the hero


def _two_towers(fig, x, vectors):
    """Query and document, each through the same encoder; `vectors` draws what comes out under one tower."""
    half = (W - 12) / 2
    for k, name in enumerate(["query", "document"]):
        bx = x + k * (half + 12)
        mid = bx + half / 2
        fig.box(bx, 118, half, 30, role="data", label=name)
        fig.arrow((mid, 149), (mid, 167))
        fig.box(bx, 168, half, 44, role="enc", label="encoder")
        fig.arrow((mid, 213), (mid, 229))
        vectors(mid)
        fig.arrow((mid, 257), (mid, 279))


def hero() -> Figure:
    fig = Figure(
        760,
        372,
        alt="Three ways to score a query against a document. A cross-encoder reads the two texts joined into one "
        "input and outputs a relevance score: one full pass of the encoder for every pair. A bi-encoder, "
        "Sentence-BERT, encodes each text separately into one vector and compares the vectors by cosine, so the "
        "documents can be encoded before any query arrives. Late interaction, ColBERT, also encodes them "
        "separately but keeps one vector per token and sums, over the query's tokens, the best match in the document",
    )
    fig.text(14, 28, "Three ways to score a query against a document", kind="title")
    fig.text(14, 50, "only the last two can read the documents before the query arrives", kind="note")
    columns = [(14, "cross-encoder", "both texts in one input"), (262, "bi-encoder", "Sentence-BERT, 2019"),
               (510, "late interaction", "ColBERT, 2020")]
    for x, name, note in columns:
        fig.text(x, 84, name, kind="subtitle")
        fig.text(x, 102, note, kind="note")

    x = 14
    fig.box(x, 118, W, 30, role="data", label="query [SEP] document")
    fig.arrow((x + W / 2, 149), (x + W / 2, 167))
    fig.box(x, 168, W, 44, role="enc", label="encoder")
    fig.arrow((x + W / 2, 213), (x + W / 2, 279))
    fig.box(x, 280, W, 32, role="att", label="relevance score", strong=True)

    _two_towers(fig, 262, lambda mid: fig.box(mid - 36, 230, 72, 26, role="data", label="vector"))
    fig.box(262, 280, W, 32, role="att", label="cosine of two vectors", strong=True)

    def many(mid):
        for i in range(6):
            fig.rect(mid - 46 + i * 16, 230, 12, 26, role="data", r=3)

    _two_towers(fig, 510, many)
    fig.box(510, 280, W, 32, role="att", label="sum of best matches", strong=True)

    notes = [("one full encoder pass", "for every query–document pair"),
             ("documents encoded once,", "one vector each, before the query"),
             ("documents encoded once,", "one vector for every token")]
    for (x, _, _), (first, second) in zip(columns, notes, strict=True):
        fig.text(x, 336, first, kind="note")
        fig.text(x, 354, second, kind="note")
    return fig


# ---------------------------------------------------------------- BM25

def bm25() -> Figure:
    fig = Figure(
        760,
        300,
        alt="BM25's score of a document for a query is a sum over the query's words of three factors: how rare the "
        "word is in the collection, the inverse document frequency; how often it occurs in the document, with "
        "returns that diminish, controlled by k1; and the document's length relative to the average, controlled "
        "by b, so that long documents don't win by repeating everything",
    )
    fig.text(14, 28, "BM25: a sum over the query's words", kind="title")
    fig.text(14, 50, "each word that the query and the document share adds three factors multiplied together",
             kind="note")
    parts = [
        ("rarity", "IDF(word)", "a word in every paper, like 'model', is worth almost nothing", "data"),
        ("repetition", "tf · (k1 + 1) / (tf + k1 · …)", "the 2nd occurrence counts, the 20th barely does", "enc"),
        ("length", "… = 1 − b + b · length / average", "a long abstract needs more hits for the same score", "att"),
    ]
    for k, (name, formula, meaning, role) in enumerate(parts):
        y = 74 + k * 62
        fig.box(14, y, 130, 44, role=role, label=name)
        fig.text(162, y + 19, formula, kind="mono")
        fig.text(162, y + 38, meaning, kind="note")
    fig.text(14, 280, "k1 = 1.2 and b = 0.75 are the usual defaults; no training, no GPU, one inverted index",
             kind="body")
    return fig


# ---------------------------------------------------------------- in-batch negatives

def inbatch() -> Figure:
    names = ["BERT", "RoBERTa", "ELECTRA", "DeBERTa"]
    fig = Figure(
        760,
        330,
        alt="Training with in-batch negatives. A batch holds the titles and abstracts of four papers. Every title "
        "is compared with every abstract; for each title, its own abstract is the right answer and the other "
        "abstracts of the batch are wrong ones, so the loss is cross-entropy over each row of the similarity matrix",
    )
    fig.text(14, 28, "In-batch negatives: the rest of the batch is the wrong answers", kind="title")
    fig.text(14, 50, "a batch of 64 pairs gives every title one right abstract and 63 wrong ones, for free",
             kind="note")
    x0, y0, cell, gap = 140, 104, 84, 6
    fig.text(14, 94, "abstracts →", kind="label")
    for j, name in enumerate(names):
        fig.text(x0 + j * (cell + gap) + cell / 2, 98, name, kind="small", anchor="middle")
    for i, name in enumerate(names):
        y = y0 + i * (36 + gap)
        fig.text(14, y + 22, f"title: {name}", kind="label")
        for j in range(len(names)):
            x = x0 + j * (cell + gap)
            if i == j:
                fig.box(x, y, cell, 36, role="enc", label="right", strong=True)
            else:
                fig.box(x, y, cell, 36, role="plain", label="wrong")
    right = x0 + len(names) * (cell + gap) + 14
    lines = [("each row: softmax over", "body"), ("cosine × 20", "mono"), ("", "body"),
             ("target: the diagonal", "body"), ("", "body"), ("loss: cross-entropy,", "body"),
             ("the mean over rows", "body")]
    for k, (s, kind) in enumerate(lines):
        if s:
            fig.text(right, 122 + k * 20, s, kind=kind)
    fig.text(14, 300, "the encoder learns to pull each pair together and push the rest of the batch away",
             kind="note")
    return fig


# ---------------------------------------------------------------- MaxSim

def maxsim() -> Figure:
    query = ["rotary", "position", "embedding", "[MASK]"]
    doc = ["rope", "encodes", "each", "token's", "position", "by", "rotation"]
    best = {0: 6, 1: 4, 2: 1, 3: 0}  # an illustration of the pattern; section 6 shows a measured one
    fig = Figure(
        760,
        330,
        alt="ColBERT's MaxSim. Every query token is compared by cosine with every token of the document; each "
        "query token keeps only its best match, and the document's score is the sum of these maxima. Here the "
        "query token 'rotary' finds 'rotation' and 'position' finds 'position'; the [MASK] tokens that pad the "
        "query also look for a match",
    )
    fig.text(14, 28, "MaxSim: every query token looks for its best match", kind="title")
    fig.text(14, 50, "score = the sum, over the query's tokens, of the highest cosine in the document", kind="note")
    x0, y0, cw, ch, gap = 136, 96, 80, 34, 6
    for j, token in enumerate(doc):
        fig.text(x0 + j * (cw + gap) + cw / 2, y0 - 10, token, kind="small", anchor="middle")
    for i, token in enumerate(query):
        y = y0 + i * (ch + gap)
        fig.chip(14, y, token, role="data", h=ch)
        for j in range(len(doc)):
            x = x0 + j * (cw + gap)
            if best[i] == j:
                fig.box(x, y, cw, ch, role="att", label="max", strong=True)
            else:
                fig.rect(x + 2, y + 2, cw - 4, ch - 4, role="att", tint=True, r=5)
    fig.text(14, 300, "a query of 32 tokens against a 180-token abstract: 5,760 cosines, 32 maxima, one sum",
             kind="body")
    return fig


# ---------------------------------------------------------------- the search engine

def pipeline() -> Figure:
    fig = Figure(
        760,
        236,
        alt="A search engine in three stages. All papers go to two retrievers, BM25 on words and a bi-encoder on "
        "meaning, each returning its top 100; reciprocal rank fusion merges the two rankings; a cross-encoder "
        "reads the query with each of the 50 best candidates and returns the final top five",
    )
    fig.text(14, 28, "A search engine: cheap stages first, expensive ones on few candidates", kind="title")
    fig.text(14, 50, "each stage sees fewer documents and reads each of them more carefully", kind="note")
    fig.box(14, 104, 110, 52, role="data", label="all papers")
    fig.box(160, 74, 150, 44, role="neutral", label="BM25: words")
    fig.box(160, 142, 150, 44, role="enc", label="bi-encoder")
    fig.arrow((125, 120), (142, 120), (142, 96), (159, 96))
    fig.arrow((125, 140), (142, 140), (142, 164), (159, 164))
    fig.box(350, 104, 130, 52, role="att", label="fusion (RRF)")
    fig.arrow((311, 96), (330, 96), (330, 122), (349, 122))
    fig.arrow((311, 164), (330, 164), (330, 138), (349, 138))
    fig.box(516, 104, 140, 52, role="dec", label="cross-encoder")
    fig.arrow((481, 130), (515, 130))
    fig.box(690, 104, 56, 52, role="data", label="top 5", strong=True)
    fig.arrow((657, 130), (689, 130))
    counts = [(69, "58k papers"), (235, "top 100 each"), (415, "top 50"), (586, "50 pairs read")]
    for x, label in counts:
        fig.text(x, 214, label, kind="count", anchor="middle")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    rows = [
        ("published", "August 2019, Reimers and Gurevych, UKP Lab, TU Darmstadt"),
        ("type", "bi-encoder: one encoder applied to each text on its own"),
        ("base", "BERT or RoBERTa in 2019; any encoder since, decoders too"),
        ("output", "one vector per text, the mean of its token vectors"),
        ("comparison", "cosine; the documents are encoded once, ahead of time"),
        ("training, 2019", "sentence pairs from SNLI and MultiNLI, about a million"),
        ("training, now", "contrastive, with in-batch negatives, up to a billion pairs"),
        ("relatives", "cross-encoder (reranking), ColBERT (a vector per token)"),
        ("good at", "semantic search, retrieval for RAG, clustering, duplicates"),
        ("not for", "exact names and numbers; the fine judgement of a reranker"),
    ]
    return arch.passport(
        "Sentence-BERT", rows,
        alt="Sentence-BERT's passport: published August 2019 by Reimers and Gurevych at TU Darmstadt; a "
        "bi-encoder, one encoder applied to each text separately; built on BERT or RoBERTa; one vector per text, "
        "the mean of its token vectors, compared by cosine; trained in 2019 on about a million sentence pairs "
        "from SNLI and MultiNLI, today contrastively with in-batch negatives on up to a billion pairs; relatives "
        "are the cross-encoder and ColBERT; good at semantic search, retrieval, clustering and duplicates; not for "
        "exact names and numbers",
    )


# ---------------------------------------------------------------- cover

def _cover():
    pairs = [("rotary", "rotation"), ("position", "position"), ("embedding", "encodes"), ("[MASK]", "rope")]
    out = [cv.label(cv.MOTIF_X0, 150, "EACH QUERY TOKEN, ITS BEST MATCH")]
    for k, (q, d) in enumerate(pairs):
        y = 182 + k * 62
        chip, w = cv.chip(cv.MOTIF_X0, y, q, size=19, kind="warm")
        out.append(chip)
        out.append(cv.text(cv.MOTIF_X0 + 150, y + 27, "→", 19, cv.MUTED, cv.MONO))
        chip, w = cv.chip(cv.MOTIF_X0 + 190, y, d, size=19, kind="hot" if k == 0 else "plain")
        out.append(chip)
        cv.fits(cv.MOTIF_X0 + 190 + w)
    out.append(cv.text(cv.MOTIF_X0, 450, "score = Σ max cos", 19, cv.MUTED, cv.MONO))
    return cv.frame("a03_search.ipynb", ["Embeddings", "and search"],
                    ["How a machine finds", "the paper you mean"], "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "bm25": bm25,
    "inbatch": inbatch,
    "maxsim": maxsim,
    "pipeline": pipeline,
    "passport": passport,
    "family": lambda: arch.family(3),
}
COVERS = {"en": _cover}
