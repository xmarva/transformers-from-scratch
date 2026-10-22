"""Static figures and covers of lesson 1.

The leakage graph uses real pairs from the Tatoeba data (they all belong to one connected component). The cover
segmentations are the outputs of the lesson's from-scratch BPE (8 000 merges), copied from the executed notebook.
"""

from tools.figures import cover as cv
from tools.figures.lib import Figure


def pipeline() -> Figure:
    fig = Figure(
        760,
        132,
        alt="Where tokenization sits: raw text is normalized, split into words, merged into "
        "subword tokens by BPE and mapped to integer ids that the model reads",
    )
    fig.text(14, 26, "From text to numbers", kind="title")
    stages = [
        ("raw text", "data", "as typed"),
        ("normalize", "neutral", "NFKC, quotes"),
        ("pre-tokenize", "neutral", "words, ▁ marks"),
        ("BPE merges", "neutral", "subword tokens"),
        ("token ids", "data", "integers"),
        ("model", "plain", "lesson 2"),
    ]
    x, w, gap = 14, 106, 20
    for i, (name, role, note) in enumerate(stages):
        fig.box(x, 46, w, 40, role=role, label=name)
        fig.text(x + w / 2, 108, note, kind="label", anchor="middle")
        if i < len(stages) - 1:
            fig.arrow((x + w + 2, 66), (x + w + gap - 2, 66))
        x += w + gap
    return fig


# Real pairs from the training data; together they form one connected component.
EN = ["Cut it out!", "Stop it.", "Stop that!", "Enough!"]
RU = ["Перестань!", "Прекрати!", "Прекрати это!", "Хватит!"]
EDGES = [(0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (2, 2), (0, 3), (3, 3)]


def leakage() -> Figure:
    fig = Figure(
        760,
        312,
        alt="A random split puts the pair 'Cut it out!' / 'Прекрати!' into validation while both "
        "sentences also appear in training pairs; splitting by connected component keeps the whole group "
        "on one side",
    )
    panels = [(14, "Random split of pairs", {(0, 1)}), (400, "Split by connected component", set(EDGES))]
    for x0, title, val in panels:
        fig.text(x0, 26, title, kind="title")
        en_x, ru_x, w_en, w_ru = x0, x0 + 222, 112, 124
        for i, s in enumerate(EN):
            fig.box(en_x, 50 + i * 52, w_en, 32, role="plain", label=s)
        for j, s in enumerate(RU):
            fig.box(ru_x, 50 + j * 52, w_ru, 32, role="plain", label=s)
        for i, j in EDGES:
            in_val = (i, j) in val
            fig.line(
                (en_x + w_en, 66 + i * 52),
                (ru_x, 66 + j * 52),
                role="dec" if in_val else "enc",
                width=2.6 if in_val else 1.6,
            )
    fig.text(14, 274, "teal: training pairs", kind="label", role="enc")
    fig.text(14, 294, "red: validation, yet seen in training", kind="label", role="dec")
    fig.text(400, 274, "whole component in validation", kind="label", role="dec")
    return fig


def bpe_loop() -> Figure:
    fig = Figure(
        560,
        252,
        alt="The BPE training loop: count adjacent symbol pairs, take the most frequent, merge "
        "it everywhere, add the new symbol to the vocabulary, and repeat until the vocabulary is full",
    )
    fig.text(14, 26, "BPE training loop", kind="title")
    boxes = {"tl": (36, 50), "tr": (304, 50), "br": (304, 170), "bl": (36, 170)}
    labels = {
        "tl": "1 · count adjacent pairs",
        "tr": "2 · take the most frequent",
        "br": "3 · merge it everywhere",
        "bl": "4 · add it to the vocabulary",
    }
    for key, (x, y) in boxes.items():
        fig.box(x, y, 220, 42, role="data" if key == "br" else "neutral", label=labels[key])
    fig.arrow((258, 71), (302, 71))
    fig.arrow((414, 94), (414, 168))
    fig.arrow((302, 191), (258, 191))
    fig.arrow((146, 168), (146, 94))
    fig.text(280, 136, "repeat until the vocabulary is full", kind="note", anchor="middle")
    return fig


def _bpe_cover(word: str, pieces: list[str], hot: int, title, hook, labels, count_line) -> str:
    assert "".join(pieces) == word, (pieces, word)
    out = [cv.label(cv.MOTIF_X0, 128, labels[0])]
    size, pad = (18, 12) if len(word) < 15 else (16, 8)
    row, spans, end = cv.chip_row(list(word), cv.MOTIF_X0, 142, size, gap=4, pad=pad)
    cv.fits(end)
    out.append(row)
    y2, i = 382, 0
    for g, piece in enumerate(pieces):
        n = len(piece)
        left, right = spans[i][0], spans[i + n - 1][0] + spans[i + n - 1][1]
        w = right - left
        color = cv.RED if g == hot else "#5E616A"
        for k in range(i, i + n):
            cx = spans[k][0] + spans[k][1] / 2
            out.append(
                f'<path d="M{cx:.1f},182 C{cx:.1f},280 {left + w / 2:.1f},286 {left + w / 2:.1f},{y2}" '
                f'fill="none" stroke="{color}" stroke-width="{2 if g == hot else 1.3}"/>'
            )
        stroke, fill, ink = (cv.RED, cv.RED, cv.BG) if g == hot else (cv.LINE, cv.CHIP_FILL, cv.INK)
        out.append(
            f'<rect x="{left:.1f}" y="{y2}" width="{w:.1f}" height="46" rx="8" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.6"/>'
        )
        out.append(cv.text(left + w / 2, y2 + 30, piece, 22, ink, cv.MONO, 500, "middle"))
        i += n
    out.append(cv.label(cv.MOTIF_X0, 458, labels[1]))
    out.append(cv.text(cv.MOTIF_X0, 500, count_line, 18, cv.MUTED, cv.MONO))
    return cv.frame("01_tokenization.ipynb", title, hook, "".join(out))


def cover_en() -> str:
    pieces = ["mis", "under", "st", "and", "ing"]  # from-scratch BPE output, see the notebook
    return _bpe_cover(
        "misunderstanding",
        pieces,
        hot=1,
        title=["Tokenization,", "from scratch"],
        hook=["How does a model read", "«misunderstanding»?"],
        labels=("CHARACTERS", "SUBWORD TOKENS"),
        count_line="16 characters → 5 tokens",
    )


def cover_ru() -> str:
    pieces = ["недо", "по", "нима", "ние"]  # from-scratch BPE output, see the notebook
    return _bpe_cover(
        "недопонимание",
        pieces,
        hot=2,
        title=["Токенизация", "с нуля"],
        hook=["Как модель читает", "«недопонимание»?"],
        labels=("СИМВОЛЫ", "ПОДСЛОВНЫЕ ТОКЕНЫ"),
        count_line="13 символов → 4 токена",
    )


FIGURES = {"pipeline": pipeline, "leakage": leakage, "bpe-loop": bpe_loop}
COVERS = {"en": cover_en, "ru": cover_ru}
