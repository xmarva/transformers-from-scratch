"""Static figures and covers of lesson 1.

Whatever a figure shows about real tokenizers or algorithms is computed here, at render time: GPT-4o's tokens come
from its tokenizer, Unicode facts from `unicodedata`, the union-find trees and the pair counts from the same code
as in the notebook. The leakage graph uses real pairs from the Tatoeba data (they all belong to one connected
component). The cover segmentations are the outputs of the lesson's from-scratch BPE (8 000 merges), copied from
the executed notebook.
"""

import collections
import unicodedata

from tools.figures import cover as cv
from tools.figures.lib import Figure

HERO_QUESTION = "How many r's are in strawberry?"


def _gpt4o_pieces(text):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("Xenova/gpt-4o")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [tokenizer.decode([i]) for i in ids], ids


def hero() -> Figure:
    pieces, ids = _gpt4o_pieces(HERO_QUESTION)
    assert "".join(pieces) == HERO_QUESTION
    hot = pieces.index(" strawberry")
    fig = Figure(
        760,
        262,
        alt=f"The question '{HERO_QUESTION}' as GPT-4o receives it: {len(ids)} tokens, each an integer id. "
        f"The word ' strawberry', with its leading space, is a single token, id {ids[hot]}, so its letters never "
        "reach the model",
    )
    fig.text(14, 28, "What GPT-4o receives", kind="title")
    x0 = 150
    fig.text(14, 74, "you type", kind="label")
    fig.text(x0, 75, HERO_QUESTION, kind="lead")
    fig.text(14, 128, "tokens", kind="label")
    shown = [piece.replace(" ", "␣") for piece in pieces]
    spans = fig.row(shown, x0, 108, strong=lambda i, t: i == hot, gap=6, h=34, size=15)
    fig.text(14, 180, "token ids", kind="label")
    for i, ((x, w), token_id) in enumerate(zip(spans, ids)):
        fig.text(x + w / 2, 180, str(token_id), kind="count_hot" if i == hot else "count", anchor="middle")
    x, w = spans[hot]
    fig.line((x + w / 2, 190), (x + w / 2, 214), role="data", width=1.4)
    note = "one token, one integer: the letters never reach the model"
    fig.text(min(x + w / 2, 754 - len(note) * 13 * 0.28), 234, note, kind="note", anchor="middle")
    fig.text(14, 234, "␣ marks a space", kind="label")
    return fig


def normalization() -> Figure:
    composed, decomposed = "\u0439", "\u0438\u0306"
    assert unicodedata.normalize("NFKC", decomposed) == composed
    fig = Figure(
        760,
        300,
        alt="The letter й stored two ways: as one code point, U+0439, or as и (U+0438) followed by a combining "
        "breve (U+0306). They look the same but are different strings; NFKC normalization turns the second into "
        "the first",
    )
    fig.text(14, 28, "One letter, two strings", kind="title")
    panels = [(14, composed, "one code point"), (394, decomposed, "two code points")]
    for x0, letter, title in panels:
        fig.box(x0, 46, 352, 186, role="plain")
        fig.text(x0 + 20, 76, title, kind="subtitle")
        fig.text(x0 + 24, 150, letter, kind="glyph")
        x = x0 + 92
        for char in letter:
            code = f"U+{ord(char):04X}"
            w = fig.chip(x, 100, code, role="data")
            name = unicodedata.name(char).lower()
            for k, line in enumerate(_wrap(name, 16)):
                fig.text(x, 150 + 16 * k, line, kind="label")
            x += max(w, 128) + 10
        utf8 = " ".join(f"{b:02X}" for b in letter.encode("utf-8"))
        fig.text(x0 + 20, 214, f"UTF-8 bytes: {utf8}", kind="label")
    fig.arrow((570, 236), (570, 262), (190, 262), (190, 240), role="data")
    fig.text(380, 286, 'unicodedata.normalize("NFKC", text)', kind="label", anchor="middle")
    return fig


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


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


def _merge_word(word, pair):
    merged, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged


def bpe_update() -> Figure:
    word, freq, pair = list("▁понимать"), 4, ("н", "и")  # a word of the notebook's toy corpus
    after = _merge_word(word, pair)
    delta = collections.Counter()
    for left, right in zip(word, word[1:]):
        delta[left, right] -= freq
    for left, right in zip(after, after[1:]):
        delta[left, right] += freq
    changes = [(p, d) for p, d in delta.items() if d != 0]
    unchanged = [p for p, d in delta.items() if d == 0]
    fig = Figure(
        760,
        300,
        alt=f"Merging {pair[0]} + {pair[1]} in the word {''.join(word)} (frequency {freq}). The pair counts change "
        "only around the merge: " + ", ".join(f"({a}, {b}) {d:+d}" for (a, b), d in changes),
    )
    fig.text(14, 28, "One merge, one word: which pair counts change", kind="title")
    fig.text(14, 74, "before", kind="label")
    fig.row(word, 84, 54, strong=lambda i, t: (t,) == (word[i],) and i in (3, 4), gap=4)
    fig.text(14, 134, "after", kind="label")
    fig.row(after, 84, 114, strong=lambda i, t: t == "".join(pair), gap=4)
    fig.text(14, 182, f"the word occurs {freq} times, so each of its pairs counts {freq} times", kind="note")
    x0 = 470
    fig.text(x0, 62, "change in pair_freq", kind="label")
    for k, ((left, right), d) in enumerate(changes):
        y = 76 + k * 34
        role = "data" if d > 0 else "plain"
        w1 = fig.chip(x0, y, left, role=role, h=28)
        w2 = fig.chip(x0 + w1 + 4, y, right, role=role, h=28)
        fig.text(x0 + w1 + w2 + 18, y + 19, f"{d:+d}".replace("-", "−"), kind="count_hot" if d > 0 else "count")
    fig.text(14, 214, "all other pairs keep their counts:", kind="note")
    fig.text(14, 236, ", ".join(f"({a}, {b})" for a, b in unchanged), kind="label")
    fig.text(14, 278, "So a merge only has to revisit the words that contain the pair.", kind="body")
    return fig


# The example pairs of the notebook's union-find cell, in the same order.
UF_PAIRS = [
    ("Cut it out!", "Перестань!"),
    ("Stop it.", "Перестань!"),
    ("Cut it out!", "Прекрати!"),
    ("Stop that!", "Прекрати!"),
    ("Stop it.", "Прекрати это!"),
    ("Stop that!", "Прекрати это!"),
    ("Cut it out!", "Хватит!"),
    ("Enough!", "Хватит!"),
]


class _UnionFind:
    """The notebook's UnionFind, plus a record of the path that find() walks."""

    def __init__(self):
        self.parent = {}
        self.last_path = []

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        path = [x]
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
            path.append(root)
        self.last_path = path
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b


def _longest_find():
    """Replay the example pairs; return the forest before and after the find that walks the longest path."""
    groups = _UnionFind()
    best = None
    for en, ru in UF_PAIRS:
        before = dict(groups.parent)
        probe = _UnionFind()
        probe.parent = dict(groups.parent)
        probe.find(en)
        if best is None or len(probe.last_path) > len(best[2]):
            best = (en, before, probe.last_path, dict(probe.parent))
        groups.union(en, ru)
    return best


def _tree_layout(parent, x0, slot, y0, dy):
    children = collections.defaultdict(list)
    roots = []
    for node, up in parent.items():
        if node == up:
            roots.append(node)
        else:
            children[up].append(node)
    assert len(roots) == 1, roots
    pos, next_slot = {}, [0]

    def place(node, depth):
        if children[node]:
            for child in children[node]:
                place(child, depth + 1)
            xs = [pos[child][0] for child in children[node]]
            x = (min(xs) + max(xs)) / 2
        else:
            x = x0 + slot * (next_slot[0] + 0.5)
            next_slot[0] += 1
        pos[node] = (x, y0 + depth * dy)

    place(roots[0], 0)
    return pos, roots[0], next_slot[0]


def union_find() -> Figure:
    query, before, path, after = _longest_find()
    changed = {node for node in after if after[node] != before.get(node)}
    fig = Figure(
        820,
        330,
        alt=f"Path compression in union-find. Before: find('{query}') follows parent links "
        f"{' → '.join(path)}. After: every sentence on that path points straight at the root, "
        f"'{path[-1]}', so the next lookup takes one step",
    )
    fig.text(14, 28, "Union–find: find() with path compression", kind="title")
    size, h, slot, dy = 12, 28, 98, 62
    for panel_x, forest, caption in [
        (14, before, f'before: find("{query}") walks {len(path) - 1} links to the root'),
        (424, after, "after: everything on the path points at the root"),
    ]:
        pos, root, n_slots = _tree_layout(forest, panel_x, slot, 64, dy)
        offset = (382 - n_slots * slot) / 2
        for node, (x, y) in pos.items():
            pos[node] = (x + offset, y)
        for node, up in forest.items():
            if node == up:
                continue
            (cx, cy), (px, py) = pos[node], pos[up]
            on_path = (forest is before and node in path) or (forest is after and node in changed)
            parent_w = len(up) * size * 0.56 + 18
            end_x = px + max(-parent_w / 2 + 14, min(parent_w / 2 - 14, (cx - px) * 0.4))  # spread the arrowheads
            fig.arrow((cx, cy), (end_x, py + h + 2), role="data" if on_path else "muted", width=2 if on_path else 1.4)
        for node, (x, y) in pos.items():
            w = len(node) * size * 0.56 + 18
            role = "data" if node in path else "plain"
            fig.box(x - w / 2, y, w, h, role=role, label=node, r=7)
            fig.els[-1].strong = node == root
            fig.els[-1].size = size
        fig.text(panel_x, 64 + 3 * dy + h + 40, caption, kind="note")
    return fig


def hf_tokenizer() -> Figure:
    fig = Figure(
        760,
        292,
        alt="A Hugging Face tokenizer holds the whole pipeline: to encode, raw text goes through the normalizer, "
        "the pre-tokenizer and the BPE model and comes out as ids; to decode, the decoder turns ids back into "
        "text. Each stage replaces one function written in the lesson, and all of them are saved in one file, "
        "tokenizer.json",
    )
    fig.text(14, 28, "One object, one file: the tokenizer owns its whole pipeline", kind="title")
    fig.box(14, 46, 732, 206, role="plain", r=12)
    fig.text(730, 70, "tokenizer.json", kind="label", anchor="end")
    top, low, h, w = 88, 178, 36, 136
    stages = [("normalizer", "normalize()"), ("pre-tokenizer", "pretokenize()"), ("BPE model", "merges, encode_word()")]
    x = 30
    w_in = fig.chip(x, top + 3, "raw text", role="data")
    x += w_in + 22
    fig.arrow((x - 20, top + h / 2), (x - 3, top + h / 2))
    for name, ours in stages:
        fig.box(x, top, w, h, role="neutral", label=name)
        fig.text(x + w / 2, top + h + 20, ours, kind="label", anchor="middle")
        x += w + 26
        fig.arrow((x - 24, top + h / 2), (x - 3, top + h / 2))
    fig.chip(x, top + 3, "ids", role="data")
    ids_x = x
    model_x = ids_x - 26 - w
    fig.chip(ids_x, low + 3, "ids", role="data")
    fig.arrow((ids_x - 2, low + h / 2), (ids_x - 23, low + h / 2))
    fig.box(model_x, low, w, h, role="neutral", label="decoder")
    fig.text(model_x + w / 2, low + h + 20, "decode()", kind="label", anchor="middle")
    text_w = Figure.chip_width("text")
    fig.arrow((model_x - 2, low + h / 2), (model_x - 23, low + h / 2))
    fig.chip(model_x - 26 - text_w, low + 3, "text", role="data")
    fig.text(30, 72, "encode", kind="label")
    fig.text(30, low + h / 2 + 5, "decode", kind="label")
    fig.text(14, 278, "Under each stage: the function from this lesson that it replaces.", kind="note")
    return fig


def embedding() -> Figure:
    fig = Figure(
        760,
        272,
        alt="The embedding table has one row per token in the vocabulary and d_model columns. A token's id "
        "selects its row, a vector of d_model numbers that the model learns. The table holds |V| times d_model "
        "parameters, so every token added to the vocabulary costs d_model parameters",
    )
    fig.text(14, 28, "Every token costs a row: the embedding table", kind="title")
    w_tok = fig.chip(14, 128, "▁недо", role="data", strong=True)
    fig.arrow((14 + w_tok + 4, 143), (14 + w_tok + 40, 143))
    fig.text(14 + w_tok + 48, 148, "its id", kind="label")
    mx, my, mw, mh, rows = 230, 60, 250, 190, 12
    fig.box(mx, my, mw, mh, role="plain", r=6)
    row_h = mh / rows
    hot = 4
    for r in range(1, rows):
        fig.line((mx, my + r * row_h), (mx + mw, my + r * row_h), role="muted", width=0.6)
    fig.box(mx + 1, my + hot * row_h, mw - 2, row_h, role="data", r=2)
    fig.els[-1].strong = True
    fig.arrow((14 + w_tok + 106, 143), (mx - 4, my + (hot + 0.5) * row_h))
    fig.text(mx + mw / 2, my - 10, "d_model columns", kind="label", anchor="middle")
    fig.text(mx + mw + 14, my + 12, "|V| rows,", kind="label")
    fig.text(mx + mw + 14, my + 30, "one per token", kind="label")
    cells_x, cells_y = 560, my + (hot + 0.5) * row_h - 14
    fig.arrow((mx + mw + 4, my + (hot + 0.5) * row_h), (cells_x - 6, cells_y + 14), role="data")
    for k in range(6):
        fig.box(cells_x + k * 30, cells_y, 26, 28, role="data", r=4)
    fig.text(cells_x, cells_y + 50, "the token's vector:", kind="label")
    fig.text(cells_x, cells_y + 68, "d_model learned numbers", kind="label")
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


FIGURES = {
    "hero": hero,
    "normalization": normalization,
    "pipeline": pipeline,
    "leakage": leakage,
    "union-find": union_find,
    "bpe-loop": bpe_loop,
    "bpe-update": bpe_update,
    "hf-tokenizer": hf_tokenizer,
    "embedding": embedding,
}
COVERS = {"en": cover_en, "ru": cover_ru}
