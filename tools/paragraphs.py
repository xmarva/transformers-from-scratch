"""Split long paragraphs of a post into shorter ones, at sentence boundaries, for reading on a screen.

Long blocks of text are hard to read on the web; one to three sentences, a few dozen words, is the comfortable size.
A paragraph longer than MAX_WORDS words or MAX_SENTENCES sentences is cut into pieces chosen by dynamic programming:
each piece is scored by how far its length is from TARGET_WORDS, and a cut is penalised where it would separate
what belongs together:
- after a sentence that ends with a colon, which introduces what follows;
- before a sentence that refers back to the previous one ("This", "It", "But", "So", "Here" ...);
- where it would leave a very short piece on its own.
A sentence end inside a quotation is not a place to cut, so a quoted passage stays in one paragraph.
Only plain prose is touched: headings, lists, quotes, tables, HTML, code, display math and figure placeholders are
left as they are. Links, inline code, inline math and inline HTML are hidden while sentences are found, so the dots
inside them do not count; so are abbreviations (et al., e.g., U.S.), initials and decimals.
"""

from __future__ import annotations

import re

TARGET_WORDS = 38  # the length a piece should have
MAX_WORDS = 60  # a paragraph longer than this is split...
MAX_SENTENCES = 3  # ...and so is one with more sentences than this
HARD_WORDS = 85  # no piece longer than this, unless it is a single sentence

# an HTML block is left alone, but a paragraph may open with an inline tag (a case's name, a key figure)
SKIP = re.compile(
    r"^(#|[-*+] |\d+[.)] |>|\||<(?!(?:a|b|i|s|em|strong|span|mark|code|sup|sub)\b)|\$\$|```|\[\[|\{[%:]|!\[|\s{4}|\t)"
)
PROTECT = re.compile(
    r"(<sup\b.*?</sup>|<[^>]+>|`[^`]*`|\$\$.*?\$\$|\$[^$\n]+\$|!?\[[^\]]*\]\([^)\s]*\))",
    re.S,
)
ABBREVIATIONS = {
    "al", "e.g", "i.e", "etc", "vs", "cf", "fig", "figs", "eq", "eqs", "no", "nos", "vol", "pp", "p", "dr", "mr",
    "mrs", "ms", "st", "jr", "sr", "inc", "ltd", "co", "approx", "ca", "u.s", "u.k", "a.d", "b.c", "ph.d", "ed",
    "eds", "rev", "dept", "univ", "sec", "ch", "op", "ibid",
}
BACK_REFERENCES = re.compile(
    r"^(?:[\"“(]?)(This|That|These|Those|It|Its|They|Their|Them|He|She|His|Her|Such|Both|Each|Either|Neither|"
    r"But|And|Or|So|Yet|Then|Thus|Hence|Still|Also|Instead|Otherwise|Here|There|Which|Its|The same|In turn|"
    r"In other words|That is|For example|For instance|Even|Only|Again)\b"
)
SENTENCE_END = re.compile(r"(?<=[.!?…])([\"”’)\]*_]*)(\s+)(?=[\"“‘(\[A-Z0-9\x00*_])")


def _mask(text: str) -> tuple[str, list[str]]:
    kept: list[str] = []

    def hide(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\x00{len(kept) - 1}\x01"

    return PROTECT.sub(hide, text), kept


def _unmask(text: str, kept: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x01", lambda m: kept[int(m.group(1))], text)


def _inside_quote(text: str) -> bool:
    """Whether a quotation opened in the text is still open at its end (straight or curly double quotes)."""
    return text.count('"') % 2 == 1 or text.count("“") > text.count("”")


def sentences(paragraph: str) -> list[str]:
    masked, kept = _mask(paragraph)
    pieces, start = [], 0
    for match in SENTENCE_END.finditer(masked):
        before = masked[start:match.start()].rstrip(".!?…")
        last_word = re.split(r"[\s(]", before)[-1].lower() if before else ""
        if last_word in ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word) or re.fullmatch(r"(\w\.)+\w", last_word):
            continue  # an abbreviation or an initial, not the end of a sentence
        if _inside_quote(masked[:match.end(1)]):
            continue  # a sentence inside a quotation: the quotation stays in one piece
        end = match.start(2)
        pieces.append(masked[start:end])
        start = match.end()
    pieces.append(masked[start:])
    return [_unmask(p, kept).strip() for p in pieces if p.strip()]


def _words(sentence: str) -> int:
    masked, kept = _mask(sentence)
    count = len(re.findall(r"\S+", re.sub(r"\x00\d+\x01", " x ", masked)))
    return max(count, 1)


def split(paragraph: str) -> list[str]:
    parts = sentences(paragraph)
    counts = [_words(s) for s in parts]
    total = sum(counts)
    if len(parts) < 2 or (total <= MAX_WORDS and len(parts) <= MAX_SENTENCES):
        return [paragraph]

    def cut_penalty(k: int) -> float:
        """The cost of starting a new paragraph before sentence k."""
        penalty = 0.0
        if parts[k - 1].rstrip().endswith(":"):
            penalty += 10_000  # a colon introduces what follows
        if BACK_REFERENCES.match(parts[k]):
            penalty += 260
        return penalty

    def piece_cost(i: int, j: int) -> float:
        words = sum(counts[i:j])
        n = j - i
        if words > HARD_WORDS and n > 1:
            return float("inf")
        cost = (words - TARGET_WORDS) ** 2
        if n > MAX_SENTENCES:
            cost += 900 * (n - MAX_SENTENCES)
        if words < 12 and len(parts) > 1:
            cost += 400  # a fragment on its own reads like a slogan
        return cost

    n = len(parts)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            cost = best[i] + piece_cost(i, j) + (cut_penalty(i) if i > 0 else 0)
            if cost < best[j]:
                best[j], back[j] = cost, i
    cuts, j = [], n
    while j > 0:
        cuts.append((back[j], j))
        j = back[j]
    return [" ".join(parts[i:j]) for i, j in reversed(cuts)]


def shorten(body: str) -> str:
    """Split the long prose paragraphs of a markdown body; everything else is returned unchanged."""
    out, in_fence = [], False
    for block in re.split(r"(\n\s*\n)", body):
        if not block.strip() or re.fullmatch(r"\n\s*\n", block):
            out.append(block)
            continue
        fences = block.count("```")
        stripped = block.lstrip("\n")
        lines = stripped.strip().split("\n")
        # a paragraph may be wrapped over several lines (notebook sources are); every line must be plain text
        prose = not in_fence and fences == 0 and not any(SKIP.match(line) for line in lines)
        if fences % 2:
            in_fence = not in_fence
        pieces = split(" ".join(line.strip() for line in lines)) if prose else []
        if len(pieces) > 1:
            lead = block[: len(block) - len(stripped)]
            out.append(lead + "\n\n".join(pieces) + block[len(block.rstrip()):])
        else:
            out.append(block)
    return "".join(out)
