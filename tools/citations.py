"""Numbered citations for posts exported to xmarva.github.io.

In the sources, a citation is a link in parentheses: "... disappear" ([Shumailov et al., 2024](https://...)). On the
site such links become small numbers, [5], that show the reference on hover or tap and lead to it in the list at the
end. A link counts as a citation when its address is one of the post's references, or when its text has a year in it
("Smithsonian, 2022"); a citation with no entry in the list gets one. Other links in the parentheses, such as
[code](...) or [model card](...), stay links, and so does every link in running text.

The "## References" section becomes one numbered list in the order the sources are first cited, followed by the ones
the text never cites. Grouping lines such as "Papers:" are dropped, since the numbers now order the list.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

LINK = r"\[([^\]]+)\]\(([^)\s]+)\)"
GROUP = re.compile(rf"[ \t]*\(\s*({LINK}(?:\s*[;,]\s*{LINK})*)\s*\)")
ONE = re.compile(LINK)
YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
HEADING = re.compile(r"^## References[ \t]*\n", re.M)


def _references(section: str) -> list[dict]:
    """The list items of the References section: their markdown and every address they link to."""
    items, current = [], None
    for line in section.splitlines():
        if line.startswith("- "):
            current = {"text": line[2:].strip()}
            items.append(current)
        elif current is not None and line.startswith("  ") and line.strip():
            current["text"] += " " + line.strip()
        elif not line.strip():
            continue
        else:
            current = None  # a grouping line ("Papers:") or a closing sentence
    for item in items:
        item["urls"] = [u for _, u in ONE.findall(item["text"])]
    return items


def _trailer(section: str) -> str:
    """What follows the list in the References section and is not a list or a grouping line: kept as it is."""
    lines = [line for line in section.splitlines() if line.strip()]
    keep = [line for line in lines if not line.startswith(("- ", "  ")) and not line.rstrip().endswith(":")]
    return "\n\n".join(keep)


def _generated(text: str, url: str) -> str:
    host = urlparse(url).netloc.removeprefix("www.")
    return f"{text}. [{host}]({url})."


def _label(markdown: str) -> str:
    """A reference as plain text, for the citation's aria-label."""
    plain = ONE.sub(lambda m: m.group(1), markdown)
    plain = re.sub(r"[*_`]", "", plain)
    return html.escape(plain, quote=True)


def numbered(body: str) -> str:
    """Rewrite a post's markdown body (without front matter) to numbered citations."""
    heading = HEADING.search(body)
    if heading:
        start = heading.end()
        end_match = re.compile(r"^(## |\{% endraw %\})", re.M).search(body, start)
        end = end_match.start() if end_match else len(body)
        text, section, tail = body[: heading.start()], body[start:end], body[end:]
        refs = _references(section)
        trailer = _trailer(section)
    else:  # no list yet: one is made from the citations, before the closing {% endraw %} if there is one
        end = body.rfind("{% endraw %}")
        text, tail = (body[:end], body[end:]) if end >= 0 else (body, "")
        refs, trailer = [], ""

    by_url = {url: k for k, ref in enumerate(refs) for url in ref["urls"]}
    order: list[int] = []  # indices into refs, in the order they are first cited
    first_cite: dict[int, str] = {}

    def number(index: int) -> int:
        if index not in order:
            order.append(index)
        return order.index(index) + 1

    counter = [0]

    def group(match: re.Match) -> str:
        cites, keep = [], []
        for label, url in ONE.findall(match.group(1)):
            if url in by_url:
                index = by_url[url]
            elif YEAR.search(label):
                refs.append({"text": _generated(label, url), "urls": [url]})
                index = by_url[url] = len(refs) - 1
            else:
                keep.append(f"[{label}]({url})")
                continue
            cites.append(index)
        if not cites:
            return match.group(0)
        marks = []
        for index in dict.fromkeys(cites):
            n = number(index)
            counter[0] += 1
            anchor = f"cite-{counter[0]}"
            first_cite.setdefault(index, anchor)
            marks.append(f'<a href="#ref-{n}" id="{anchor}" data-ref="{n}" '
                         f'aria-label="Reference {n}: {_label(refs[index]["text"])}">{n}</a>')
        out = '<sup class="cite">[' + ", ".join(marks) + "]</sup>"
        if keep:
            out += " (" + "; ".join(keep) + ")"
        return out

    # code blocks and code spans are left alone: parentheses in code are not citations
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    text = "".join(part if k % 2 else GROUP.sub(group, part) for k, part in enumerate(parts))
    if not refs:
        return body

    cited = [refs[i] for i in order]
    rest = [ref for k, ref in enumerate(refs) if k not in order]
    lines = []
    for n, ref in enumerate(cited + rest, start=1):
        index = refs.index(ref)
        back = (f' <a class="ref-back" href="#{first_cite[index]}" aria-label="Back to the text">&#8617;</a>'
                if index in first_cite else "")
        lines.append(f"{n}. {{:#ref-{n}}} {ref['text']}{back}")
    section = "\n".join(lines) + "\n{: .tfs-refs}\n"
    if trailer:
        section += "\n" + trailer + "\n"
    return text.rstrip("\n") + "\n\n## References\n\n" + section + "\n" + tail.lstrip("\n")
