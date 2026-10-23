"""Export executed lessons as posts for the course's site, xmarva.github.io (Jekyll, al-folio).

For every lesson it writes
    _posts/<published_on>-transformers-from-scratch-NN-<name>.md
    assets/posts/tfs/NN/   the cover, the lesson's figures (light and dark) and every image the notebook produced

What changes on the way from notebook to post:
- the notebook header (cover and badges) becomes front matter and a line of links to Kaggle, Colab and the course;
- inline math $...$ becomes $$...$$, which kramdown renders as inline math (the site's MathJax reads its output);
- a figure from the repository is shown in its light or dark version, following the site's theme;
- tables keep their structure but take the site's colours; text outputs are set apart from code;
- the body is wrapped in {% raw %} so that Liquid leaves Python's braces alone;
- links to the previous and next lessons close the post.

Usage:
    uv run python tools/export_post.py 01 02 03 --site ../../websites/xmarva.github.io

The executed notebooks are read from build/executed/<lesson>.ipynb (tools/run.py, tools/kaggle_run.py).
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import sys
from pathlib import Path

import nbformat
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://xmarva.github.io"
COURSE_PAGE = "/teaching/"
FIGURE = re.compile(r'<img src="https://raw\.githubusercontent\.com/[^"]+/assets/figures/(\d\d)/([\w-]+)-light\.png"([^>]*)>')
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')
# code blocks, code spans, display math, and inline math, which may wrap onto the next line but not past a paragraph
MATH = re.compile(r"(```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$(?:[^$\n]|\n(?!\n))+?\$)", re.S)
TITLE = re.compile(r"""(?:set_title|title=|title\s*=)\(?\s*f?["']([^"'{}]+)["']""")


def lessons() -> dict[str, tuple[Path, dict]]:
    found = {}
    for folder in sorted((ROOT / "lessons").iterdir()):
        toml = folder / "lesson.toml"
        if toml.exists():
            found[folder.name[:2]] = (folder, tomllib.loads(toml.read_text(encoding="utf-8")))
    return found


def post_slug(folder: Path) -> str:
    number, name = folder.name.split("_", 1)
    return f"transformers-from-scratch-{number}-{name.replace('_', '-')}"


def post_url(folder: Path, meta: dict) -> str:
    return f"/blog/{str(meta['published_on'])[:4]}/{post_slug(folder)}/"


def inline_math(text: str) -> str:
    """$x$ -> $$x$$, leaving code spans, code blocks and display math alone.

    A code span that starts or ends with a space (` strawberry`) is not code to kramdown; it becomes <code> with
    non-breaking spaces, which keeps the space visible."""

    def convert(match: re.Match) -> str:
        piece = match.group(0)
        if piece.startswith("```") or piece.startswith("$$"):
            return piece
        if piece.startswith("`"):
            inner = piece[1:-1]
            if inner != inner.strip():
                return '<code markdown="0">' + html.escape(inner).replace(" ", "&nbsp;") + "</code>"
            return piece
        return "$" + piece + "$"

    return MATH.sub(convert, text)


def figure(match: re.Match, assets: Path, web: str) -> str:
    number, name, attributes = match.groups()
    source = ROOT / "assets" / "figures" / number
    shown = []
    for theme in ("light", "dark"):
        shutil.copy(source / f"{name}-{theme}.png", assets / f"{name}-{theme}.png")
        extra = " ".join(f'{key}="{html.escape(html.unescape(value))}"' for key, value in ATTRIBUTE.findall(attributes))
        shown.append(f'<img class="tfs-figure tfs-{theme}" src="{web}/{name}-{theme}.png" {extra} loading="lazy">')
    return "\n".join(shown)


def markdown_cell(source: str, assets: Path, web: str, first: bool) -> str:
    if first:  # the lesson's title is the post's title
        source = re.sub(r"\A# [^\n]*\n+", "", source)
    source = FIGURE.sub(lambda match: figure(match, assets, web), source)
    return inline_math(source)


def fenced(code: str) -> str:
    """A code block whose fence is longer than any run of backticks inside it (a model card quotes Python)."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}python\n{code}\n{fence}"


def alt_text(code: str) -> str:
    titles = TITLE.findall(code)
    return titles[0] if titles else "Output of the code above"


def output_blocks(cell, assets: Path, web: str, counter: list[int]) -> list[str]:
    blocks, text = [], []

    def flush():
        if text:
            printed = html.escape("".join(text).rstrip("\n"))
            blocks.append(f'<pre class="tfs-printed" markdown="0"><code>{printed}</code></pre>')
            text.clear()

    for output in cell.get("outputs", []):
        if output.output_type == "stream" and output.name == "stdout":
            text.append(output.text)
        elif output.output_type in ("display_data", "execute_result"):
            data = output.get("data", {})
            if "image/png" in data:
                flush()
                counter[0] += 1
                name = f"output-{counter[0]:02d}.png"
                (assets / name).write_bytes(base64.b64decode(data["image/png"]))
                alt = html.escape(alt_text(cell.source))
                blocks.append(f'<img class="tfs-output" src="{web}/{name}" alt="{alt}" loading="lazy">')
            elif "text/html" in data:
                flush()
                table = re.sub(r"color:#[0-9A-Fa-f]{6};background:#[0-9A-Fa-f]{6};", "", data["text/html"])
                blocks.append(f'<div class="tfs-table-wrap">{table}</div>')
            elif "text/plain" in data:
                text.append(data["text/plain"] + "\n")
        elif output.output_type == "error":
            raise SystemExit(f"the notebook has an error in a cell: {output.ename}")
    flush()
    return blocks


def links_line(folder: Path, meta: dict) -> str:
    url = build.links(folder.name, meta)
    return (
        f'<p class="tfs-links"><strong>{build.COURSE}</strong> · lesson {meta["number"]} of {build.TOTAL_LESSONS} · '
        f'<a href="{COURSE_PAGE}">all lessons</a> · <a href="{url["kaggle"]}">run on Kaggle</a> · '
        f'<a href="{url["colab"]}">open in Colab</a> · <a href="{url["github"]}">code on GitHub</a></p>'
    )


def neighbours(number: str, exported: dict[str, tuple[Path, dict]]) -> str:
    keys = sorted(exported)
    position = keys.index(number)
    parts = []
    if position > 0:
        folder, meta = exported[keys[position - 1]]
        parts.append(f'<a href="{post_url(folder, meta)}">← Lesson {meta["number"]}: {meta["short_title"]}</a>')
    if position < len(keys) - 1:
        folder, meta = exported[keys[position + 1]]
        parts.append(f'<a href="{post_url(folder, meta)}">Lesson {meta["number"]}: {meta["short_title"]} →</a>')
    if not parts:
        return ""
    return '<p class="tfs-neighbours">' + " ".join(parts) + "</p>"


def export(number: str, site: Path, exported: dict[str, tuple[Path, dict]]) -> Path:
    folder, meta = exported[number]
    notebook_path = ROOT / "build" / "executed" / f"{folder.name}.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    assets = site / "assets" / "posts" / "tfs" / number
    web = f"/assets/posts/tfs/{number}"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    shutil.copy(ROOT / "assets" / "covers" / f"{number}-en.png", assets / "cover.png")

    title = None
    body, counter = [], [0]
    cells = nb.cells[1:]  # the first cell is the generated notebook header
    for index, cell in enumerate(cells):
        if cell.cell_type == "markdown":
            if title is None:
                heading = re.match(r"# Lesson \d+\. (.+)", cell.source)
                if not heading:
                    raise SystemExit(f"{folder.name}: the first markdown cell must start with '# Lesson N. Title'")
                title = heading.group(1).strip()
            body.append(markdown_cell(cell.source, assets, web, first=index == 0))
        elif cell.cell_type == "code" and cell.source.strip():
            if not cell.get("outputs") and cell.get("execution_count") is None:
                raise SystemExit(f"{folder.name}: cell {index} was never run; export an executed notebook")
            body.append(fenced(cell.source.rstrip()))
            body.extend(output_blocks(cell, assets, web, counter))

    front = [
        "---",
        "layout: post",
        f'title: "{title}"',
        f"date: {meta['published_on']} {9 + int(number):02d}:00:00",
        f'description: "{meta["description"]}"',
        f"tags: [transformers-from-scratch, {meta['tags']}]",
        "categories: machine-learning",
        f"thumbnail: {web}/cover.png",
        f"og_image: {SITE_URL}{web}/cover.png",
        "stylesheets: [/assets/css/tfs-course.css]",
        "toc:",
        "  beginning: true",
        "---",
        "",
    ]
    text = "\n".join(front) + "{% raw %}\n\n" + links_line(folder, meta) + "\n\n"
    text += "\n\n".join(body)
    text += "\n\n" + neighbours(number, exported) + "\n\n{% endraw %}\n"
    post = site / "_posts" / f"{meta['published_on']}-{post_slug(folder)}.md"
    post.write_text(text, encoding="utf-8")
    shutil.copy(Path(__file__).with_name("tfs-course.css"), site / "assets" / "css" / "tfs-course.css")
    return post


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lessons", nargs="+", help="lesson numbers, e.g. 01 02")
    parser.add_argument("--site", type=Path, required=True, help="path to the xmarva.github.io checkout")
    args = parser.parse_args(argv)

    available = lessons()
    exported = {}
    for number in args.lessons:
        folder, meta = available[number]
        for key in ("published_on", "description", "tags"):
            if key not in meta:
                raise SystemExit(f"{folder.name}/lesson.toml needs '{key}' to be exported")
        exported[number] = (folder, meta)
    for number in exported:
        post = export(number, args.site, exported)
        print(f"wrote {post.relative_to(args.site)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
