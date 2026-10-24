"""Export an article (articles/<name>/post.md) as a post of xmarva.github.io.

post.md is the post as it will be published: front matter, then markdown with $...$ math and placeholders
[[figure-name]] for the article's figures and charts (articles/<name>/figures.py). On the way to the site:
- every placeholder becomes the light and the dark version of the picture, with the figure's alt text;
- inline math $...$ becomes $$...$$, which kramdown renders as inline math;
- citations in parentheses become numbers that lead to the references (tools/citations.py);
- long paragraphs are split at sentence boundaries into pieces of one to three sentences (tools/paragraphs.py).

The pictures themselves are rendered by tools/render.py --article <name> --out <site>/<assets>.

Usage:
    uv run python tools/export_article.py inference_optimization --site ../../websites/xmarva.github.io \\
        --post _posts/2025-06-19-inference-optimization.md --assets assets/posts/inference-optimization
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_post  # noqa: E402
import render  # noqa: E402

import citations  # noqa: E402
import paragraphs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"^\[\[([\w-]+)\]\]$", re.M)
WIDTH = 760  # px: the figures are drawn 760 wide, the charts scaled to match


def export(name: str, site: Path, post: Path, assets: str) -> Path:
    module = render.load(ROOT / "articles" / name)
    figures, charts = getattr(module, "FIGURES", {}), getattr(module, "CHARTS", {})
    chart_alts = module.chart_alts() if charts else {}
    source = (ROOT / "articles" / name / "post.md").read_text(encoding="utf-8")
    front, body = re.match(r"\A(---\n.*?\n---\n)(.*)\Z", source, re.S).groups()

    def picture(match: re.Match) -> str:
        key = match.group(1)
        if key in figures:
            alt = figures[key]().alt
        elif key in chart_alts:
            alt = chart_alts[key]
        else:
            raise SystemExit(f"{name}/post.md: no figure or chart named {key!r}")
        shown = []
        for theme in ("light", "dark"):
            path = f"{assets}/{key}-{theme}.png"
            if not (site / path).exists():
                raise SystemExit(f"{path} is missing; render the article first")
            shown.append(f'<img class="tfs-figure tfs-{theme}" src="/{path}" width="{WIDTH}" '
                         f'alt="{html.escape(alt)}" loading="lazy">')
        return "\n".join(shown)

    body = paragraphs.shorten(citations.numbered(PLACEHOLDER.sub(picture, body)))
    out = site / post
    out.write_text(front + export_post.inline_math(body), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("article", help="folder name under articles/")
    parser.add_argument("--site", type=Path, required=True, help="path to the xmarva.github.io checkout")
    parser.add_argument("--post", type=Path, required=True, help="the post's path inside the site")
    parser.add_argument("--assets", required=True, help="the folder of the article's pictures inside the site")
    args = parser.parse_args(argv)
    print("wrote", export(args.article, args.site, args.post, args.assets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
