"""Render a lesson's static figures and covers to SVG and PNG.

A lesson can define lessons/NN_name/figures.py with
    FIGURES = {"name": function returning a tools.figures.lib.Figure}
    COVERS = {"en": function returning the cover SVG, "ru": ...}

Outputs (committed, the notebooks link to them on GitHub):
    assets/figures/NN/<name>-light.{svg,png}, <name>-dark.{svg,png}
    assets/covers/NN-<lang>.{svg,png}

PNGs are rendered at 2x by headless Chrome with the bundled IBM Plex fonts. Set CHROME=/path/to/chrome if it
isn't found automatically.

Usage:
    uv run python tools/render.py 01
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.figures.cover import H as COVER_H  # noqa: E402
from tools.figures.cover import W as COVER_W  # noqa: E402
from tools.figures.cover import check_latin  # noqa: E402

FONTS = ROOT / "src" / "tfs" / "viz" / "fonts"
FACES = [
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular.ttf"),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium.ttf"),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold.ttf"),
    ("IBM Plex Mono", 400, "IBMPlexMono-Regular.ttf"),
    ("IBM Plex Mono", 500, "IBMPlexMono-Medium.ttf"),
]


def chrome() -> str:
    candidates = [
        os.environ.get("CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome not found: set CHROME=/path/to/chrome")


def png(svg: str, out: Path, width: int, height: int) -> None:
    faces = "".join(
        f"@font-face{{font-family:'{fam}';font-weight:{wt};src:url('{(FONTS / f).as_uri()}')}}" for fam, wt, f in FACES
    )
    html = (
        f"<!doctype html><meta charset='utf-8'><style>{faces}html,body{{margin:0;background:transparent}}</style>{svg}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "figure.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                chrome(),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--default-background-color=00000000",
                "--virtual-time-budget=3000",
                f"--window-size={width},{height}",
                f"--screenshot={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )


def load(lesson_dir: Path):
    path = lesson_dir / "figures.py"
    spec = importlib.util.spec_from_file_location(f"figures_{lesson_dir.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_lesson(lesson_dir: Path, assets: Path) -> list[Path]:
    module = load(lesson_dir)
    number = lesson_dir.name[:2]
    written = []
    for name, make in getattr(module, "FIGURES", {}).items():
        fig = make()
        problems = fig.problems()
        if problems:
            raise SystemExit(f"{lesson_dir.name}/{name}:\n  " + "\n  ".join(problems))
        for theme in ("light", "dark"):
            base = assets / "figures" / number / f"{name}-{theme}"
            base.parent.mkdir(parents=True, exist_ok=True)
            svg = fig.to_svg(theme)
            base.with_suffix(".svg").write_text(svg, encoding="utf-8")
            png(svg, base.with_suffix(".png"), fig.w, fig.h)
            written += [base.with_suffix(".svg"), base.with_suffix(".png")]
    for lang, make in getattr(module, "COVERS", {}).items():
        svg = make()
        if lang == "en":
            check_latin(svg)
        base = assets / "covers" / f"{number}-{lang}"
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".svg").write_text(svg, encoding="utf-8")
        png(svg, base.with_suffix(".png"), COVER_W, COVER_H)
        written += [base.with_suffix(".svg"), base.with_suffix(".png")]
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lessons", nargs="+", help="lesson folder prefixes, e.g. 01")
    args = parser.parse_args(argv)
    for d in sorted((ROOT / "lessons").iterdir()):
        if (d / "figures.py").exists() and any(d.name.startswith(p) for p in args.lessons):
            for path in render_lesson(d, ROOT / "assets"):
                print(f"rendered {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
