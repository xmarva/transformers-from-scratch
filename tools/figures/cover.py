"""Lesson covers, 1200x630: notebook tab, title, hook question and a motif that shows what the lesson is about.

Dark background and the site's red, like xmarva.github.io. English covers must stay Latin-only (readers can't
read Cyrillic); `check_latin` enforces it. The motif area is x in [MOTIF_X0, MOTIF_X1]; helpers report the
right edge they reached so a lesson's cover can assert it fits.
"""

from __future__ import annotations

import re
from html import escape

from tfs.viz.palette import BRAND

W, H = 1200, 630
MOTIF_X0, MOTIF_X1 = 616, 1128
BG, INK, MUTED, RED, RED_TINT = BRAND["bg"], BRAND["ink"], BRAND["muted"], BRAND["red"], BRAND["red_tint"]
GRID, FAINT, LINE, CHIP_INK, CHIP_FILL = "#232327", "#6E7380", "#4A4D55", "#C9CCD3", "#1E1F23"
MONO, SANS = "'IBM Plex Mono', monospace", "'IBM Plex Sans', sans-serif"
COURSE = "Transformers: from scratch to production"


def text(x, y, s, size, color, family=SANS, weight=400, anchor="start", extra="") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" {extra}>{escape(s)}</text>'
    )


def label(x, y, s) -> str:
    return text(x, y, s, 15, FAINT, MONO, extra='letter-spacing="1"')


def chip_w(s: str, size: float = 18, pad: float = 12) -> float:
    return round(len(s) * 0.6 * size + 2 * pad)


def chip(x, y, s, size=18, kind="plain", h=40, pad=12) -> tuple[str, float]:
    """kind: plain | match (matched, brighter) | hot (filled red) | warm (red outline)."""
    w = chip_w(s, size, pad)
    stroke, fill, color = {
        "plain": (LINE, BG, CHIP_INK),
        "match": ("#7C808A", CHIP_FILL, INK),
        "hot": (RED, RED, BG),
        "warm": (RED, RED_TINT, RED),
    }[kind]
    out = (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="7" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    )
    out += text(x + w / 2, y + h / 2 + size * 0.36, s, size, color, MONO, 500, "middle")
    return out, w


def morph_chip(x, y, parts: tuple[str, str, str], size=18, h=40) -> tuple[str, float]:
    """Plain chip whose middle part, a grammatical marker, is drawn in red: (prefix, marker, suffix)."""
    prefix, marker, suffix = parts
    w = chip_w(prefix + marker + suffix, size)
    out = (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="7" '
        f'fill="{BG}" stroke="{LINE}" stroke-width="1.6"/>'
    )
    out += (
        f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + size * 0.36:.1f}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="500" fill="{CHIP_INK}" text-anchor="middle">{escape(prefix)}'
        f'<tspan fill="{RED}">{escape(marker)}</tspan>{escape(suffix)}</text>'
    )
    return out, w


def chip_row(tokens, x, y, size=18, gap=8, kinds=None, pad=12) -> tuple[str, list[tuple[float, float]], float]:
    """Tokens are strings or (prefix, marker, suffix) tuples. Returns svg, [(x, w)] per chip, right edge."""
    parts, spans = [], []
    for i, t in enumerate(tokens):
        if isinstance(t, tuple):
            svg, w = morph_chip(x, y, t, size)
        else:
            svg, w = chip(x, y, t, size, (kinds or {}).get(i, "plain"), pad=pad)
        parts.append(svg)
        spans.append((x, w))
        x += w + gap
    return "".join(parts), spans, x - gap


def fits(right_edge: float) -> None:
    if right_edge > MOTIF_X1:
        raise ValueError(f"motif reaches x={right_edge:.0f}, beyond the right margin {MOTIF_X1}")


def frame(tab: str, title_lines: list[str], hook_lines: list[str], motif: str) -> str:
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        '<defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">'
        f'<circle cx="2" cy="2" r="1.2" fill="{GRID}"/></pattern></defs>',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<rect x="600" width="{W - 600}" height="{H}" fill="url(#dots)"/>',
    ]
    tw = chip_w(tab, 18, 16) + 22
    s.append(f'<rect x="72" y="64" width="{tw}" height="40" rx="8" fill="none" stroke="{LINE}" stroke-width="1.4"/>')
    s.append(f'<circle cx="92" cy="84" r="5" fill="{RED}"/>')
    s.append(text(106, 90, tab, 18, MUTED, MONO))
    y = 214
    for line in title_lines:
        s.append(text(68, y, line, 66, INK, SANS, 600, extra='letter-spacing="-1"'))
        y += 74
    y += 10
    for line in hook_lines:
        s.append(text(72, y, line, 26, MUTED))
        y += 36
    s.append(f'<rect x="72" y="548" width="28" height="3" fill="{RED}"/>')
    s.append(text(72, 580, COURSE, 17, FAINT, MONO))
    s.append(motif)
    s.append("</svg>")
    return "\n".join(s)


def check_latin(svg: str) -> None:
    """English covers must not contain Cyrillic: their readers can't read it."""
    found = re.findall(r"[Ѐ-ӿ]+", svg)
    if found:
        raise ValueError(f"Cyrillic on an English cover: {found}")
