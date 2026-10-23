"""Static concept diagrams as code.

A Figure is a list of elements (chips, boxes, arrows, text) placed on a fixed canvas. The same figure renders
to light and dark SVG with the course palette; tools/render.py turns them into PNGs for the notebooks.

Before rendering, `problems()` checks what a reviewer would otherwise have to catch by eye: elements outside
the canvas and text labels that overlap each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

from tfs.viz.palette import palette

SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"
MONO_ADVANCE = 0.6  # em; exact for IBM Plex Mono
SANS_ADVANCE = 0.56  # em; average for IBM Plex Sans, good enough for overlap checks

# role -> (stroke, fill, text) palette keys
ROLES = {
    "neutral": ("line", "card_alt", "ink"),
    "plain": ("line", "card", "ink"),
    "data": ("data", "data_tint", "data_ink"),
    "enc": ("enc", "enc_tint", "enc_ink"),
    "dec": ("dec", "dec_tint", "dec_ink"),
    "att": ("att", "att_tint", "att_ink"),
}

TEXT_KINDS = {  # size px, weight, family, palette key for color (None: role color)
    "title": (16, 600, SANS, "ink"),
    "subtitle": (14, 600, SANS, "ink"),
    "block": (15, 600, SANS, None),
    "label": (12, 400, MONO, "muted"),
    "note": (13, 400, SANS, "muted"),
    "body": (13, 400, SANS, "ink"),
    "count": (13, 500, MONO, "muted"),
    "count_hot": (13, 600, MONO, "data_ink"),
    "lead": (18, 400, SANS, "ink"),
    "glyph": (46, 400, SANS, "ink"),
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    role: str
    label: str | None = None
    mono: bool = False
    strong: bool = False
    r: float = 9
    size: float = 13


@dataclass
class Arrow:
    pts: list[tuple[float, float]]
    role: str = "muted"
    width: float = 1.6


@dataclass
class Line:
    """A plain connector without an arrowhead, e.g. an edge of a graph."""

    pts: list[tuple[float, float]]
    role: str = "muted"
    width: float = 1.6
    dash: bool = False


@dataclass
class Text:
    x: float
    y: float
    s: str
    kind: str = "label"
    role: str | None = None
    anchor: str = "start"


@dataclass
class Figure:
    w: int
    h: int
    alt: str
    els: list = field(default_factory=list)

    # ---- building ----
    @staticmethod
    def chip_width(s: str, size: float = 13, pad: float | None = None) -> float:
        pad = 0.62 * size if pad is None else pad
        return max(2.2 * size, len(s) * MONO_ADVANCE * size + 2 * pad)

    def chip(self, x, y, s, role="data", strong=False, h=30, size=13) -> float:
        w = self.chip_width(s, size)
        self.els.append(Box(x, y, w, h, role, s, mono=True, strong=strong, r=7, size=size))
        return w

    def row(self, tokens, x, y, role="data", strong=lambda i, t: False, gap=5, h=30, size=13):
        """Chips side by side; returns (x, width) of each chip."""
        spans = []
        for i, t in enumerate(tokens):
            w = self.chip(x, y, t, role, strong(i, t), h=h, size=size)
            spans.append((x, w))
            x += w + gap
        return spans

    def box(self, x, y, w, h, role="neutral", label=None, r=9):
        self.els.append(Box(x, y, w, h, role, label, r=r))

    def arrow(self, *pts, role="muted", width=1.6):
        self.els.append(Arrow(list(pts), role, width))

    def line(self, *pts, role="muted", width=1.6, dash=False):
        self.els.append(Line(list(pts), role, width, dash))

    def text(self, x, y, s, kind="label", role=None, anchor="start"):
        self.els.append(Text(x, y, s, kind, role, anchor))

    # ---- checking ----
    def _text_bbox(self, t: Text) -> tuple[float, float, float, float]:
        size, _, family, _ = TEXT_KINDS[t.kind]
        width = len(t.s) * size * (MONO_ADVANCE if family == MONO else SANS_ADVANCE)
        x0 = {"start": t.x, "middle": t.x - width / 2, "end": t.x - width}[t.anchor]
        return x0, t.y - size * 0.8, x0 + width, t.y + size * 0.25

    def problems(self, margin: float = 4) -> list[str]:
        out = []
        for e in self.els:
            if isinstance(e, Box):
                x0, y0, x1, y1 = e.x, e.y, e.x + e.w, e.y + e.h
            elif isinstance(e, (Arrow, Line)):
                xs, ys = [p[0] for p in e.pts], [p[1] for p in e.pts]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = self._text_bbox(e)
            if x0 < margin or y0 < margin or x1 > self.w - margin or y1 > self.h - margin:
                out.append(f"outside canvas: {e}")
        texts = [e for e in self.els if isinstance(e, Text)]
        boxes = [e for e in self.els if isinstance(e, Box)]
        for t in texts:
            tx0, ty0, tx1, ty1 = self._text_bbox(t)
            for b in boxes:
                bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
                inside = bx0 <= tx0 and tx1 <= bx1 and by0 <= ty0 and ty1 <= by1
                if not inside and tx0 < bx1 and bx0 < tx1 and ty0 < by1 and by0 < ty1:
                    out.append(f"text crosses a box: {t.s!r} and {b.label or 'a box'!r}")
        for i, a in enumerate(texts):
            ax0, ay0, ax1, ay1 = self._text_bbox(a)
            for b in texts[i + 1 :]:
                bx0, by0, bx1, by1 = self._text_bbox(b)
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    out.append(f"text overlap: {a.s!r} and {b.s!r}")
        return out

    # ---- rendering ----
    def to_svg(self, theme: str = "light") -> str:
        c = palette(theme)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-label="{escape(self.alt)}">',
            f"<title>{escape(self.alt)}</title>",
            "<defs>",
        ]
        for name, key in (("muted", "muted"), ("data", "data"), ("enc", "enc"), ("dec", "dec"), ("att", "att")):
            parts.append(
                f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c[key]}"/></marker>'
            )
        parts.append("</defs>")
        parts.append(f'<rect width="{self.w}" height="{self.h}" rx="12" fill="{c["card"]}"/>')
        for e in self.els:
            if isinstance(e, Box):
                stroke, fill, ink = (c[k] for k in ROLES[e.role])
                if e.strong:
                    fill, ink = stroke, c["card"]
                parts.append(
                    f'<rect x="{e.x:.1f}" y="{e.y:.1f}" width="{e.w:.1f}" height="{e.h:.1f}" rx="{e.r}" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
                )
                if e.label:
                    family = MONO if e.mono else SANS
                    parts.append(
                        f'<text x="{e.x + e.w / 2:.1f}" y="{e.y + e.h / 2 + 0.35 * e.size:.1f}" '
                        f'text-anchor="middle" font-family="{family}" font-size="{e.size}" font-weight="500" '
                        f'fill="{ink}" xml:space="preserve">{escape(e.label)}</text>'
                    )
            elif isinstance(e, Arrow):
                key = e.role if e.role in ("data", "enc", "dec", "att") else "muted"
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in e.pts)
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{c[key]}" stroke-width="{e.width}" '
                    f'stroke-linejoin="round" marker-end="url(#arrow-{key})"/>'
                )
            elif isinstance(e, Line):
                color = c["muted"] if e.role == "muted" else c[ROLES[e.role][0]]
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in e.pts)
                dash = ' stroke-dasharray="5 4"' if e.dash else ""
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{e.width}"{dash} '
                    'stroke-linecap="round" stroke-linejoin="round"/>'
                )
            else:
                size, weight, family, key = TEXT_KINDS[e.kind]
                color = c[key] if key else c[ROLES[e.role or "neutral"][0]]
                parts.append(
                    f'<text x="{e.x:.1f}" y="{e.y:.1f}" text-anchor="{e.anchor}" font-family="{family}" '
                    f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(e.s)}</text>'
                )
        parts.append("</svg>")
        return "\n".join(parts)
