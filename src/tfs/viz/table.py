"""Show a pandas DataFrame as a clean HTML table: no index, wrapped text, numbers aligned on the right."""

from __future__ import annotations

from html import escape
from numbers import Integral, Real

from .palette import palette

FONT = "'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


class Table:
    """Displays itself as HTML in a notebook; `print()` shows plain text."""

    def __init__(self, frame, digits: int = 2, theme: str = "light"):
        self.frame = frame
        self.digits = digits
        self.theme = theme

    def _cell(self, value) -> tuple[str, bool]:
        if isinstance(value, bool):
            return str(value), False
        if isinstance(value, Integral):
            return f"{value:,}", True
        if isinstance(value, Real):
            return f"{value:,.{self.digits}f}", True
        return escape(str(value)), False

    def _repr_html_(self) -> str:
        c = palette(self.theme)
        cell = f"padding:7px 14px;border-bottom:1px solid {c['grid']};vertical-align:top;"
        head = (
            f"padding:7px 14px;border-bottom:2px solid {c['line']};font-weight:600;color:{c['muted']};"
            "font-size:12px;letter-spacing:0.02em;"
        )
        numeric = [self.frame[col].dtype.kind in "iuf" for col in self.frame.columns]
        parts = [
            f'<table class="tfs-table" style="border-collapse:collapse;font-family:{FONT};font-size:13.5px;'
            f'line-height:1.45;color:{c["ink"]};background:{c["card"]};margin:4px 0 8px;">',
            "<thead><tr>",
        ]
        for col, is_num in zip(self.frame.columns, numeric, strict=True):
            align = "right" if is_num else "left"
            parts.append(f'<th style="{head}text-align:{align};">{escape(str(col))}</th>')
        parts.append("</tr></thead><tbody>")
        for row in self.frame.itertuples(index=False):
            parts.append("<tr>")
            for value in row:
                text, is_num = self._cell(value)
                style = "text-align:right;font-variant-numeric:tabular-nums;" if is_num else "text-align:left;"
                parts.append(f'<td style="{cell}{style}max-width:460px;white-space:normal;">{text}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    def __repr__(self) -> str:
        return self.frame.to_string(index=False)


def table(frame, digits: int = 2) -> Table:
    """Show a DataFrame without its index, e.g. `table(train.sample(5))`. Floats get `digits` decimals."""
    return Table(frame, digits)
