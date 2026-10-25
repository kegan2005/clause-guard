"""Safe HTML rendering of a contract with risk highlighting.

All user text passes through :func:`esc_text`, so contract content can never
inject markup.  Highlights are inserted using the exact character offsets of
each clause, so the visible text is the original text.
"""
from __future__ import annotations

import html
from typing import Iterable

from ml.inference import ClauseResult
from ml.risk_map import HIGH, LOW, MEDIUM, SHORT_TAG

_CSS_CLASS = {HIGH: "high", MEDIUM: "med", LOW: "low"}


def esc_text(text: str) -> str:
    """Escape ``text`` for HTML.

    Newlines become ``&#10;`` and ``$`` becomes ``&#36;`` so the result is a
    single line that Streamlit's Markdown renderer leaves untouched (blank lines
    would otherwise end an HTML block, and ``$..$`` would be parsed as LaTeX).
    """
    return (
        html.escape(text.replace("\r", ""), quote=True)
        .replace("\n", "&#10;")
        .replace("$", "&#36;")
    )


def build_highlighted_html(text: str, results: Iterable[ClauseResult]) -> str:
    """Return the contract as HTML with ``[HIGH]``/``[MED]``/``[LOW]`` highlighted clauses."""
    flagged = sorted(
        (r for r in results if r.is_flagged and r.risk_level in _CSS_CLASS),
        key=lambda r: r.start,
    )
    parts: list[str] = []
    pos = 0
    for r in flagged:
        if r.start < pos or r.end > len(text):  # skip overlaps / out-of-range defensively
            continue
        parts.append(esc_text(text[pos : r.start]))
        css = _CSS_CLASS[r.risk_level]
        tip = f"{r.category} | {r.risk_level} | confidence {r.confidence:.2f} | {r.risk_explanation}"
        parts.append(
            f'<span class="hl hl-{css}" id="clause-{r.index + 1}" title="{esc_text(tip)}">'
            f'<span class="hl-tag tag-{css}" aria-label="{r.risk_level} risk">{SHORT_TAG[r.risk_level]}</span>'
            f"{esc_text(text[r.start : r.end])}</span>"
        )
        pos = r.end
    parts.append(esc_text(text[pos:]))
    return '<div class="contract-view" tabindex="0">' + "".join(parts) + "</div>"
