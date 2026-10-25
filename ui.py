"""Shared UI helpers: CSS injection, windows, badges, banners, tables, status bar and charts.

Every HTML string built here is a *single line* (no raw newlines) so that
Streamlit's Markdown renderer treats it as one HTML block.  All dynamic text is
passed through :func:`ml.highlight.esc_text`.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable, Iterator, Sequence

import streamlit as st

from ml import inference
from ml.documents import DocumentError, extract_text, normalize_text
from ml.highlight import esc_text
from ml.paths import CSS_FILE, MODEL_KEYS, MODEL_LABELS, SAMPLE_CONTRACT
from ml.risk_map import HIGH, LOW, MEDIUM, SHORT_TAG

# Palette (mirrors assets/style.css)
DESKTOP, SURFACE, INK, MAROON = "#E6DCC8", "#F3ECDC", "#1A1210", "#5A0F1E"
C_HIGH, C_MED, C_LOW, C_OFF = "#B3202A", "#B8863B", "#5E6B3A", "#9B9283"

DISCLAIMER = "This tool assists review and is not legal advice."


def one_line(markup: str) -> str:
    """Collapse raw newlines so Markdown cannot break an HTML block."""
    return re.sub(r"\s*\n\s*", " ", markup).strip()


def html(markup: str) -> None:
    """Render trusted HTML (already escaped by the caller)."""
    st.markdown(one_line(markup), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject ``assets/style.css`` (minified so blank lines cannot end the style block)."""
    try:
        css = CSS_FILE.read_text(encoding="utf-8")
    except OSError:
        st.warning("assets/style.css is missing; the default Streamlit style is being used.")
        return
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Windows, headers, banners
# ---------------------------------------------------------------------------
def _titlebar(title: str) -> str:
    return (
        f'<div class="cg-titlebar"><span>&#9632; {esc_text(title)}</span>'
        '<span class="cg-ctl" aria-hidden="true"><i>_</i><i>&#9633;</i><i>x</i></span></div>'
    )


def window_html(title: str, body_html: str) -> str:
    """HTML for a window whose body is already-safe HTML."""
    return f'<div class="cg-win">{_titlebar(title)}<div class="cg-body">{body_html}</div></div>'


def window(title: str, body_html: str) -> None:
    """Render a window with an HTML body."""
    html(window_html(title, body_html))


@contextmanager
def window_container(title: str) -> Iterator[None]:
    """A window that can hold normal Streamlit widgets."""
    with st.container(border=True):
        html(_titlebar(title))
        yield


def page_header(title: str, subtitle: str) -> None:
    html(f'<div class="cg-pagehead"><h1>{esc_text(title)}</h1><p>{esc_text(subtitle)}</p></div>')


def demo_banner(message: str = "") -> None:
    """Visible DEMO MODE banner."""
    extra = esc_text(message) if message else "No trained model found. Results come from a simple rule-based demo classifier."
    html(f'<div class="cg-banner" role="alert"><b>DEMO MODE</b>{extra}</div>')


def disclaimer() -> None:
    html(f'<div class="cg-disclaimer" role="note">{esc_text(DISCLAIMER)}</div>')


def risk_badge(level: str) -> str:
    """Return a ``[HIGH]``/``[MED]``/``[LOW]`` badge (text label plus colour)."""
    css = {HIGH: "high", MEDIUM: "med", LOW: "low"}.get(level, "none")
    return f'<span class="badge badge-{css}">{SHORT_TAG.get(level, "[---]")}</span>'


def summary_cards(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """Render cards given as ``(label, value, sublabel, css_class)`` tuples."""
    items = "".join(
        f'<div class="cg-card {css}"><div class="cg-card-h">{esc_text(label)}</div>'
        f'<div class="cg-card-v">{esc_text(value)}</div><div class="cg-card-s">{esc_text(sub)}</div></div>'
        for label, value, sub, css in cards
    )
    html(f'<div class="cg-cards">{items}</div>')


def html_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    numeric_cols: Sequence[int] = (),
    raw_cols: Sequence[int] = (),
    empty_message: str = "No rows to display.",
) -> None:
    """Render a table.  Cells are escaped unless their column index is in ``raw_cols`` (trusted HTML)."""
    head = "".join(f'<th class="{"num" if i in numeric_cols else ""}">{esc_text(h)}</th>' for i, h in enumerate(headers))
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i in numeric_cols else ""}">{cell if i in raw_cols else esc_text(str(cell))}</td>'
            for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body = [f'<tr><td colspan="{len(headers)}">{esc_text(empty_message)}</td></tr>']
    html(f'<div class="cg-tablewrap"><table class="cg-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Sidebar status panel and bottom status bar
# ---------------------------------------------------------------------------
def sidebar_status() -> None:
    """Small model-availability panel under the navigation."""
    status = inference.model_status()
    lines = "".join(
        f"<p><b>{esc_text(MODEL_LABELS[k])}</b><br>"
        f'<span class="badge {"badge-ok" if status[k] else "badge-none"}">{"TRAINED" if status[k] else "NOT TRAINED"}</span></p>'
        for k in MODEL_KEYS
    )
    with st.sidebar:
        window("MODEL STATUS", lines)
        html(f'<div class="cg-body"><small>{esc_text(DISCLAIMER)}</small></div>')


def status_bar(message: str = "Ready") -> None:
    """Windows-style status bar fixed to the bottom of the screen."""
    status = inference.model_status()
    models = " ".join(f"{k.upper()}: {'OK' if status[k] else '--'}" for k in MODEL_KEYS)
    mode = "TRAINED MODELS AVAILABLE" if any(status.values()) else "DEMO MODE (no trained models)"
    html(
        '<div class="cg-status" role="status">'
        f'<span class="grow">{esc_text(message)}</span>'
        f'<span class="mono hide-sm">{esc_text(models)}</span>'
        f'<span class="hide-sm">{esc_text(mode)}</span>'
        f'<span class="mono">{datetime.now():%Y-%m-%d}</span>'
        f"<span>{esc_text(DISCLAIMER.upper())}</span></div>"
    )


# ---------------------------------------------------------------------------
# Contract input widget (shared by Analyze and Compare Models)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def _parse_upload(data: bytes, name: str) -> str:
    return extract_text(data, name)


def load_sample_contract() -> str:
    """Read the bundled sample contract."""
    try:
        return normalize_text(SAMPLE_CONTRACT.read_text(encoding="utf-8"))
    except OSError:
        st.error("data/sample_contract.txt is missing.")
        return ""


def contract_input_widget(prefix: str) -> tuple[str, str]:
    """Let the user paste, upload or load a sample.  Returns ``(text, source_description)``."""
    source = st.radio(
        "Input source",
        ["Paste text", "Upload file", "Sample contract"],
        horizontal=True,
        key=f"{prefix}_source",
    )
    text, label = "", ""
    if source == "Paste text":
        text = st.text_area(
            "Contract text",
            height=240,
            key=f"{prefix}_paste",
            placeholder="Paste the contract text here...",
        )
        label = "pasted text"
    elif source == "Upload file":
        up = st.file_uploader("PDF, DOCX or TXT file", type=["pdf", "docx", "txt"], key=f"{prefix}_upload")
        if up is not None:
            try:
                text = _parse_upload(up.getvalue(), up.name)
                label = up.name
                st.caption(f"Loaded {up.name}: {len(text):,} characters.")
            except DocumentError as exc:
                st.error(str(exc))
            except Exception as exc:  # unexpected parser failure: never crash the page
                st.error(f"Could not read {up.name}: {exc.__class__.__name__}")
    else:
        text = load_sample_contract()
        label = "sample contract"
        if text:
            with st.expander("Preview sample contract"):
                st.code(text[:2500] + ("..." if len(text) > 2500 else ""), language=None)
    return normalize_text(text), label


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def style_figure(fig: Any, height: int | None = None) -> Any:
    """Apply the ClauseGuard palette and fonts to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor="#FBF8F0",
        font=dict(family="Tahoma, Segoe UI, sans-serif", color=INK, size=12),
        title_font=dict(family="Tahoma, Segoe UI, sans-serif", size=15, color=MAROON),
        colorway=[MAROON, C_LOW, C_MED, C_HIGH, C_OFF],
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor=SURFACE, bordercolor=INK, borderwidth=2),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(family="Courier New, monospace", color=INK), bordercolor=INK),
    )
    axis = dict(linecolor=INK, linewidth=2, gridcolor="#D9CFB9", zerolinecolor=INK, tickfont=dict(family="Courier New, monospace"))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if height:
        fig.update_layout(height=height)
    return fig
