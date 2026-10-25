"""Page 1 - Analyze: classify clauses in a contract and highlight risky terms."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import streamlit as st

import ui
from ml import inference
from ml.highlight import build_highlighted_html, esc_text
from ml.paths import MODEL_KEYS, MODEL_LABELS
from ml.risk_map import HIGH, LOW, MEDIUM, RISK_ORDER, SHORT_TAG
from ml.text_utils import available_backends

ui.page_header("Analyze Contract", "Classify clauses, flag risky terms and export the results.")

status = inference.model_status()

# ---------------------------------------------------------------- step 1: input
with ui.window_container("Step 1 - Contract input"):
    text, source_label = ui.contract_input_widget("an")

# ---------------------------------------------------------------- step 2: settings
with ui.window_container("Step 2 - Analysis settings"):
    c1, c2 = st.columns([1, 1])
    with c1:
        model_key = st.radio(
            "Model",
            list(MODEL_KEYS),
            format_func=lambda k: f"{MODEL_LABELS[k]}  [{'TRAINED' if status[k] else 'NOT TRAINED'}]",
            key="an_model",
        )
        if not status[model_key]:
            st.caption("This model is not trained yet. Analysis will use the rule-based DEMO classifier.")
    with c2:
        threshold = st.slider(
            "Confidence threshold",
            0.0, 1.0, 0.5, 0.05,
            key="an_threshold",
            help="Predictions below this confidence are treated as 'None' and are not flagged.",
        )
        with st.expander("Advanced"):
            splitter = st.selectbox("Sentence splitter", available_backends(), key="an_splitter")
    run = st.button("RUN ANALYSIS", type="primary", key="an_run", disabled=not text.strip())
    if not text.strip():
        st.caption("Provide contract text above to enable analysis.")

if run:
    with st.spinner("Splitting sentences and classifying clauses..."):
        try:
            output = inference.analyze_text(text, model_key, threshold, splitter)
            st.session_state["an_output"] = output
            st.session_state["an_text"] = text
            st.session_state["an_params"] = (model_key, threshold, splitter, hash(text))
            st.session_state["an_source_label"] = source_label
        except Exception as exc:  # keep the app usable whatever happens
            st.error(f"Analysis failed: {exc.__class__.__name__}: {exc}")

output: inference.AnalysisOutput | None = st.session_state.get("an_output")
if output is None:
    ui.window("Ready", "<p>Choose a contract and press <b>RUN ANALYSIS</b>. The sample contract is a good first test.</p>")
    st.stop()

analysed_text: str = st.session_state["an_text"]
if st.session_state.get("an_params") != (model_key, threshold, splitter, hash(text)):
    st.info("Input or settings changed since the last run. Press RUN ANALYSIS to refresh the results.")

# ---------------------------------------------------------------- results
if output.demo_mode:
    ui.demo_banner(output.notice)
else:
    st.success(output.notice)

stats = inference.summarize(output.results)
ui.summary_cards(
    [
        ("Total clauses", str(stats["total"]), f"of {len(output.results)} sentences", ""),
        ("[HIGH] risk", str(stats["high"]), "review first", "c-high"),
        ("[MED] risk", str(stats["medium"]), "review next", "c-med"),
        ("[LOW] risk", str(stats["low"]), "informational", "c-low"),
        ("Overall risk score", f"{stats['score']}/100", stats["band"], ""),
    ]
)
st.caption("Score = 15 points per HIGH clause + 6 per MEDIUM + 1 per LOW, capped at 100. It is a simple heuristic, not a legal verdict.")

legend = (
    '<div class="cg-legend">'
    '<span><span class="k k-high"></span><b>[HIGH]</b> red tint</span>'
    '<span><span class="k k-med"></span><b>[MED]</b> diagonal hatch</span>'
    '<span><span class="k k-low"></span><b>[LOW]</b> dotted underline</span>'
    "</div>"
)
ui.window("Contract viewer - risk highlighting", legend + build_highlighted_html(analysed_text, output.results))

# ---------------------------------------------------------------- filterable table
flagged = output.flagged
with ui.window_container("Detected clauses"):
    if not flagged:
        st.info("No clauses were detected at this confidence threshold. Try lowering the threshold.")
    else:
        f1, f2, f3 = st.columns([1, 1, 1])
        levels = f1.multiselect("Risk level", [HIGH, MEDIUM, LOW], default=[HIGH, MEDIUM, LOW], key="an_f_level")
        cats = f2.multiselect("Category", sorted({r.category for r in flagged}), key="an_f_cat", placeholder="All categories")
        query = f3.text_input("Search clause text", key="an_f_q")
        rows = [
            r for r in flagged
            if r.risk_level in levels
            and (not cats or r.category in cats)
            and (not query or query.lower() in r.text.lower())
        ]
        rows.sort(key=lambda r: (-RISK_ORDER[r.risk_level], r.index))
        ui.html_table(
            ["#", "Risk", "Category", "Confidence", "Explanation", "Clause (start)"],
            [
                (
                    r.index + 1,
                    ui.risk_badge(r.risk_level),
                    r.category,
                    f"{r.confidence:.2f}",
                    r.risk_explanation,
                    (r.text[:110] + "...") if len(r.text) > 110 else r.text,
                )
                for r in rows
            ],
            numeric_cols=(0, 3),
            raw_cols=(1,),
            empty_message="No clauses match the current filters.",
        )
        st.caption(f"Showing {len(rows)} of {len(flagged)} detected clauses, highest risk first.")

        # ------------------------------------------------------------ inspector
        if rows:
            by_index = {r.index: r for r in rows}
            pick = st.selectbox(
                "Inspect a clause",
                list(by_index),
                format_func=lambda i: f"#{i + 1} {SHORT_TAG[by_index[i].risk_level]} {by_index[i].category} - {by_index[i].text[:60]}",
                key="an_pick",
            )
            r = by_index[pick]
            ui.html(
                '<dl class="cg-detail">'
                f"<dt>Category</dt><dd>{esc_text(r.category)} &nbsp;{ui.risk_badge(r.risk_level)}</dd>"
                f'<dt>Confidence</dt><dd class="mono">{r.confidence:.4f} (model output: {esc_text(r.raw_category)}; model: {esc_text(r.model)})</dd>'
                f'<dt>Position</dt><dd class="mono">characters {r.start} to {r.end}</dd>'
                f"<dt>Why this risk level</dt><dd>{esc_text(r.risk_explanation)}</dd>"
                f'<dt>Full clause text</dt><dd><div class="cg-quote">{esc_text(r.text)}</div></dd></dl>'
            )

# ---------------------------------------------------------------- export
records = inference.results_to_records(output.results)
buffer = io.StringIO()
writer = csv.DictWriter(
    buffer,
    fieldnames=["clause_no", "category", "risk_level", "confidence", "start", "end", "clause_text", "risk_explanation", "model"],
)
writer.writeheader()
writer.writerows(records)
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source": st.session_state.get("an_source_label", ""),
    "model_used": output.model_used,
    "demo_mode": output.demo_mode,
    "confidence_threshold": output.threshold,
    "summary": stats,
    "disclaimer": ui.DISCLAIMER,
    "clauses": records,
}
with ui.window_container("Export results"):
    e1, e2 = st.columns(2)
    e1.download_button("Download CSV", buffer.getvalue(), "clauseguard_results.csv", "text/csv", key="an_dl_csv")
    e2.download_button("Download JSON", json.dumps(payload, indent=2), "clauseguard_results.json", "application/json", key="an_dl_json")
    st.caption("Exports contain all detected clauses (not only the filtered rows).")
