"""Page 3 - Compare Models: run one contract through both trained models."""
from __future__ import annotations

import streamlit as st

import ui
from ml import inference
from ml.paths import MODEL_KEYS, MODEL_LABELS, NONE_LABEL
from ml.risk_map import RISK_ORDER
from ml.text_utils import available_backends

ui.page_header("Compare Models", "Run the same contract through SVM and Legal-BERT and inspect disagreements.")
status = inference.model_status()

ui.window(
    "Model availability",
    "".join(
        f"<p><b>{MODEL_LABELS[k]}</b>: "
        f'<span class="badge {"badge-ok" if status[k] else "badge-none"}">{"TRAINED" if status[k] else "NOT TRAINED"}</span></p>'
        for k in MODEL_KEYS
    ),
)

if not any(status.values()):
    ui.window(
        "Nothing to compare yet",
        "<p>Neither model has been trained, so there are no real predictions to compare. "
        "The rule-based demo classifier is deliberately not used here.</p>"
        '<p>Train both models:</p><pre class="cg-quote">python -m data.prepare_cuad --cuad-json path/to/CUAD_v1.json&#10;'
        "python -m ml.train_baseline&#10;python -m ml.train_legalbert</pre>"
        "<p>You can still use the <b>Analyze</b> page in demo mode.</p>",
    )
    st.stop()
if not all(status.values()):
    missing = [MODEL_LABELS[k] for k in MODEL_KEYS if not status[k]]
    st.warning(f"{', '.join(missing)} is not trained. Only the available model's predictions are shown; train both to see disagreements.")

with ui.window_container("Contract input"):
    text, _ = ui.contract_input_widget("cmp")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, key="cmp_threshold")
    splitter = st.selectbox("Sentence splitter", available_backends(), key="cmp_splitter")
    run = st.button("COMPARE MODELS", type="primary", key="cmp_run", disabled=not text.strip())

if run:
    outputs: dict[str, inference.AnalysisOutput] = {}
    errors: dict[str, str] = {}
    with st.spinner("Running both models..."):
        for key in MODEL_KEYS:
            if not status[key]:
                continue
            try:
                outputs[key] = inference.analyze_text(text, key, threshold, splitter, allow_demo=False)
            except Exception as exc:
                errors[key] = f"{exc.__class__.__name__}: {exc}"
    st.session_state["cmp_outputs"], st.session_state["cmp_errors"] = outputs, errors

outputs = st.session_state.get("cmp_outputs")
if not outputs and not st.session_state.get("cmp_errors"):
    st.info("Choose a contract and press COMPARE MODELS.")
    st.stop()
for key, message in st.session_state.get("cmp_errors", {}).items():
    st.error(f"{MODEL_LABELS[key]} failed: {message}")
if not outputs:
    st.stop()


def snippet(s: str, n: int = 90) -> str:
    return s if len(s) <= n else s[:n] + "..."


svm, bert = outputs.get("svm"), outputs.get("legalbert")
if svm and bert:
    pairs = list(zip(svm.results, bert.results))  # same splitter => identical sentence list
    flagged_either = [(a, b) for a, b in pairs if a.is_flagged or b.is_flagged]
    disagree = [(a, b) for a, b in flagged_either if a.category != b.category]
    agree_n = len(flagged_either) - len(disagree)
    rate = (agree_n / len(flagged_either) * 100) if flagged_either else 0.0
    risk_gap = sum(1 for a, b in disagree if a.risk_level != b.risk_level)
    ui.summary_cards(
        [
            ("Sentences", str(len(pairs)), "analysed", ""),
            ("Flagged by either", str(len(flagged_either)), "clause candidates", ""),
            ("Agreement", f"{rate:.0f}%", f"{agree_n} of {len(flagged_either)}", ""),
            ("Disagreements", str(len(disagree)), "different category", "c-high" if disagree else ""),
            ("Risk-level gaps", str(risk_gap), "disagree on risk", "c-med" if risk_gap else ""),
        ]
    )
    only_diff = st.checkbox("Show disagreements only", value=False, key="cmp_only_diff")
    shown = disagree if only_diff else flagged_either
    shown = sorted(shown, key=lambda p: (-max(RISK_ORDER[p[0].risk_level], RISK_ORDER[p[1].risk_level]), p[0].index))
    ui.html_table(
        ["#", "Clause (start)", "SVM", "SVM conf.", "SVM risk", "Legal-BERT", "BERT conf.", "BERT risk", "Match"],
        [
            (
                a.index + 1, snippet(a.text), a.category, f"{a.confidence:.2f}", ui.risk_badge(a.risk_level),
                b.category, f"{b.confidence:.2f}", ui.risk_badge(b.risk_level),
                "[SAME]" if a.category == b.category else "[DIFF]",
            )
            for a, b in shown
        ],
        numeric_cols=(0, 3, 6),
        raw_cols=(4, 7),
        empty_message="No disagreements: both models predicted the same categories.",
    )
    st.caption("Confidence values are model scores; the SVM score is a softmax of its margins and is not a calibrated probability.")
else:
    key, out = next(iter(outputs.items()))
    st.info(f"Showing {MODEL_LABELS[key]} only. Train the other model to compare.")
    ui.html_table(
        ["#", "Clause (start)", "Category", "Confidence", "Risk"],
        [(r.index + 1, snippet(r.text), r.category, f"{r.confidence:.2f}", ui.risk_badge(r.risk_level)) for r in out.flagged],
        numeric_cols=(0, 3),
        raw_cols=(4,),
    )
