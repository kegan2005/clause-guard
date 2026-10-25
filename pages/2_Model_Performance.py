"""Page 2 - Model Performance: shows only metrics read from saved metrics.json files."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import ui
from ml import inference
from ml.metrics import load_metrics, metrics_path
from ml.paths import MODEL_KEYS, MODEL_LABELS, NONE_LABEL

ui.page_header("Model Performance", "Evaluation on held-out test contracts, read from saved metrics files.")

loaded = {key: load_metrics(key) for key in MODEL_KEYS}

TRAIN_HELP = (
    "<p>No evaluation metrics were found for this model. ClauseGuard never invents numbers.</p>"
    "<p>Train the models from the project folder:</p>"
    '<pre class="cg-quote">python -m data.prepare_cuad --cuad-json path/to/CUAD_v1.json\n'
    "python -m ml.train_baseline\n"
    "python -m ml.train_legalbert</pre>"
    "<p>Then reload this page. See README.md for details.</p>"
)

if not any(loaded.values()):
    ui.window("No metrics available", TRAIN_HELP.replace("\n", "&#10;"))
    st.stop()


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _short_label(label: str, max_chars: int = 14) -> str:
    """Shorten axis labels for display only; original labels remain available on hover."""
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1].rstrip() + "…"


def show_model(key: str, metrics: dict) -> None:
    """Cards, F1 chart, per-class table and compact confusion matrix for one model."""
    labels: list[str] = metrics["labels"]
    per_class: dict = metrics["per_class"]
    overall: dict = metrics["overall"]
    name = MODEL_LABELS[key]

    ui.summary_cards(
        [
            ("Accuracy", pct(overall["accuracy"]), "test split", ""),
            ("Macro F1", f"{overall['macro_f1']:.3f}", "classes present in test", ""),
            ("Weighted F1", f"{overall['weighted_f1']:.3f}", "weighted by support", ""),
            ("Test sentences", f"{overall['n_samples']:,}", "held-out contracts", ""),
        ]
    )
    meta = [f"Model: {name}"]
    if metrics.get("trained_at"):
        meta.append(f"trained {metrics['trained_at']}")
    if metrics.get("n_train"):
        meta.append(f"{metrics['n_train']:,} training sentences")
    st.caption(" | ".join(meta) + f" | file: {metrics_path(key).as_posix()}")

    # --- per-category F1 bar chart
    f1_order = sorted(labels, key=lambda l: per_class[l]["f1"])
    fig = go.Figure(
        go.Bar(
            x=[per_class[l]["f1"] for l in f1_order],
            y=f1_order,
            orientation="h",
            marker=dict(color=ui.MAROON, line=dict(color=ui.INK, width=2)),
            text=[f"{per_class[l]['f1']:.2f}" for l in f1_order],
            textposition="outside",
            hovertemplate="%{y}<br>F1 = %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"F1-SCORE BY CLAUSE CATEGORY - {name.upper()}", xaxis_title="F1-score", yaxis_title="")
    fig.update_xaxes(range=[0, 1.1])
    ui.style_figure(fig, height=max(340, 30 * len(labels) + 120))
    st.plotly_chart(fig, use_container_width=True, key=f"perf_f1_{key}")

    # --- precision / recall / F1 / support table
    ui.html_table(
        ["Category", "Precision", "Recall", "F1-score", "Support"],
        [
            (l, f"{per_class[l]['precision']:.3f}", f"{per_class[l]['recall']:.3f}", f"{per_class[l]['f1']:.3f}", f"{per_class[l]['support']:,}")
            for l in labels
        ],
        numeric_cols=(1, 2, 3, 4),
    )

    # --- compact confusion matrix heatmap
    matrix = metrics["confusion_matrix"]
    mode = "Row-normalised (recall)"
    show_cell_values = st.checkbox(
        "Show values inside cells",
        value=False,
        help="For 42 classes, hover over cells for exact values. Turn this on only when needed.",
        key=f"perf_cm_text_{key}",
    )

    z = [[(v / sum(row)) if sum(row) else 0.0 for v in row] for row in matrix]
    hover_values = [[f"{v:.3f}" for v in row] for row in z]
    colorbar_title = "Recall"

    st.caption(
        "Each row is scaled to sum to 1.0 (100%). Dark diagonal cells indicate correct predictions; "
        "off-diagonal cells show which other categories the model confuses with the actual class."
    )

    display_labels = [_short_label(label) for label in labels]
    cell_text = hover_values if show_cell_values else None

    heat = go.Figure(
        go.Heatmap(
            z=z,
            x=display_labels,
            y=display_labels,
            text=cell_text,
            texttemplate="%{text}" if show_cell_values else None,
            textfont=dict(size=5),
            customdata=hover_values,
            colorscale=[[0, ui.SURFACE], [1, ui.MAROON]],
            xgap=1,
            ygap=1,
            colorbar=dict(title=colorbar_title, thickness=12, len=0.78, outlinewidth=0),
            hovertemplate=(
                "Actual: %{customdata[1]}<br>"
                "Predicted: %{customdata[0]}<br>"
                "Value: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Preserve full labels and values in hover data while using shortened axis labels.
    full_hover = []
    for i, actual_label in enumerate(labels):
        row = []
        for j, predicted_label in enumerate(labels):
            row.append([predicted_label, actual_label, hover_values[i][j]])
        full_hover.append(row)
    heat.data[0].customdata = full_hover

    heat.update_layout(
        title=dict(text=f"CONFUSION MATRIX - {name.upper()}", x=0.02, xanchor="left"),
        xaxis_title="Predicted clause category",
        yaxis_title="Actual clause category",
        height=610,
        autosize=True,
        margin=dict(l=115, r=35, t=60, b=155, pad=2),
        font=dict(size=10),
    )
    heat.update_yaxes(
        autorange="reversed",
        tickangle=0,
        tickfont=dict(size=7),
        automargin=True,
        ticks="",
    )
    heat.update_xaxes(
        tickangle=-55,
        tickfont=dict(size=7),
        automargin=True,
        ticklabeloverflow="allow",
        ticks="",
    )

    # Keep styling but override the height and margins after the helper.
    ui.style_figure(heat, height=610)
    heat.update_layout(
        height=610,
        margin=dict(l=115, r=35, t=60, b=155, pad=2),
        autosize=True,
    )
    st.plotly_chart(
        heat,
        use_container_width=True,
        key=f"perf_cm_{key}",
        config={"displaylogo": False, "responsive": True},
    )
    st.caption(
        "Compact view: axis labels are abbreviated to fit all 42 classes. Hover over any cell to see full class names and the exact value."
    )


def show_comparison() -> None:
    """Side-by-side comparison, only when both metrics files exist."""
    a, b = loaded["svm"], loaded["legalbert"]
    if not (a and b):
        missing = MODEL_LABELS["legalbert" if a else "svm"]
        st.info(f"Comparison needs metrics for both models. Metrics for {missing} are missing - train it to enable this view.")
        return
    labels = [l for l in a["labels"] if l in b["labels"]]
    fig = go.Figure()
    for key, m, color in (("svm", a, ui.MAROON), ("legalbert", b, ui.C_LOW)):
        fig.add_bar(
            name=MODEL_LABELS[key], x=labels, y=[m["per_class"][l]["f1"] for l in labels],
            marker=dict(color=color, line=dict(color=ui.INK, width=2)),
            hovertemplate="%{x}<br>F1 = %{y:.3f}<extra>" + MODEL_LABELS[key] + "</extra>",
        )
    fig.update_layout(barmode="group", title="F1-SCORE BY CATEGORY: SVM VS LEGAL-BERT", yaxis_title="F1-score", xaxis_title="")
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(range=[0, 1.05])
    ui.style_figure(fig, height=520)
    st.plotly_chart(fig, use_container_width=True, key="perf_compare")
    rows = []
    for title, field in (("Accuracy", "accuracy"), ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")):
        va, vb = a["overall"].get(field), b["overall"].get(field)
        if va is None or vb is None:
            continue
        winner = "SVM" if va > vb else "Legal-BERT" if vb > va else "tie"
        rows.append((title, f"{va:.3f}", f"{vb:.3f}", winner))
    ui.html_table(["Metric", "SVM", "Legal-BERT", "Higher"], rows, numeric_cols=(1, 2))
    st.caption("Both models are evaluated on the same held-out test contracts only if they were trained from the same prepared dataset.")


tabs = st.tabs([MODEL_LABELS["svm"], MODEL_LABELS["legalbert"], "Comparison"])
for tab, key in zip(tabs[:2], MODEL_KEYS):
    with tab:
        if loaded[key]:
            show_model(key, loaded[key])
        else:
            trained = inference.is_trained(key)
            note = "A trained model exists but has no metrics.json; re-run its training script." if trained else "This model has not been trained."
            ui.window(f"No metrics for {MODEL_LABELS[key]}", f"<p>{note}</p>" + TRAIN_HELP.replace("\n", "&#10;"))
with tabs[2]:
    show_comparison()
