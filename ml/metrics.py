"""Compute, save and load evaluation metrics (shared by both trainers and the UI)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ml import paths


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metrics dictionary stored in ``metrics.json``.

    Macro averages are computed over classes that actually occur in ``y_true``
    so that absent classes do not silently drag the score to zero.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    labels = list(labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lab: {
            "precision": float(report[lab]["precision"]),
            "recall": float(report[lab]["recall"]),
            "f1": float(report[lab]["f1-score"]),
            "support": int(report[lab]["support"]),
        }
        for lab in labels
    }
    present = [lab for lab in labels if per_class[lab]["support"] > 0]
    n_present = max(1, len(present))
    total = sum(per_class[lab]["support"] for lab in labels)
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": sum(per_class[l]["precision"] for l in present) / n_present,
        "macro_recall": sum(per_class[l]["recall"] for l in present) / n_present,
        "macro_f1": sum(per_class[l]["f1"] for l in present) / n_present,
        "weighted_f1": (sum(per_class[l]["f1"] * per_class[l]["support"] for l in labels) / total) if total else 0.0,
        "n_samples": int(total),
    }
    metrics: dict[str, Any] = {
        "model": model_name,
        "split": "test",
        "labels": labels,
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        metrics.update(extra)
    return metrics


def metrics_path(model_key: str) -> Path:
    """Where ``metrics.json`` for ``model_key`` lives."""
    return paths.model_dir(model_key) / "metrics.json"


def save_metrics(model_key: str, metrics: dict[str, Any]) -> Path:
    """Write metrics to ``models/<model>/metrics.json``."""
    path = metrics_path(model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_metrics(model_key: str) -> dict[str, Any] | None:
    """Load saved metrics, or ``None`` if missing or unreadable (never invents numbers)."""
    path = metrics_path(model_key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"labels", "overall", "per_class", "confusion_matrix"}
    return data if isinstance(data, dict) and required.issubset(data) else None
