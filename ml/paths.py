"""Central place for filesystem locations and model identifiers.

``MODELS_DIR`` is read at *call time* by :func:`model_dir`, so tests can
monkeypatch ``ml.paths.MODELS_DIR`` to point at an empty temporary folder.
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = ROOT / "models"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
SAMPLE_CONTRACT: Path = ROOT / "data" / "sample_contract.txt"
CSS_FILE: Path = ROOT / "assets" / "style.css"

#: Internal model keys, in display order.
MODEL_KEYS: tuple[str, ...] = ("svm", "legalbert")

#: Human readable model names.
MODEL_LABELS: dict[str, str] = {
    "svm": "SVM (TF-IDF + LinearSVC)",
    "legalbert": "Legal-BERT",
}

#: Folder names under ``MODELS_DIR``.
_MODEL_FOLDERS: dict[str, str] = {"svm": "svm", "legalbert": "legalbert_v1"}
#: Label used for sentences that are not a review-relevant clause.
NONE_LABEL: str = "None"


def model_dir(key: str) -> Path:
    """Return the folder that holds the artefacts of model ``key``."""
    if key not in _MODEL_FOLDERS:
        raise KeyError(f"Unknown model key: {key!r}")
    return MODELS_DIR / _MODEL_FOLDERS[key]
