#!/usr/bin/env python
"""Train the TF-IDF (word + character n-grams) + LinearSVC baseline.

Run from the project root after preparing the data::

    python -m ml.train_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from ml import paths  # noqa: E402
from ml.metrics import compute_metrics, save_metrics  # noqa: E402


def read_split(data_dir: Path, name: str) -> pd.DataFrame:
    """Read ``<name>.csv``. ``keep_default_na=False`` stops pandas turning the label 'None' into NaN."""
    file = data_dir / f"{name}.csv"
    if not file.is_file():
        raise SystemExit(f"Missing {file}. Run: python -m data.prepare_cuad --cuad-json path/to/CUAD_v1.json")
    return pd.read_csv(file, keep_default_na=False, dtype={"label": str, "text": str})


def build_pipeline(c: float, min_df: int, word_features: int, char_features: int) -> Pipeline:
    """Word 1-2 grams + character 3-5 grams (char_wb) -> class-weighted LinearSVC."""
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, strip_accents="unicode", max_features=word_features)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True, max_features=char_features)),
        ]
    )
    return Pipeline([("tfidf", features), ("svm", LinearSVC(C=c, class_weight="balanced", max_iter=5000))])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=paths.PROCESSED_DIR)
    p.add_argument("--out", type=Path, default=None, help="Output folder (default models/svm)")
    p.add_argument("--C", type=float, default=0.5)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--word-features", type=int, default=200_000)
    p.add_argument("--char-features", type=int, default=300_000)
    args = p.parse_args(argv)

    labels = json.loads((args.data_dir / "labels.json").read_text(encoding="utf-8"))
    train, val, test = (read_split(args.data_dir, n) for n in ("train", "val", "test"))
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}  labels={len(labels)}")

    model = build_pipeline(args.C, args.min_df, args.word_features, args.char_features)
    model.fit(train["text"], train["label"])

    val_f1 = float(f1_score(val["label"], model.predict(val["text"]), average="macro", zero_division=0))
    pred = model.predict(test["text"])
    metrics = compute_metrics(
        test["label"].tolist(), pred.tolist(), labels, "svm",
        extra={"n_train": int(len(train)), "n_val": int(len(val)), "val_macro_f1": val_f1,
               "hyperparameters": {"C": args.C, "class_weight": "balanced", "min_df": args.min_df}},
    )

    out = args.out or paths.model_dir("svm")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    if args.out is None:
        save_metrics("svm", metrics)
    else:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    o = metrics["overall"]
    print(f"Saved model to {out}\nTest accuracy={o['accuracy']:.3f}  macro-F1={o['macro_f1']:.3f}  weighted-F1={o['weighted_f1']:.3f}  (val macro-F1={val_f1:.3f})")


if __name__ == "__main__":
    main()
