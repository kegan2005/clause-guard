#!/usr/bin/env python
"""Convert CUAD span annotations into sentence-level classification CSVs.

Usage (from the project root)::

    python -m data.prepare_cuad --cuad-json path/to/CUAD_v1.json
    python -m data.prepare_cuad --hf                      # Hugging Face "theatticusproject/cuad-qa"
    python -m data.prepare_cuad --cuad-json CUAD_v1.json --num-categories 20 --include-risk-categories

Outputs (default ``data/processed/``): ``train.csv``, ``val.csv``, ``test.csv``,
``labels.json`` and ``stats.json``.  Contracts are split *by contract*, never by
sentence, so no contract appears in more than one split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.paths import NONE_LABEL, PROCESSED_DIR  # noqa: E402
from ml.risk_map import RISK_MAP, canonical_name  # noqa: E402
from ml.text_utils import split_sentences  # noqa: E402

CSV_FIELDS = ["contract_id", "sentence_id", "start", "end", "label", "text"]
_QUESTION_CATEGORY = re.compile(r'related to "([^"]+)"')


@dataclass
class ContractRecord:
    """One contract: its text and the annotated answer spans per category."""

    contract_id: str
    text: str
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _category_of(qa_id: str, question: str) -> str | None:
    """Extract the clause category from a CUAD question id or question text."""
    if "__" in qa_id:
        return canonical_name(qa_id.rsplit("__", 1)[-1])
    m = _QUESTION_CATEGORY.search(question or "")
    return canonical_name(m.group(1)) if m else None


def parse_cuad_json(path: Path) -> Iterator[ContractRecord]:
    """Yield contracts from the original CUAD (SQuAD-style) JSON file."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for doc in data["data"]:
        title = str(doc.get("title", "contract"))
        for para in doc.get("paragraphs", []):
            spans: dict[str, list[tuple[int, int]]] = {}
            for qa in para.get("qas", []):
                category = _category_of(str(qa.get("id", "")), str(qa.get("question", "")))
                if not category:
                    continue
                for ans in qa.get("answers", []):
                    text, start = ans.get("text", ""), ans.get("answer_start")
                    if text and start is not None:
                        spans.setdefault(category, []).append((int(start), int(start) + len(text)))
            yield ContractRecord(title, para["context"], spans)


def parse_hf_dataset(name: str = "theatticusproject/cuad-qa") -> Iterator[ContractRecord]:
    """Yield contracts from the Hugging Face CUAD question-answering dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The 'datasets' package is required for --hf. Run: pip install datasets") from exc
    ds = load_dataset(name)
    grouped: dict[str, ContractRecord] = {}
    for split in ds.values():
        for row in split:
            title = str(row["title"])
            rec = grouped.setdefault(title, ContractRecord(title, row["context"]))
            category = _category_of(str(row["id"]), str(row.get("question", "")))
            if not category:
                continue
            answers = row["answers"]
            for text, start in zip(answers["text"], answers["answer_start"]):
                if text:
                    rec.spans.setdefault(category, []).append((int(start), int(start) + len(text)))
    yield from grouped.values()


# ---------------------------------------------------------------------------
# Splitting and labelling
# ---------------------------------------------------------------------------
def split_by_contract(ids: Iterable[str], val_frac: float, test_frac: float, seed: int) -> dict[str, set[str]]:
    """Randomly assign whole contracts to train / val / test (no leakage)."""
    unique = sorted(set(ids))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_test = min(n, max(1, round(n * test_frac))) if n >= 3 else 0
    n_val = min(n - n_test, max(1, round(n * val_frac))) if n >= 3 else 0
    return {
        "test": set(unique[:n_test]),
        "val": set(unique[n_test : n_test + n_val]),
        "train": set(unique[n_test + n_val :]),
    }


def top_categories(records: Iterable[ContractRecord], k: int, forced: Iterable[str] = ()) -> list[str]:
    """The ``k`` most frequent categories (by annotated span count), plus any ``forced`` ones."""
    counts: Counter[str] = Counter()
    for rec in records:
        for cat, spans in rec.spans.items():
            counts[cat] += len(spans)
    chosen = [c for c, _ in counts.most_common(k)]
    for cat in forced:
        if cat in counts and cat not in chosen:
            chosen.append(cat)
    return chosen


def label_contract(
    rec: ContractRecord,
    categories: set[str],
    min_overlap_ratio: float = 0.5,
    drop_other_category: bool = False,
    splitter: str = "regex",
) -> list[dict[str, Any]]:
    """Return one labelled row per sentence of ``rec``.

    A sentence gets category C when it overlaps an annotated answer of C by at
    least ``min_overlap_ratio`` of the *shorter* of (sentence, answer); the
    category with the largest overlap wins.  Everything else is ``None``.
    If ``drop_other_category`` is set, sentences that overlap an annotation from a
    category outside ``categories`` are omitted instead of being labelled ``None``.
    """
    rows: list[dict[str, Any]] = []
    for i, sent in enumerate(split_sentences(rec.text, backend=splitter)):
        best_label, best_overlap, other = NONE_LABEL, 0, False
        for cat, spans in rec.spans.items():
            for a, b in spans:
                overlap = min(sent.end, b) - max(sent.start, a)
                if overlap <= 0:
                    continue
                if cat not in categories:
                    other = True
                    continue
                needed = min(sent.end - sent.start, b - a) * min_overlap_ratio
                if overlap >= needed and overlap > best_overlap:
                    best_label, best_overlap = cat, overlap
        if best_label == NONE_LABEL and other and drop_other_category:
            continue
        rows.append(
            {"contract_id": rec.contract_id, "sentence_id": i, "start": sent.start, "end": sent.end, "label": best_label, "text": sent.text}
        )
    return rows


def downsample_none(rows: list[dict[str, Any]], ratio: float | None, rng: random.Random) -> list[dict[str, Any]]:
    """Keep at most ``ratio`` x (number of labelled rows) ``None`` rows, preserving order."""
    if ratio is None:
        return rows
    positives = [r for r in rows if r["label"] != NONE_LABEL]
    negatives = [r for r in rows if r["label"] == NONE_LABEL]
    limit = math.ceil(ratio * len(positives)) if positives else min(len(negatives), 50)
    if len(negatives) > limit:
        keep = set(rng.sample(range(len(negatives)), limit))
        negatives = [r for i, r in enumerate(negatives) if i in keep]
    keep_ids = {id(r) for r in positives} | {id(r) for r in negatives}
    return [r for r in rows if id(r) in keep_ids]


def build_datasets(
    records: list[ContractRecord],
    num_categories: int = 15,
    include_risk_categories: bool = False,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    max_none_ratio: float | None = 3.0,
    full_test: bool = False,
    min_overlap_ratio: float = 0.5,
    drop_other_category: bool = False,
    splitter: str = "regex",
) -> tuple[dict[str, list[dict[str, Any]]], list[str], dict[str, Any]]:
    """Run the whole preparation and return ``(rows_per_split, labels, stats)``."""
    splits = split_by_contract((r.contract_id for r in records), val_frac, test_frac, seed)
    train_records = [r for r in records if r.contract_id in splits["train"]]
    forced = list(RISK_MAP) if include_risk_categories else []
    # Frequencies come from TRAIN contracts only, so label selection never sees val/test.
    categories = top_categories(train_records, num_categories, forced)
    labels = [NONE_LABEL] + sorted(categories)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, ids in splits.items():
        rows: list[dict[str, Any]] = []
        for rec in records:
            if rec.contract_id in ids:
                rows.extend(label_contract(rec, set(categories), min_overlap_ratio, drop_other_category, splitter))
        out[name] = rows if (name == "test" and full_test) else downsample_none(rows, max_none_ratio, rng)
    stats = {
        "n_contracts": {k: len(v) for k, v in splits.items()},
        "n_rows": {k: len(v) for k, v in out.items()},
        "label_counts": {k: dict(Counter(r["label"] for r in v)) for k, v in out.items()},
        "categories": categories,
        "settings": {
            "num_categories": num_categories, "include_risk_categories": include_risk_categories, "seed": seed,
            "max_none_ratio": max_none_ratio, "full_test": full_test, "min_overlap_ratio": min_overlap_ratio,
            "drop_other_category": drop_other_category,
        },
    }
    return out, labels, stats


def write_outputs(out_dir: Path, rows: dict[str, list[dict[str, Any]]], labels: list[str], stats: dict[str, Any]) -> None:
    """Write CSV files, ``labels.json`` and ``stats.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in rows.items():
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(data)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cuad-json", type=Path, help="Path to the original CUAD_v1.json file")
    src.add_argument("--hf", action="store_true", help="Load 'theatticusproject/cuad-qa' with the datasets library")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--num-categories", type=int, default=15, help="Number of most frequent categories to keep (default 15)")
    p.add_argument("--include-risk-categories", action="store_true", help="Also keep every category listed in ml/risk_map.py")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-none-ratio", type=float, default=3.0, help="Max 'None' rows per labelled row in each split (use -1 to keep all)")
    p.add_argument("--full-test", action="store_true", help="Keep ALL 'None' rows in the test split (realistic but larger)")
    p.add_argument("--min-overlap-ratio", type=float, default=0.5)
    p.add_argument("--drop-other-category", action="store_true", help="Drop sentences annotated with categories outside the kept set")
    p.add_argument("--splitter", choices=["regex", "spacy"], default="regex")
    args = p.parse_args(argv)

    if args.cuad_json is not None and not args.cuad_json.is_file():
        raise SystemExit(f"File not found: {args.cuad_json}")
    records = list(parse_cuad_json(args.cuad_json) if args.cuad_json else parse_hf_dataset())
    if len(records) < 3:
        raise SystemExit("Need at least 3 contracts to create train/val/test splits.")
    ratio = None if args.max_none_ratio < 0 else args.max_none_ratio
    rows, labels, stats = build_datasets(
        records, args.num_categories, args.include_risk_categories, args.val_frac, args.test_frac, args.seed,
        ratio, args.full_test, args.min_overlap_ratio, args.drop_other_category, args.splitter,
    )
    write_outputs(args.out_dir, rows, labels, stats)
    print(f"Wrote {args.out_dir} | contracts {stats['n_contracts']} | rows {stats['n_rows']}")
    print("Labels:", ", ".join(labels))


if __name__ == "__main__":
    main()
