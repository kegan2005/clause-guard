"""Tests for CUAD preparation (synthetic data) and the SVM training/inference round trip."""
import json

import pytest

from data.prepare_cuad import ContractRecord, build_datasets, parse_cuad_json, write_outputs


def make_records(n: int = 12) -> list[ContractRecord]:
    records = []
    for i in range(n):
        gov = f"This Agreement shall be governed by the laws of State {i}."
        nc = "Customer shall not engage in any business that competes with Supplier."
        filler = "The parties will meet each quarter to review progress."
        text = f"{filler} {gov} {filler} {nc} {filler}"
        spans = {
            "Governing Law": [(text.index(gov), text.index(gov) + len(gov))],
            "Non-Compete": [(text.index(nc), text.index(nc) + len(nc))],
        }
        records.append(ContractRecord(f"contract_{i}", text, spans))
    return records


def test_split_is_by_contract_without_leakage():
    rows, labels, stats = build_datasets(make_records(), num_categories=2, max_none_ratio=None)
    ids = {name: {r["contract_id"] for r in data} for name, data in rows.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert all(ids.values())  # every split is non-empty
    assert labels[0] == "None" and set(labels[1:]) == {"Governing Law", "Non-Compete"}


def test_labels_follow_overlap():
    rows, _, _ = build_datasets(make_records(), num_categories=2, max_none_ratio=None)
    for data in rows.values():
        for r in data:
            if "governed by" in r["text"]:
                assert r["label"] == "Governing Law"
            elif "competes" in r["text"]:
                assert r["label"] == "Non-Compete"
            else:
                assert r["label"] == "None"


def test_num_categories_is_configurable():
    _, labels, _ = build_datasets(make_records(), num_categories=1, max_none_ratio=None)
    assert len(labels) == 2  # "None" + 1 category


def test_none_downsampling():
    rows, _, _ = build_datasets(make_records(), num_categories=2, max_none_ratio=1.0)
    for data in rows.values():
        positives = sum(r["label"] != "None" for r in data)
        negatives = sum(r["label"] == "None" for r in data)
        assert negatives <= positives


def test_parse_cuad_json_and_csv_output(tmp_path):
    text = "Intro text here. This Agreement is governed by the laws of Ohio. End of text."
    start = text.index("This Agreement")
    answer = "This Agreement is governed by the laws of Ohio."
    data = {"data": [{"title": "c1", "paragraphs": [{"context": text, "qas": [
        {"id": "c1__Governing Law", "question": 'Highlight the parts related to "Governing Law"', "answers": [{"text": answer, "answer_start": start}]},
        {"id": "c1__Parties", "question": "x", "answers": []},
    ]}]}]}
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    recs = list(parse_cuad_json(path))
    assert recs[0].spans == {"Governing Law": [(start, start + len(answer))]}
    write_outputs(tmp_path / "out", {"train": [], "val": [], "test": []}, ["None"], {})
    assert (tmp_path / "out" / "train.csv").is_file() and (tmp_path / "out" / "labels.json").is_file()


def test_baseline_training_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    pytest.importorskip("joblib")
    from ml import inference, paths
    from ml.metrics import load_metrics
    from ml.train_baseline import main as train_main

    rows, labels, stats = build_datasets(make_records(30), num_categories=2, max_none_ratio=None)
    write_outputs(tmp_path / "processed", rows, labels, stats)
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    train_main(["--data-dir", str(tmp_path / "processed"), "--min-df", "1"])

    assert inference.model_status()["svm"] is True
    metrics = load_metrics("svm")
    assert metrics and set(metrics["labels"]) == set(labels)
    out = inference.analyze_text("Customer shall not engage in any business that competes with Supplier.", "svm", 0.0)
    assert out.demo_mode is False and out.model_used == "svm"
    assert out.results[0].category == "Non-Compete"
