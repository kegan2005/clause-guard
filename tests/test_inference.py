"""Tests for sentence splitting, offsets, inference structure and the demo fallback."""
import pytest

from ml import inference
from ml.inference import ClauseResult, RuleBasedDemoClassifier, analyze_text, model_status, results_to_records, summarize
from ml.paths import ROOT
from ml.text_utils import split_sentences

SAMPLE = (ROOT / "data" / "sample_contract.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------- splitting
def test_offsets_match_source_text():
    for s in split_sentences(SAMPLE):
        assert SAMPLE[s.start : s.end] == s.text
        assert s.text == s.text.strip()


def test_empty_and_blank_input():
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []


def test_abbreviations_do_not_split():
    text = "Acme Inc. shall pay Beta Ltd. the fee. The fee is due monthly."
    sents = [s.text for s in split_sentences(text)]
    assert sents == ["Acme Inc. shall pay Beta Ltd. the fee.", "The fee is due monthly."]


def test_clause_numbers_stay_with_their_sentence():
    sents = [s.text for s in split_sentences("1. Term. The agreement lasts one year.")]
    assert sents[0] == "1. Term."
    assert sents[1] == "The agreement lasts one year."


def test_wrapped_lines_are_joined_and_paragraphs_split():
    text = "The Supplier shall deliver\nthe goods on time.\n\nThe Customer shall pay."
    sents = split_sentences(text)
    assert [s.text for s in sents] == ["The Supplier shall deliver\nthe goods on time.", "The Customer shall pay."]
    assert all(text[s.start : s.end] == s.text for s in sents)


def test_leading_whitespace_offsets():
    text = "   First sentence.   Second one."
    sents = split_sentences(text)
    assert text[sents[0].start : sents[0].end] == "First sentence."
    assert sents[1].start == text.index("Second")


# ----------------------------------------------------------------- demo classifier
@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Customer shall not engage in any business that competes with Supplier.", "Non-Compete"),
        ("The Supplier's liability under this Agreement shall be unlimited.", "Uncapped Liability"),
        ("Supplier shall pay liquidated damages of 2% per day.", "Liquidated Damages"),
        ("This Agreement shall be governed by the laws of Delaware.", "Governing Law"),
        ("Either party may terminate this Agreement for convenience on 30 days notice.", "Termination For Convenience"),
        ("The weather was pleasant on the day of signing.", "None"),
    ],
)
def test_demo_classifier_rules(sentence, expected):
    assert RuleBasedDemoClassifier().classify(sentence)[0] == expected


# ----------------------------------------------------------------- inference
def test_no_models_means_demo_mode(empty_models_dir):
    assert model_status() == {"svm": False, "legalbert": False}
    out = analyze_text(SAMPLE, "svm", 0.5)
    assert out.demo_mode is True
    assert out.model_used == "demo"
    assert "demo" in out.notice.lower()


def test_result_structure(empty_models_dir):
    out = analyze_text(SAMPLE, "legalbert", 0.5)
    assert out.results and out.flagged
    for r in out.results:
        assert isinstance(r, ClauseResult)
        assert SAMPLE[r.start : r.end] == r.text
        assert 0.0 <= r.confidence <= 1.0
        assert r.risk_level in {"HIGH", "MEDIUM", "LOW", "NONE"}
        assert isinstance(r.risk_explanation, str) and r.risk_explanation
    levels = {r.risk_level for r in out.flagged}
    assert {"HIGH", "MEDIUM", "LOW"} <= levels  # the sample contains all three


def test_threshold_suppresses_low_confidence_predictions(empty_models_dir):
    out = analyze_text(SAMPLE, "svm", 0.99)
    assert out.flagged == []
    assert any(r.raw_category != "None" for r in out.results)  # raw prediction is preserved


def test_allow_demo_false_raises(empty_models_dir):
    with pytest.raises(RuntimeError):
        analyze_text(SAMPLE, "svm", 0.5, allow_demo=False)


def test_summary_and_export_records(empty_models_dir):
    out = analyze_text(SAMPLE, "svm", 0.5)
    stats = summarize(out.results)
    assert stats["total"] == stats["high"] + stats["medium"] + stats["low"] == len(out.flagged)
    assert 0 <= stats["score"] <= 100
    records = results_to_records(out.results)
    assert len(records) == len(out.flagged)
    assert {"category", "risk_level", "confidence", "start", "end", "clause_text", "risk_explanation"} <= set(records[0])


def test_empty_contract_returns_no_results(empty_models_dir):
    out = analyze_text("", "svm", 0.5)
    assert out.results == [] and summarize(out.results)["score"] == 0
