"""Tests for the risk map and overall score."""
import pytest

from ml.risk_map import (
    DEFAULT_RISK, HIGH, LOW, MEDIUM, NONE, RISK_MAP, get_risk, normalize_category, overall_risk_score, risk_band,
)

HIGH_CATS = ["Uncapped Liability", "Non-Compete", "Exclusivity", "Termination For Convenience", "Ip Ownership Assignment", "Liquidated Damages"]
MEDIUM_CATS = ["Anti-Assignment", "Change Of Control", "Audit Rights"]
LOW_CATS = ["Governing Law", "Document Name", "Parties", "Effective Date"]


@pytest.mark.parametrize("cat", HIGH_CATS)
def test_high_risk_categories(cat):
    assert get_risk(cat).level == HIGH


@pytest.mark.parametrize("cat", MEDIUM_CATS)
def test_medium_risk_categories(cat):
    assert get_risk(cat).level == MEDIUM


@pytest.mark.parametrize("cat", LOW_CATS)
def test_low_risk_categories(cat):
    assert get_risk(cat).level == LOW


@pytest.mark.parametrize("variant", ["non-compete", "NON COMPETE", "Non_Compete", "  non-compete "])
def test_lookup_is_case_and_punctuation_insensitive(variant):
    assert get_risk(variant).level == HIGH
    assert normalize_category(variant) == "noncompete"


def test_unknown_category_gets_default():
    info = get_risk("Totally Made Up Clause")
    assert (info.level, info.explanation) == DEFAULT_RISK
    assert info.known is False


@pytest.mark.parametrize("value", [None, "", "None", "none"])
def test_no_clause_maps_to_none_level(value):
    assert get_risk(value).level == NONE


def test_every_entry_has_level_and_explanation():
    for cat, (level, why) in RISK_MAP.items():
        assert level in (HIGH, MEDIUM, LOW), cat
        assert len(why) > 20, cat
        assert get_risk(cat).known


def test_overall_score_weights_and_cap():
    assert overall_risk_score([]) == 0
    assert overall_risk_score([HIGH, MEDIUM, LOW]) == 15 + 6 + 1
    assert overall_risk_score([HIGH] * 20) == 100
    assert overall_risk_score([NONE]) == 0


def test_risk_band_thresholds():
    assert risk_band(0) == "LOW RISK"
    assert risk_band(30) == "MODERATE RISK"
    assert risk_band(60) == "HIGH RISK"
