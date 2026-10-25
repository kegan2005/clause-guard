"""Clause classification: trained models plus a rule-based DEMO fallback.

Public API
----------
* :func:`model_status`   - which trained models exist on disk
* :func:`analyze_text`   - split a contract and classify every sentence
* :func:`summarize`      - counts and overall risk score for the UI
* :func:`results_to_records` - plain dictionaries for CSV / JSON export

If the requested model has not been trained, :func:`analyze_text` uses
:class:`RuleBasedDemoClassifier` and sets ``AnalysisOutput.demo_mode = True``
so the UI can show a visible DEMO MODE banner.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Protocol

from ml import paths
from ml.paths import MODEL_KEYS, MODEL_LABELS, NONE_LABEL
from ml.risk_map import HIGH, LOW, MEDIUM, NONE, canonical_name, get_risk, overall_risk_score, risk_band
from ml.text_utils import Sentence, split_sentences  # noqa: F401  (re-exported)

#: Temperature applied to LinearSVC margins before softmax (scores are NOT calibrated probabilities).
SVM_SCORE_SCALE = 3.0
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ClauseResult:
    """Classification of one sentence of the contract."""

    index: int
    text: str
    start: int
    end: int
    category: str  # after thresholding ("None" if below threshold)
    confidence: float
    risk_level: str
    risk_explanation: str
    raw_category: str  # model output before thresholding
    model: str  # "svm", "legalbert" or "demo"

    @property
    def is_flagged(self) -> bool:
        """True when a review-relevant category was assigned."""
        return self.category != NONE_LABEL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisOutput:
    """Everything the UI needs after an analysis run."""

    results: list[ClauseResult]
    model_requested: str
    model_used: str  # "svm" | "legalbert" | "demo"
    demo_mode: bool
    notice: str
    threshold: float
    splitter: str = "regex"
    flagged: list[ClauseResult] = field(init=False)

    def __post_init__(self) -> None:
        self.flagged = [r for r in self.results if r.is_flagged]


# ---------------------------------------------------------------------------
# Classifier interface and implementations
# ---------------------------------------------------------------------------
class Classifier(Protocol):
    """Anything that maps sentences to ``(label, confidence)`` pairs."""

    key: str

    def predict(self, texts: list[str]) -> list[tuple[str, float]]: ...


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


class SVMClassifier:
    """TF-IDF + LinearSVC pipeline saved by ``ml/train_baseline.py``."""

    key = "svm"

    def __init__(self) -> None:
        import joblib

        self.pipeline = joblib.load(paths.model_dir("svm") / "model.joblib")
        self.classes: list[str] = [str(c) for c in self.pipeline.classes_]

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        if not texts:
            return []
        scores = self.pipeline.decision_function(texts)
        if getattr(scores, "ndim", 2) == 1:  # binary problem -> two columns
            scores = [[-s, s] for s in scores]
        out: list[tuple[str, float]] = []
        for row in scores:
            row = [float(v) * SVM_SCORE_SCALE for v in row]
            probs = _softmax(row)
            best = max(range(len(probs)), key=probs.__getitem__)
            out.append((self.classes[best], probs[best]))
        return out


class LegalBertClassifier:
    """Fine-tuned Legal-BERT saved by ``ml/train_legalbert.py``."""

    key = "legalbert"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        folder = str(paths.model_dir("legalbert"))
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model = AutoModelForSequenceClassification.from_pretrained(folder).to(self.device).eval()
        id2label = self.model.config.id2label
        self.classes = [str(id2label[i]) for i in range(len(id2label))]

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        torch = self.torch
        for i in range(0, len(texts), BERT_BATCH_SIZE):
            batch = texts[i : i + BERT_BATCH_SIZE]
            enc = self.tokenizer(batch, truncation=True, max_length=BERT_MAX_LENGTH, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            conf, idx = probs.max(dim=-1)
            out.extend((self.classes[int(j)], float(c)) for c, j in zip(conf, idx))
        return out


def _c(*patterns: str, flags: int = re.IGNORECASE) -> list[re.Pattern[str]]:
    return [re.compile(p, flags) for p in patterns]


#: Keyword rules for demo mode. Order = tie-break priority (riskier categories first).
_DEMO_RULES: dict[str, list[re.Pattern[str]]] = {
    "Uncapped Liability": _c(
        r"\bunlimited\s+liabilit",
        r"liabilit\w*[^.;]{0,120}\b(unlimited|shall not be (limited|capped))",
        r"\bno\s+(cap|limit(ation)?)\s+(on|to|of)\b[^.;]{0,40}liabilit",
        r"shall not be (subject to|limited by)[^.;]{0,40}(cap|limitation of liability|limit)",
    ),
    "Non-Compete": _c(
        r"\bnon-?\s?compet",
        r"shall not[^.;]{0,80}\bcompete\b",
        r"engage in[^.;]{0,80}\bcompet(e[sd]?|ing|itive)\b",
        r"agrees? not to[^.;]{0,60}compet",
    ),
    "Exclusivity": _c(
        r"\bsole and exclusive (supplier|provider|distributor|source|vendor|manufacturer|basis)",
        r"\bexclusive (supplier|provider|distributor|dealer|licensee|right to (sell|distribute|supply|purchase))",
        r"\bon an exclusive basis",
        r"\bexclusively\b[^.;]{0,60}(from|purchase|supply|through)",
        r"shall not[^.;]{0,60}(purchase|procure|obtain|source)[^.;]{0,80}(from any|other than)",
    ),
    "Termination For Convenience": _c(
        r"terminat\w+[^.;]{0,80}\bfor convenience",
        r"terminat\w+[^.;]{0,100}(for any reason|without cause|for no reason)",
        r"\bfor convenience\b",
    ),
    "Ip Ownership Assignment": _c(
        r"\bhereby\s+(irrevocably\s+)?(assigns?|transfers?|conveys?)",
        r"assigns?[^.;]{0,60}all (of )?(its |their )?right,? title,? and interest",
        r"works? made for hire",
        r"(shall|will) (be|remain|become)[^.;]{0,40}(sole and )?exclusive property of",
        r"\bshall (vest|be vested) in",
    ),
    "Liquidated Damages": _c(
        r"liquidated damages",
        r"penalt(y|ies)[^.;]{0,60}(per day|per week|late|delay)",
        r"shall pay[^.;]{0,60}\bper day of delay",
    ),
    "Anti-Assignment": _c(
        r"(may|shall|will|can) not[^.;]{0,40}\b(assign|transfer|delegate|sublicense)\b",
        r"neither party (may|shall|will) (assign|transfer)",
        r"\bno assignment\b",
        r"assign(ment)?[^.;]{0,80}without (the )?(prior )?(express )?written consent",
    ),
    "Change Of Control": _c(
        r"change (of|in) control",
        r"(merger|consolidation|acquisition)[^.;]{0,80}(all or substantially all|controlling interest|voting (securities|power|stock))",
        r"(acquires?|acquisition of)[^.;]{0,40}(more than|in excess of|at least)\s+\d+\s?%[^.;]{0,40}(voting|equity|shares)",
    ),
    "Audit Rights": _c(
        r"\baudit(s|ed|ing)?\b",
        r"(inspect|examine)[^.;]{0,60}(books|records|accounts)",
        r"books and records",
    ),
    "Cap On Liability": _c(
        r"(aggregate|total|maximum)\s+liabilit[^.;]{0,100}(shall not|will not|not)\s+exceed",
        r"in no event[^.;]{0,120}(exceed|in excess of)",
        r"liabilit[^.;]{0,80}limited to (the )?(amounts?|fees)",
    ),
    "Renewal Term": _c(r"(automatically|auto)[- ]?renew", r"successive (renewal )?(terms|periods)", r"renewal term"),
    "Insurance": _c(r"(maintain|carry)[^.;]{0,60}insurance"),
    "Governing Law": _c(
        r"governed by[^.;]{0,60}\blaws?\b",
        r"\bgoverning law\b",
        r"construed in accordance with[^.;]{0,40}laws",
    ),
    "Parties": _c(
        r"\bby and between\b",
        r"\bis (made|entered into) (and entered into )?(by|between)",
        r"\(\s*(the\s+)?[\"\u201c']?(Supplier|Customer|Company|Client|Vendor|Licensor|Licensee|Buyer|Seller|Provider|Distributor)[\"\u201d']?\s*\)",
    ),
    "Effective Date": _c(
        r"\beffective (as of|date)\b[^.;]{0,30}(means|is|shall be|:)",
        r"(entered into|made) (as of|on|effective)",
        r"\beffective as of\b",
    ),
}
_DOCUMENT_NAME = re.compile(r"^[A-Z][A-Z0-9 &,.'\-]{4,100}\b(AGREEMENT|CONTRACT)\s*$")


class RuleBasedDemoClassifier:
    """Keyword/regex classifier used when no trained model is available.

    Its scores are *heuristic rule strengths*, not probabilities.  It exists so
    the application remains demonstrable before training finishes.
    """

    key = "demo"

    def classify(self, text: str) -> tuple[str, float]:
        """Return ``(category, heuristic_score)`` for one sentence."""
        stripped = text.strip()
        if _DOCUMENT_NAME.match(stripped):
            return "Document Name", 0.85
        if len(stripped) < 90 and stripped.upper() == stripped:  # all-caps section heading
            return NONE_LABEL, 0.90
        best_cat, best_hits = NONE_LABEL, 0
        for category, patterns in _DEMO_RULES.items():
            hits = sum(1 for p in patterns if p.search(text))
            if hits > best_hits:  # strict '>' keeps earlier (riskier) category on ties
                best_cat, best_hits = category, hits
        if best_hits == 0:
            return NONE_LABEL, 0.90
        return best_cat, min(0.92, 0.62 + 0.12 * (best_hits - 1))

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Model discovery and cached loading
# ---------------------------------------------------------------------------
def _cache_resource(func: Callable) -> Callable:
    """Use ``st.cache_resource`` inside a Streamlit runtime, ``lru_cache`` elsewhere (e.g. pytest)."""
    try:
        from streamlit.runtime import exists

        if exists():
            import streamlit as st

            return st.cache_resource(show_spinner=False)(func)
    except Exception:  # Streamlit missing or no runtime: fall through
        pass
    return functools.lru_cache(maxsize=4)(func)


def is_trained(model_key: str) -> bool:
    """True if the artefacts for ``model_key`` exist on disk."""
    folder = paths.model_dir(model_key)
    if model_key == "svm":
        return (folder / "model.joblib").is_file()
    return (folder / "config.json").is_file() and any(
        (folder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def model_status() -> dict[str, bool]:
    """Map each model key to ``True`` when a trained model is present."""
    return {key: is_trained(key) for key in MODEL_KEYS}


def _signature(model_key: str) -> str:
    """Cache key that changes when the model files are re-trained."""
    folder = paths.model_dir(model_key)
    marker = folder / ("model.joblib" if model_key == "svm" else "config.json")
    try:
        return f"{folder}:{marker.stat().st_mtime_ns}"
    except OSError:
        return f"{folder}:missing"


@_cache_resource
def _load_classifier(model_key: str, signature: str) -> Classifier:  # noqa: ARG001 - signature busts the cache
    return SVMClassifier() if model_key == "svm" else LegalBertClassifier()


def get_classifier(model_key: str) -> tuple[Classifier, bool, str]:
    """Return ``(classifier, demo_mode, notice)`` for ``model_key``."""
    name = MODEL_LABELS.get(model_key, model_key)
    if model_key in MODEL_KEYS and is_trained(model_key):
        try:
            return _load_classifier(model_key, _signature(model_key)), False, f"Using trained model: {name}."
        except Exception as exc:  # e.g. torch not installed, corrupted file
            reason = f"{name} could not be loaded ({exc.__class__.__name__}: {exc})."
    else:
        reason = f"No trained {name} found in models/."
    return RuleBasedDemoClassifier(), True, f"{reason} Using the rule-based demo classifier instead."


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def classify_sentences(
    sentences: list[Sentence],
    classifier: Classifier,
    threshold: float,
) -> list[ClauseResult]:
    """Classify ``sentences`` and apply the confidence ``threshold``."""
    predictions = classifier.predict([s.text for s in sentences])
    results: list[ClauseResult] = []
    for i, (sent, (raw, conf)) in enumerate(zip(sentences, predictions)):
        raw = str(raw)
        is_none = raw.strip().lower() == NONE_LABEL.lower()
        category = NONE_LABEL if is_none or conf < threshold else canonical_name(raw)
        info = get_risk(category)
        results.append(
            ClauseResult(
                index=i,
                text=sent.text,
                start=sent.start,
                end=sent.end,
                category=category,
                confidence=round(float(conf), 4),
                risk_level=info.level,
                risk_explanation=info.explanation,
                raw_category=raw,
                model=classifier.key,
            )
        )
    return results


def analyze_text(
    text: str,
    model_key: str = "svm",
    threshold: float = 0.5,
    splitter: str = "regex",
    allow_demo: bool = True,
) -> AnalysisOutput:
    """Split ``text`` into sentences and classify each one.

    With ``allow_demo=False`` and no trained model, a :class:`RuntimeError` is
    raised instead of silently using the demo classifier (used by Compare Models).
    """
    classifier, demo, notice = get_classifier(model_key)
    if demo and not allow_demo:
        raise RuntimeError(notice)
    sentences = split_sentences(text, backend=splitter)
    results = classify_sentences(sentences, classifier, threshold)
    return AnalysisOutput(
        results=results,
        model_requested=model_key,
        model_used=classifier.key,
        demo_mode=demo,
        notice=notice,
        threshold=threshold,
        splitter=splitter,
    )


def summarize(results: Iterable[ClauseResult]) -> dict[str, Any]:
    """Counts per risk level and the overall 0-100 risk score."""
    flagged = [r for r in results if r.is_flagged]
    levels = [r.risk_level for r in flagged]
    score = overall_risk_score(levels)
    return {
        "total": len(flagged),
        "high": levels.count(HIGH),
        "medium": levels.count(MEDIUM),
        "low": levels.count(LOW),
        "score": score,
        "band": risk_band(score),
    }


def results_to_records(results: Iterable[ClauseResult], only_flagged: bool = True) -> list[dict[str, Any]]:
    """Flat dictionaries suitable for CSV / JSON export."""
    return [
        {
            "clause_no": r.index + 1,
            "category": r.category,
            "risk_level": r.risk_level,
            "confidence": r.confidence,
            "start": r.start,
            "end": r.end,
            "clause_text": r.text,
            "risk_explanation": r.risk_explanation,
            "model": r.model,
        }
        for r in results
        if r.is_flagged or not only_flagged
    ]
