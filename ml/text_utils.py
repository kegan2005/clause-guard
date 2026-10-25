"""Sentence splitting with exact character offsets.

Two back-ends are available:

* ``"regex"`` (default) - a deterministic, legal-text-aware splitter that knows
  about numbering ("1.", "(a)"), abbreviations ("Inc.", "Ltd.") and headings.
* ``"spacy"`` - spaCy's rule-based ``sentencizer`` (no model download needed),
  applied paragraph by paragraph.  Used only if spaCy is installed.

Every returned :class:`Sentence` satisfies ``text[s.start:s.end] == s.text``.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    """A sentence and its half-open character span ``[start, end)`` in the source text."""

    text: str
    start: int
    end: int


# Words that end with a full stop but do not end a sentence (lower-case, no dot).
_ABBREVIATIONS = {
    "inc", "ltd", "co", "corp", "llc", "llp", "lp", "plc", "no", "nos", "sec", "secs", "art",
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "jr", "sr", "dept",
    "approx", "fig", "ex", "para", "cl", "sch", "pty", "gmbh", "ag", "s.a", "n.v", "b.v",
}

# Sentence terminator followed by whitespace and something that looks like a new sentence.
_TERMINATOR = re.compile(r"([.!?])([\"\u201d')\]]*)(\s+)(?=[\"\u201c(\[]?[A-Z0-9])")
# A token that is only a clause number, e.g. "1." "2.3." "(a)" "iv."
_NUMBERING = re.compile(r"^\(?(\d+(\.\d+)*|[A-Za-z]|[ivxlcdm]+)\)?$", re.IGNORECASE)
# An upper-case heading line such as "ARTICLE 3 - TERM" or "1. DEFINITIONS".
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Z0-9 ,&'\"/()\-]{2,}\.?$")
_LIST_MARKER = re.compile(r"^\s*(\(\w{1,4}\)|\d+(\.\d+)*[.)])\s")
_NEXT_CHAR = re.compile(r"\s*(\S)")


def _newline_cuts(text: str) -> set[int]:
    """Positions where a line break should also end a sentence."""
    cuts: set[int] = set()
    for m in re.finditer(r"\n", text):
        i = m.start()
        line_start = text.rfind("\n", 0, i) + 1
        prev = text[line_start:i].strip()
        if not prev:  # blank line = paragraph boundary
            cuts.add(i + 1)
            continue
        nxt = _NEXT_CHAR.match(text, i + 1)
        next_char = nxt.group(1) if nxt else ""
        is_heading = bool(_HEADING.match(prev)) and len(prev) <= 90
        if is_heading:
            cuts.add(i + 1)
        elif next_char.islower():  # a wrapped line continues the sentence
            continue
        elif prev[-1] in ".!?:;" or prev[-2:] in (".\"", ".\u201d"):
            cuts.add(i + 1)
        elif _LIST_MARKER.match(text[i + 1 : i + 12]):
            cuts.add(i + 1)
    return cuts


def _regex_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into raw (unstripped) spans using the regex rules."""
    bounds = [0] + sorted(_newline_cuts(text)) + [len(text)]
    cuts: list[int] = []
    for a, b in zip(bounds, bounds[1:]):
        seg_start = a
        for m in _TERMINATOR.finditer(text, a, b):
            if m.group(1) == ".":
                segment = text[seg_start : m.start()]
                words = segment.split()
                token = words[-1] if words else ""
                tok = token.lstrip("(\"'[\u201c").rstrip(",").lower().rstrip(".")
                if tok in _ABBREVIATIONS:
                    continue
                if segment.strip() == token and _NUMBERING.match(token):
                    continue  # lone clause number such as "1." - keep with its text
                if len(tok) == 1 and tok.isalpha() and token[:1].isupper():
                    continue  # an initial such as "J."
            seg_start = m.end()
            cuts.append(m.end())
    all_cuts = sorted(set(cuts) | set(bounds[1:-1]))
    edges = [0] + all_cuts + [len(text)]
    return list(zip(edges, edges[1:]))


@functools.lru_cache(maxsize=1)
def _spacy_nlp():
    """Build (once) a blank English pipeline with spaCy's rule-based sentencizer."""
    import spacy  # imported lazily: spaCy is optional

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def _spacy_spans(text: str) -> list[tuple[int, int]]:
    """Split with spaCy's sentencizer, paragraph by paragraph, merging bare numbering."""
    nlp = _spacy_nlp()
    nlp.max_length = max(nlp.max_length, len(text) + 1)
    spans: list[tuple[int, int]] = []
    pos = 0
    for para in re.split(r"(\n\s*\n)", text):
        if para and not para.isspace():
            doc = nlp(para)
            spans.extend((pos + s.start_char, pos + s.end_char) for s in doc.sents)
        pos += len(para)
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for a, b in spans:
        if carry is not None:
            a, carry = carry, None
        if re.fullmatch(r"\s*\(?\d+(\.\d+)*[.)]?\s*", text[a:b]):
            carry = a  # a bare "1." belongs with the sentence that follows
            continue
        merged.append((a, b))
    return merged


def available_backends() -> list[str]:
    """Return the sentence-splitting back-ends usable in this environment."""
    backends = ["regex"]
    try:
        import spacy  # noqa: F401

        backends.append("spacy")
    except Exception:  # pragma: no cover - depends on the environment
        pass
    return backends


def split_sentences(text: str, backend: str = "regex") -> list[Sentence]:
    """Split ``text`` into sentences with accurate character offsets.

    Unknown or unavailable back-ends fall back to ``"regex"``.
    """
    if not text or not text.strip():
        return []
    if backend == "spacy" and "spacy" in available_backends():
        spans = _spacy_spans(text)
    else:
        spans = _regex_spans(text)
    sentences: list[Sentence] = []
    for a, b in spans:
        chunk = text[a:b]
        stripped = chunk.strip()
        if not stripped or not re.search(r"[A-Za-z]", stripped):
            continue
        start = a + (len(chunk) - len(chunk.lstrip()))
        end = start + len(stripped)
        sentences.append(Sentence(stripped, start, end))
    return sentences
