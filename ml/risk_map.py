"""Transparent mapping from CUAD clause categories to risk levels.

The mapping is an *educational heuristic*, not legal advice.  Every entry
carries a plain-English explanation of why it has its risk level so that the
reasoning can be inspected, challenged and edited by the reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

#: Sort weight for risk levels (bigger = riskier).
RISK_ORDER: dict[str, int] = {HIGH: 3, MEDIUM: 2, LOW: 1, NONE: 0}

#: Short text tags used so risk is never conveyed by colour alone.
SHORT_TAG: dict[str, str] = {HIGH: "[HIGH]", MEDIUM: "[MED]", LOW: "[LOW]", NONE: "[---]"}

#: Points added to the overall score per clause of each level (score is capped at 100).
SCORE_WEIGHTS: dict[str, int] = {HIGH: 15, MEDIUM: 6, LOW: 1, NONE: 0}


@dataclass(frozen=True)
class RiskInfo:
    """Risk information for one clause category."""

    level: str
    explanation: str
    known: bool  # False when the category is not in RISK_MAP (default applied)


# ---------------------------------------------------------------------------
# The risk dictionary.  Keys are CUAD category names.
# ---------------------------------------------------------------------------
RISK_MAP: dict[str, tuple[str, str]] = {
    # ----- HIGH ------------------------------------------------------------
    "Uncapped Liability": (HIGH, "No ceiling on damages: a single claim could exceed the value of the whole contract and threaten the business."),
    "Non-Compete": (HIGH, "Restricts the party from working in its own market or with competitors, which can limit revenue and growth."),
    "Exclusivity": (HIGH, "Locks a party into one counterparty and removes the freedom to use alternative suppliers, customers or partners."),
    "Termination For Convenience": (HIGH, "Allows one side to exit without cause, so planned revenue or supply can disappear with little warning."),
    "Ip Ownership Assignment": (HIGH, "Transfers ownership of intellectual property; the assigning party may permanently lose rights to its own work."),
    "Liquidated Damages": (HIGH, "Pre-agreed penalty amounts are payable on breach whether or not actual loss occurred, and may be hard to dispute."),
    # ----- MEDIUM ----------------------------------------------------------
    "Anti-Assignment": (MEDIUM, "Contract rights cannot be transferred without consent, which complicates sales, mergers and restructuring."),
    "Change Of Control": (MEDIUM, "A merger or ownership change can trigger termination, consent or renegotiation rights for the other side."),
    "Audit Rights": (MEDIUM, "Gives the counterparty access to books, records or systems, creating cost, disruption and confidentiality exposure."),
    "Cap On Liability": (MEDIUM, "Limits how much can be recovered; protective for the payer but may under-compensate the injured party."),
    "Renewal Term": (MEDIUM, "Automatic or successive renewals can extend obligations unless notice is given on time."),
    "Notice Period To Terminate Renewal": (MEDIUM, "Missing a notice deadline may lock the party into another full term."),
    "Insurance": (MEDIUM, "Requires specific insurance cover that adds cost and must be maintained throughout the term."),
    "Minimum Commitment": (MEDIUM, "Fixed purchase or payment minimums apply even when demand falls."),
    "Revenue/Profit Sharing": (MEDIUM, "Obliges a share of revenue or profit; the calculation basis can be disputed."),
    "Price Restrictions": (MEDIUM, "Limits price changes or sets price ceilings, reducing commercial flexibility."),
    "Volume Restriction": (MEDIUM, "Caps or floors on quantities restrict how the relationship can scale."),
    "Most Favored Nation": (MEDIUM, "Requires matching the best terms given to anyone else, creating pricing and monitoring burdens."),
    "No-Solicit Of Customers": (MEDIUM, "Restricts approaching the other side's customers, narrowing future business development."),
    "No-Solicit Of Employees": (MEDIUM, "Restricts hiring or approaching the other side's staff."),
    "Non-Disparagement": (MEDIUM, "Limits what a party may say publicly about the other side, which may restrict legitimate speech."),
    "Joint Ip Ownership": (MEDIUM, "Shared IP ownership can require the other owner's cooperation to use or license the work."),
    "Irrevocable Or Perpetual License": (MEDIUM, "A licence that cannot be withdrawn permanently limits the licensor's control of its asset."),
    "License Grant": (MEDIUM, "Defines what may be used and how; scope errors can cause infringement or lost revenue."),
    "Competitive Restriction Exception": (MEDIUM, "Exceptions to a competition restriction change how far that restriction reaches."),
    "Rofr/Rofo/Rofn": (MEDIUM, "First-refusal or first-offer rights give the counterparty priority in future deals."),
    "Covenant Not To Sue": (MEDIUM, "A promise not to bring claims removes a legal remedy that may be needed later."),
    # ----- LOW -------------------------------------------------------------
    "Governing Law": (LOW, "Identifies which law applies; important to know but normally a neutral, standard term."),
    "Document Name": (LOW, "Identifies the document; informational only."),
    "Parties": (LOW, "Identifies who is bound by the agreement; informational but must be accurate."),
    "Effective Date": (LOW, "States when obligations start; informational but drives all deadlines."),
    "Agreement Date": (LOW, "Records the signing date; informational."),
    "Expiration Date": (LOW, "States when the term ends; informational but worth diarising."),
    "Post-Termination Services": (LOW, "Describes support after termination; usually routine."),
    "Warranty Duration": (LOW, "States how long a warranty lasts; routine but affects remedies."),
    "Third Party Beneficiary": (LOW, "Names outsiders who may enforce the contract; usually narrow in scope."),
    "Source Code Escrow": (LOW, "Protects access to source code; typically a safeguard rather than a burden."),
    "Non-Transferable License": (LOW, "Standard licence restriction preventing transfer to others."),
    "Affiliate License-Licensor": (LOW, "Extends licensed rights to the licensor's affiliates; routine."),
    "Affiliate License-Licensee": (LOW, "Extends licensed rights to the licensee's affiliates; routine."),
    "Unlimited/All-You-Can-Eat-License": (LOW, "Licence without usage limits; generally favourable to the licensee."),
}

#: Level and explanation used for categories that are not in RISK_MAP.
DEFAULT_RISK: tuple[str, str] = (
    MEDIUM,
    "This clause type has no entry in the risk map, so it is rated MEDIUM by default. Review it manually.",
)

#: Explanation attached to sentences with no review-relevant clause.
NO_CLAUSE_EXPLANATION = "No review-relevant clause category was detected."


def normalize_category(name: str) -> str:
    """Normalise a category name so ``"Non Compete"`` and ``"non-compete"`` match."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_INDEX: dict[str, str] = {normalize_category(k): k for k in RISK_MAP}


def canonical_name(category: str) -> str:
    """Return the RISK_MAP spelling of ``category`` (or the stripped input if unknown)."""
    return _INDEX.get(normalize_category(category), str(category).strip())


def get_risk(category: str | None) -> RiskInfo:
    """Return risk level and explanation for ``category``.

    * ``None`` / ``"None"`` / empty  -> level ``NONE``
    * a known category              -> its mapped level
    * anything else                 -> :data:`DEFAULT_RISK` with ``known=False``
    """
    if category is None or normalize_category(category) in ("", "none"):
        return RiskInfo(NONE, NO_CLAUSE_EXPLANATION, True)
    key = _INDEX.get(normalize_category(category))
    if key is None:
        return RiskInfo(DEFAULT_RISK[0], DEFAULT_RISK[1], False)
    level, why = RISK_MAP[key]
    return RiskInfo(level, why, True)


def overall_risk_score(levels: Iterable[str]) -> int:
    """Overall 0-100 score: 15 points per HIGH, 6 per MEDIUM, 1 per LOW, capped at 100."""
    return min(100, sum(SCORE_WEIGHTS.get(level, 0) for level in levels))


def risk_band(score: int) -> str:
    """Name the band for an overall score (0-29 low, 30-59 moderate, 60+ high)."""
    if score >= 60:
        return "HIGH RISK"
    if score >= 30:
        return "MODERATE RISK"
    return "LOW RISK"
