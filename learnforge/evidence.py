"""Deterministic evidence assessment (Milestone M3).

Sits between M2 retrieval and M4 generation:

    query -> M2 retrieval -> **M3 assessment** -> M4 generation (later)

M3 decides — **deterministically, with no LLM** — whether the retrieved evidence is
safe to answer from, and exposes a structured, explainable assessment object. It
never generates a customer-facing answer.

Assessment states map onto M0 §8/§11/§12 exactly (see ``docs/evidence-assessment.md``):

======================  ==============================  ===========================
M3 state                M0 evidence decision (§8.4)     M0 routing (§8.5)
======================  ==============================  ===========================
``answerable``          ``supported``                   generate
``conflicting_evidence`` ``conflicting-stale``          generate-with-disclosure,
                                                        or escalate (M0 §11.3)
``insufficient_evidence`` ``low-evidence``/``no-evidence`` decline
``clarification_required`` (route CLARIFY)              clarify
``escalation_required``  (route ESCALATE)               escalate
``security_escalation``  (M0 §12.2 security, FR-17)     escalate (safe handling)
======================  ==============================  ===========================

Eight separate evidence dimensions are evaluated and reported individually
(relevance, authority, freshness, contradiction, ambiguity, coverage,
security/PII, escalation indicators). Nothing is collapsed into a single
unexplained number; the optional confidence value is a *categorical heuristic*
(high/medium/low) whose components are always exposed.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from learnforge.lexical import tokenize
from learnforge.retrieval import RetrievalResult

# --------------------------------------------------------------------------- #
# Assessment states (machine-readable) and M0 routing
# --------------------------------------------------------------------------- #

ANSWERABLE = "answerable"
CLARIFICATION_REQUIRED = "clarification_required"
CONFLICTING_EVIDENCE = "conflicting_evidence"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ESCALATION_REQUIRED = "escalation_required"
SECURITY_ESCALATION = "security_escalation"

ASSESSMENT_STATES = (
    ANSWERABLE,
    CLARIFICATION_REQUIRED,
    CONFLICTING_EVIDENCE,
    INSUFFICIENT_EVIDENCE,
    ESCALATION_REQUIRED,
    SECURITY_ESCALATION,
)

ROUTE_GENERATE = "generate"
ROUTE_CLARIFY = "clarify"
ROUTE_ESCALATE = "escalate"
ROUTE_DECLINE = "decline"

# --------------------------------------------------------------------------- #
# Confidence (categorical heuristic — NOT a calibrated probability)
# --------------------------------------------------------------------------- #

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

#: M0 §12.1 confidence levels, used verbatim for traceability.
M0_CONFIDENT_AND_SUPPORTED = "confident_and_supported"
M0_SUPPORTED_BUT_NUANCED = "supported_but_nuanced"
M0_CONFLICTING_OR_STALE = "conflicting_or_stale"
M0_LOW_EVIDENCE = "low_evidence"
M0_NO_EVIDENCE = "no_evidence"

# --------------------------------------------------------------------------- #
# Freshness / role / relevance vocabulary
# --------------------------------------------------------------------------- #

FRESHNESS_CURRENT = "current"      # explicitly dated and not flagged stale
FRESHNESS_STALE = "stale"          # explicitly flags an outdated prior version
FRESHNESS_UNDATED = "undated"      # no date line in the source (never invented)

ROLE_POLICY = "authoritative_policy"
ROLE_FAQ = "operational_guidance"
ROLE_TICKET = "historical_evidence"

ROLE_BY_TYPE = {"policy": ROLE_POLICY, "faq": ROLE_FAQ, "ticket": ROLE_TICKET}

RELEVANCE_HIGH = "high"
RELEVANCE_MEDIUM = "medium"
RELEVANCE_LOW = "low"

#: Relevance-band thresholds. Derived from the observed separation on this corpus:
#: on-topic queries score ~0.39-0.82 cosine while off-topic queries score ~0.13,
#: so 0.30 / 0.45 separate the bands without tuning to individual records.
RELEVANCE_HIGH_MIN = 0.45
RELEVANCE_MEDIUM_MIN = 0.30

# --------------------------------------------------------------------------- #
# Security / PII signals (M0 §12.4, FR-17; corpus POLICY-07/10, FAQ-15)
# --------------------------------------------------------------------------- #

#: Terms that unambiguously reference sensitive payment/identity data that must
#: never be requested or repeated.
SENSITIVE_DATA_RE = re.compile(
    r"\b(cvv|cvc|card\s+numbers?|credit\s+card\s+number|full\s+card|"
    r"banking\s+password|authentication\s+codes?|auth\s+codes?|pins?|"
    r"government\s+id|social\s+security|passport)\b",
    re.IGNORECASE,
)

#: A password/credential reference only becomes security-sensitive when the user
#: is offering to *share* it ("send my password"), not when resetting one.
SHARING_VERB_RE = re.compile(
    r"\b(send|sending|share|sharing|give|giving|provide|providing|email|tell|"
    r"include|submit|submitting)\b",
    re.IGNORECASE,
)
PASSWORD_RE = re.compile(r"\b(password|credentials)\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Query topic detection (M3-local; does not modify M1 metadata)
# --------------------------------------------------------------------------- #

#: Topics the *query* is about. These mirror the M1 contradiction topics (plus the
#: payment-information phrasing) so conflicts only fire for subjects the user
#: actually asked about.
QUERY_TOPIC_PATTERNS: dict[str, str] = {
    "refund_window": r"refund|money[\s-]*back",
    "annual_billing": r"\bannual\b|billed\s+monthly",
    "family_plan": r"family\s+plan|five[\s-]*user",
    "browser_support": r"browsers?\b|internet\s+explorer|\bie\b",
    "offline_downloads": r"offline|download",
    "progress_sync": r"\bprogress\b|synchron|saved\s+instantly",
    "captions_accessibility": r"caption|transcript|accessib",
    "payment_data_collection": (
        r"payment\s+information|payment\s+method|card\s+number|first\s+six|"
        r"last\s+four|\bcvv\b|\bcvc\b|\bpin\b|banking\s+password|"
        r"authentication\s+code"
    ),
}
_QUERY_TOPIC_RE = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in QUERY_TOPIC_PATTERNS.items()
}

# --------------------------------------------------------------------------- #
# Known contradiction families (M0 §6.4 / §11.1 — rule-based, corpus-specific)
# --------------------------------------------------------------------------- #

#: Each family names a stale/historical claim pattern and the current-guidance
#: pattern it conflicts with. All patterns are matched against the *verbatim*
#: record text (M1 stale notes are quoted inside ``chunk_text``, so one record can
#: legitimately supply both sides — that is exactly the contradiction to expose).
CONTRADICTION_FAMILIES: dict[str, dict[str, Any]] = {
    "refund_window": {
        "topic": "refund_window",
        "label": "refund window",
        "stale_claims": (r"\b7[\s-]+days?\b",),
        "current_claims": (r"\b14[\s-]+days?\b",),
        "exception_claims": (r"\b30[\s-]+days?\b", r"money[\s-]*back\s+guarantee"),
    },
    "annual_billing": {
        "topic": "annual_billing",
        "label": "annual subscription billing",
        "stale_claims": (r"billed\s+monthly",),
        "current_claims": (r"annual\s+price", r"shown\s+at\s+checkout"),
        "exception_claims": (),
    },
    "offline_downloads": {
        "topic": "offline_downloads",
        "label": "offline/desktop downloads",
        "stale_claims": (r"desktop\s+download", r"download[^.\n]{0,60}\blaptop\b"),
        "current_claims": (r"mobile\s+application", r"mobile\s+app\b", r"does\s+not\s+currently\s+provide"),
        "exception_claims": (),
    },
    "browser_support": {
        "topic": "browser_support",
        "label": "browser support",
        "stale_claims": (r"internet\s+explorer",),
        "current_claims": (r"\bchrome\b", r"\bedge\b", r"\bfirefox\b", r"\bsafari\b"),
        "exception_claims": (),
    },
    "captions_accessibility": {
        "topic": "captions_accessibility",
        "label": "caption requirement",
        "stale_claims": (r"captions\s+for\s+every\s+video", r"every\s+video\s+before\s+publication"),
        "current_claims": (r"whenever\s+practical",),
        "exception_claims": (),
    },
    "progress_sync": {
        "topic": "progress_sync",
        "label": "progress saving/synchronization",
        "stale_claims": (r"\binstantly\b",),
        "current_claims": (r"automatically\s+synchronized", r"synchron"),
        "exception_claims": (),
    },
    "payment_data_collection": {
        "topic": "payment_data_collection",
        "label": "payment-data collection by support",
        "stale_claims": (r"first\s+six\s+and\s+last\s+four",),
        "current_claims": (
            r"minimum\s+payment\s+information",
            r"must\s+not\s+request\s+complete\s+card",
            r"do\s+not\s+send\s+the\s+full\s+card",
            r"never\s+request",
        ),
        "exception_claims": (),
    },
}

#: M0 §11.2 conflict classes, derived per family from which sides were found.
CONFLICT_CLASS_CURRENT_VS_STALE = "current_vs_stale"
CONFLICT_CLASS_USER_CLAIM = "user_claim_vs_current_policy"

# --------------------------------------------------------------------------- #
# Ambiguity detection (M0 §8.2 "Cancel my LearnForge" -> clarify)
# --------------------------------------------------------------------------- #

#: Action verbs that indicate a *request for action* rather than a question.
AMBIGUOUS_ACTION_VERBS = frozenset(
    {"cancel", "close", "terminate", "remove", "delete", "transfer", "move", "switch"}
)
WH_WORDS = frozenset({"what", "when", "where", "which", "who", "whom", "why", "how"})
SPECIFIC_REFERENCE_RE = re.compile(
    r"\b(order|transaction|receipt|invoice|course|subscription|plan|account|"
    r"email|lesson|module)\b",
    re.IGNORECASE,
)
MAX_AMBIGUOUS_TOKENS = 6

# --------------------------------------------------------------------------- #
# Query-level deterministic signals
# --------------------------------------------------------------------------- #

_DAY_COUNT_RE = re.compile(r"\b(\d{1,3})\s*days?\b", re.IGNORECASE)

#: The current standard individual-course refund window (FAQ-02 / POLICY-02).
STANDARD_REFUND_WINDOW_DAYS = 14
#: Upper bound of the historical/promotional 30-day claim seen in the corpus
#: (TICKET-03, TICKET-08).
PROMOTIONAL_REFUND_WINDOW_DAYS = 30


def detect_query_topics(query: str) -> set[str]:
    """Return the contradiction-family topics the query is about."""
    return {
        name for name, regex in _QUERY_TOPIC_RE.items() if regex.search(query or "")
    }


def extract_day_counts(query: str) -> list[int]:
    """Return explicit day counts mentioned in the query (e.g. "20 days" -> [20])."""
    return [int(match.group(1)) for match in _DAY_COUNT_RE.finditer(query or "")]


def detect_security(query: str) -> tuple[bool, list[str], str]:
    """Detect queries that reference sensitive payment/identity data.

    Returns ``(triggered, matched_terms, rule)``. A query is security-sensitive
    when it names sensitive data outright, or offers to share a password.
    Asking *what* information support may request (without naming sensitive
    data) is **not** security-sensitive — that is answerable from policy.
    """
    text = query or ""
    matched = sorted({match.group(0).lower() for match in SENSITIVE_DATA_RE.finditer(text)})
    if matched:
        return True, matched, "query names sensitive payment/identity data"
    if PASSWORD_RE.search(text) and SHARING_VERB_RE.search(text):
        return True, ["password/credentials"], "query offers to share a password/credential"
    return False, [], ""


def detect_query_ambiguity(query: str, query_topics: set[str]) -> tuple[bool, str]:
    """Heuristically detect an underspecified action request.

    Mirrors the TICKET-07 pattern ("Cancel my LearnForge"): a short action
    request that names no specific object and no recognized topic. Informational
    questions (wh-word or "?") and queries that name a specific reference are not
    ambiguous. The clarification *question itself* is M4's job, not M3's.
    """
    text = (query or "").strip()
    tokens = tokenize(text)
    if not tokens:
        return False, "no usable tokens"
    if query_topics:
        return False, "query names a recognized topic"
    if text.endswith("?"):
        return False, "informational question"
    if tokens[0] in WH_WORDS:
        return False, "informational question"
    if not any(token in AMBIGUOUS_ACTION_VERBS for token in tokens):
        return False, "no action verb"
    if SPECIFIC_REFERENCE_RE.search(text):
        return False, "query names a specific object"
    if len(tokens) > MAX_AMBIGUOUS_TOKENS:
        return False, "query is too specific to be ambiguous"
    return True, "short action request without a specific object (TICKET-07 pattern)"


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


# --------------------------------------------------------------------------- #
# Assessment structures
# --------------------------------------------------------------------------- #

@dataclass
class EvidenceItem:
    """One retrieved record plus its assessed dimensions (evidence is not mutated)."""

    result: RetrievalResult
    role: str
    freshness_class: str
    authority: int
    relevant_topics: list[str]
    claims: dict[str, list[str]]
    is_stale: bool
    stale_reason: Optional[str]
    freshness: str
    ticket_status: Optional[str]
    escalated: bool
    unresolved: bool
    ambiguity_flags: list[str]
    contradiction_topics: list[str]
    semantic_score: float
    lexical_score: float
    fused_score: float

    @property
    def source_id(self) -> str:
        return self.result.source_id

    @property
    def citation_key(self) -> str:
        return self.result.record["citation_key"]

    @property
    def source_type(self) -> str:
        return self.result.source_type

    @property
    def chunk_text(self) -> str:
        return self.result.record["chunk_text"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "citation_key": self.citation_key,
            "source_type": self.source_type,
            "title": self.result.record["title"],
            "role": self.role,
            "authority": self.authority,
            "freshness_class": self.freshness_class,
            "freshness": self.freshness,
            "is_stale": self.is_stale,
            "stale_reason": self.stale_reason,
            "ticket_status": self.ticket_status,
            "escalated": self.escalated,
            "unresolved": self.unresolved,
            "ambiguity_flags": list(self.ambiguity_flags),
            "contradiction_topics": list(self.contradiction_topics),
            "relevant_topics": list(self.relevant_topics),
            "claims": {family: list(sides) for family, sides in self.claims.items()},
            "scores": {
                "semantic_score": round(self.semantic_score, 6),
                "lexical_score": round(self.lexical_score, 6),
                "fused_score": round(self.fused_score, 6),
            },
            "chunk_text": self.chunk_text,
        }


@dataclass
class ConflictIndicator:
    """One detected contradiction family within the relevant evidence."""

    family: str
    label: str
    topic: str
    classes: list[str]
    current_evidence_ids: list[str]
    stale_evidence_ids: list[str]
    exception_evidence_ids: list[str]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "topic": self.topic,
            "classes": list(self.classes),
            "current_evidence_ids": list(self.current_evidence_ids),
            "stale_evidence_ids": list(self.stale_evidence_ids),
            "exception_evidence_ids": list(self.exception_evidence_ids),
            "description": self.description,
        }


@dataclass
class AmbiguityAssessment:
    """Query/evidence ambiguity signals (M0 §8.2, §12.3)."""

    detected: bool
    reason: str
    flags: list[str]
    evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "reason": self.reason,
            "flags": list(self.flags),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class SecurityAssessment:
    """Security/PII signals (M0 §12.4, FR-17)."""

    triggered: bool
    matched_terms: list[str]
    rule: str
    evidence_ids_with_pii_rules: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "matched_terms": list(self.matched_terms),
            "rule": self.rule,
            "evidence_ids_with_pii_rules": list(self.evidence_ids_with_pii_rules),
        }


@dataclass
class EscalationAssessment:
    """Escalation indicators (M0 §12.2), lifted only when query-aligned."""

    required: bool
    reasons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"required": self.required, "reasons": [dict(r) for r in self.reasons]}


@dataclass
class CoverageAssessment:
    """Does the retrieved evidence actually cover what the query asks about?"""

    query_topics: list[str]
    covered_topics: list[str]
    uncovered_topics: list[str]
    relevant_record_count: int
    has_current_authoritative_evidence: bool
    stale_only_for_queried_topics: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_topics": list(self.query_topics),
            "covered_topics": list(self.covered_topics),
            "uncovered_topics": list(self.uncovered_topics),
            "relevant_record_count": self.relevant_record_count,
            "has_current_authoritative_evidence": self.has_current_authoritative_evidence,
            "stale_only_for_queried_topics": self.stale_only_for_queried_topics,
        }


@dataclass
class ConfidenceAssessment:
    """Categorical confidence plus every component that produced it."""

    level: str
    relevance_band: str
    components: dict[str, Any]
    m0_level: Optional[str]
    note: str = (
        "Internal heuristic for routing/debugging only; NOT a calibrated "
        "probability of correctness."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "relevance_band": self.relevance_band,
            "components": dict(self.components),
            "m0_level": self.m0_level,
            "note": self.note,
        }


@dataclass
class EvidenceAssessment:
    """Structured, explainable assessment of retrieved evidence (M3 output)."""

    query: str
    state: str
    routing: str
    confidence: ConfidenceAssessment
    evidence: list[EvidenceItem]
    conflicts: list[ConflictIndicator]
    ambiguity: AmbiguityAssessment
    security: SecurityAssessment
    escalation: EscalationAssessment
    coverage: CoverageAssessment
    explanation: str
    results: list[RetrievalResult] = field(default_factory=list, repr=False)

    @property
    def evidence_ids(self) -> list[str]:
        return [item.source_id for item in self.evidence]

    @property
    def citation_keys(self) -> list[str]:
        return [item.citation_key for item in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "state": self.state,
            "routing": self.routing,
            "confidence": self.confidence.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_ids": self.evidence_ids,
            "citation_keys": self.citation_keys,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "ambiguity": self.ambiguity.to_dict(),
            "security": self.security.to_dict(),
            "escalation": self.escalation.to_dict(),
            "coverage": self.coverage.to_dict(),
            "explanation": self.explanation,
        }


def _relevance_band(best_semantic: float) -> str:
    if best_semantic >= RELEVANCE_HIGH_MIN:
        return RELEVANCE_HIGH
    if best_semantic >= RELEVANCE_MEDIUM_MIN:
        return RELEVANCE_MEDIUM
    return RELEVANCE_LOW


def _freshness_class(is_stale: bool, freshness: str) -> str:
    """Buckets: explicitly dated+not stale / explicitly stale / undated."""
    if is_stale:
        return FRESHNESS_STALE
    if freshness and freshness != "undated":
        return FRESHNESS_CURRENT
    return FRESHNESS_UNDATED


def _item_claims(text: str) -> dict[str, list[str]]:
    """Map each contradiction family to the claim sides present in ``text``."""
    claims: dict[str, list[str]] = {}
    for family, spec in CONTRADICTION_FAMILIES.items():
        sides: list[str] = []
        if _matches_any(text, spec["stale_claims"]):
            sides.append("stale")
        if _matches_any(text, spec["current_claims"]):
            sides.append("current")
        if _matches_any(text, spec["exception_claims"]):
            sides.append("exception")
        if sides:
            claims[family] = sides
    return claims


def _build_evidence_item(result: RetrievalResult, query_topics: set[str]) -> EvidenceItem:
    meta = result.metadata
    is_stale = bool(meta.get("is_stale"))
    freshness = meta.get("freshness") or "undated"
    return EvidenceItem(
        result=result,
        role=ROLE_BY_TYPE[result.source_type],
        freshness_class=_freshness_class(is_stale, freshness),
        authority=int(meta.get("authority", 0)),
        relevant_topics=sorted(set(meta.get("contradiction_topics") or []) & query_topics),
        claims=_item_claims(result.record["chunk_text"]),
        is_stale=is_stale,
        stale_reason=meta.get("stale_reason"),
        freshness=freshness,
        ticket_status=meta.get("ticket_status"),
        escalated=bool(meta.get("escalated")),
        unresolved=bool(meta.get("unresolved")),
        ambiguity_flags=list(meta.get("ambiguity_flags") or []),
        contradiction_topics=list(meta.get("contradiction_topics") or []),
        semantic_score=result.semantic_score,
        lexical_score=result.lexical_score,
        fused_score=result.fused_score,
    )


def _detect_conflicts(
    items: Sequence[EvidenceItem], query_topics: set[str]
) -> list[ConflictIndicator]:
    """Detect known contradiction families among query-relevant evidence.

    A family only fires when (a) the query is about that family's topic and
    (b) query-relevant evidence contains both a current-guidance claim and a
    stale/historical/exception claim. Detecting the conflict does NOT decide
    which claim is true (M0 §11.4).
    """
    conflicts: list[ConflictIndicator] = []
    for family, spec in CONTRADICTION_FAMILIES.items():
        topic = spec["topic"]
        if topic not in query_topics:
            continue
        scoped = [item for item in items if topic in item.relevant_topics]
        if not scoped:
            continue

        current_ids: set[str] = set()
        stale_ids: set[str] = set()
        exception_ids: set[str] = set()
        for item in scoped:
            sides = item.claims.get(family, [])
            if "current" in sides:
                current_ids.add(item.source_id)
            if "stale" in sides:
                stale_ids.add(item.source_id)
            if "exception" in sides:
                exception_ids.add(item.source_id)

        if not current_ids or not (stale_ids or exception_ids):
            continue

        classes: list[str] = []
        if stale_ids:
            classes.append(CONFLICT_CLASS_CURRENT_VS_STALE)
        if exception_ids:
            classes.append(CONFLICT_CLASS_USER_CLAIM)

        parts = []
        if current_ids:
            parts.append("current guidance: " + ", ".join(sorted(current_ids)))
        if stale_ids:
            parts.append("explicitly outdated claim: " + ", ".join(sorted(stale_ids)))
        if exception_ids:
            parts.append(
                "historical/promotional or user claim: " + ", ".join(sorted(exception_ids))
            )

        conflicts.append(
            ConflictIndicator(
                family=family,
                label=spec["label"],
                topic=topic,
                classes=classes,
                current_evidence_ids=sorted(current_ids),
                stale_evidence_ids=sorted(stale_ids),
                exception_evidence_ids=sorted(exception_ids),
                description=f"{spec['label']}: " + "; ".join(parts),
            )
        )
    return conflicts


# --------------------------------------------------------------------------- #
# Escalation indicators (M0 §12.2) — query-aligned, evidence-supported
# --------------------------------------------------------------------------- #

_ACCOUNT_RE = re.compile(
    r"\b(account|verify|verification|login|log[\s-]*in|sign[\s-]*in|email)\b", re.IGNORECASE
)
_TRANSFER_RE = re.compile(r"\b(transfer|certificate|name)\b", re.IGNORECASE)
_MISSING_COURSE_RE = re.compile(
    r"\b(disappear|disappeared|missing|vanished|not\s+showing|"
    r"can(?:'|no)?t\s+(?:see|find|access))\b",
    re.IGNORECASE,
)


def _escalation_reasons(
    query: str,
    query_topics: set[str],
    items: Sequence[EvidenceItem],
    day_counts: Sequence[int],
) -> list[dict[str, Any]]:
    """Lift M1 escalation flags into the assessment only when query-aligned.

    A flag on a retrieved ticket describes *that ticket's* handling; it must not
    automatically escalate an unrelated query. Each rule therefore requires both
    (a) the flag to be present in the retrieved evidence and (b) the query to be
    about the same subject.
    """
    evidence_flags: dict[str, set[str]] = {
        item.source_id: set(item.ambiguity_flags) for item in items
    }
    all_flags: set[str] = set()
    for flags in evidence_flags.values():
        all_flags.update(flags)

    def evidence_with(flag: str) -> list[str]:
        return sorted(
            source_id for source_id, flags in evidence_flags.items() if flag in flags
        )

    reasons: list[dict[str, Any]] = []

    def add(flag: str, reason: str) -> None:
        reasons.append(
            {"flag": flag, "reason": reason, "evidence_ids": evidence_with(flag)}
        )

    if "promotional_terms_may_differ" in all_flags and "refund_window" in query_topics:
        outside = [
            days
            for days in day_counts
            if STANDARD_REFUND_WINDOW_DAYS < days <= PROMOTIONAL_REFUND_WINDOW_DAYS
        ]
        if outside:
            add(
                "promotional_terms_may_differ",
                f"query cites a {outside[0]}-day refund window, outside the current "
                f"{STANDARD_REFUND_WINDOW_DAYS}-day standard; promotional/contractual "
                "terms may differ and need verification (M0 §12.2, TICKET-03)",
            )

    if (
        "policy_wording_ambiguity" in all_flags
        and "refund_window" in query_topics
        and re.search(r"subscription", query, re.IGNORECASE)
    ):
        add(
            "policy_wording_ambiguity",
            "query concerns subscription refund wording; the corpus records "
            "contradictory cancellation-page wording (M0 §12.2, TICKET-08)",
        )

    if "outdated_documentation_reliance" in all_flags and "offline_downloads" in query_topics:
        add(
            "outdated_documentation_reliance",
            "query concerns downloads where official documentation the user relied "
            "on was outdated and material to a purchase (M0 §12.2, TICKET-15)",
        )

    if "identity_verification_required" in all_flags and _ACCOUNT_RE.search(query):
        add(
            "identity_verification_required",
            "query concerns account/identity details the corpus shows require "
            "verification before action (M0 §12.2, TICKET-10)",
        )

    if "enrollment_transfer_requires_review" in all_flags and _TRANSFER_RE.search(query):
        add(
            "enrollment_transfer_requires_review",
            "query concerns a learner/certificate transfer the corpus shows is "
            "pending review (M0 §12.2, TICKET-13)",
        )

    if "purchase_type_conflict" in all_flags and _MISSING_COURSE_RE.search(query):
        add(
            "purchase_type_conflict",
            "query reports a course that disappeared; the corpus shows an ambiguous "
            "individual-purchase vs subscription conflict (M0 §12.2, TICKET-11)",
        )

    return reasons


def _build_ambiguity(
    query: str, query_topics: set[str], items: Sequence[EvidenceItem]
) -> AmbiguityAssessment:
    heuristic_detected, reason = detect_query_ambiguity(query, query_topics)
    flagged_ids = [
        item.source_id
        for item in items
        if "ambiguous_intent_requires_clarification" in item.ambiguity_flags
    ]
    detected = heuristic_detected and bool(flagged_ids)
    if detected:
        reason = (
            f"{reason}; retrieved evidence supports the TICKET-07 ambiguity "
            "classification"
        )
    flags = ["ambiguous_intent_requires_clarification"] if detected else []
    return AmbiguityAssessment(
        detected=detected, reason=reason, flags=flags, evidence_ids=flagged_ids
    )


def _build_security(query: str, items: Sequence[EvidenceItem]) -> SecurityAssessment:
    triggered, matched, rule = detect_security(query)
    pii_ids = sorted(
        item.source_id
        for item in items
        if "payment_data_collection" in item.contradiction_topics
    )
    return SecurityAssessment(
        triggered=triggered,
        matched_terms=matched,
        rule=rule,
        evidence_ids_with_pii_rules=pii_ids,
    )


def _build_coverage(query_topics: set[str], items: Sequence[EvidenceItem]) -> CoverageAssessment:
    relevant = [item for item in items if item.relevant_topics]
    covered = sorted(
        {topic for item in relevant for topic in item.relevant_topics} & query_topics
    )
    uncovered = sorted(set(query_topics) - set(covered))
    current_authoritative = [
        item
        for item in relevant
        if item.source_type in ("policy", "faq")
        and item.freshness_class != FRESHNESS_STALE
    ]
    has_current_side = any(
        "current" in item.claims.get(family, [])
        for item in relevant
        for family in item.relevant_topics
    )
    return CoverageAssessment(
        query_topics=sorted(query_topics),
        covered_topics=covered,
        uncovered_topics=uncovered,
        relevant_record_count=len(relevant),
        has_current_authoritative_evidence=bool(current_authoritative),
        stale_only_for_queried_topics=(
            bool(query_topics) and bool(relevant) and not has_current_side
        ),
    )


# --------------------------------------------------------------------------- #
# State decision, routing, confidence, explanation
# --------------------------------------------------------------------------- #

_ROUTING_BY_STATE = {
    ANSWERABLE: ROUTE_GENERATE,
    CONFLICTING_EVIDENCE: ROUTE_GENERATE,      # disclosure required (M0 §11.3)
    ESCALATION_REQUIRED: ROUTE_ESCALATE,
    SECURITY_ESCALATION: ROUTE_ESCALATE,
    CLARIFICATION_REQUIRED: ROUTE_CLARIFY,
    INSUFFICIENT_EVIDENCE: ROUTE_DECLINE,
}


def _decide_state(
    *,
    security_triggered: bool,
    ambiguity_detected: bool,
    band: str,
    has_items: bool,
    coverage: CoverageAssessment,
    has_conflicts: bool,
    escalation_reasons: Sequence[Any],
) -> str:
    if security_triggered:
        return SECURITY_ESCALATION
    if ambiguity_detected:
        return CLARIFICATION_REQUIRED
    if not has_items or band == RELEVANCE_LOW:
        return INSUFFICIENT_EVIDENCE
    if coverage.uncovered_topics or coverage.stale_only_for_queried_topics:
        return INSUFFICIENT_EVIDENCE
    if has_conflicts:
        return CONFLICTING_EVIDENCE
    if escalation_reasons:
        return ESCALATION_REQUIRED
    return ANSWERABLE


def _routing_for(state: str, escalation_required: bool) -> str:
    """M0 §8.5 routing; a conflict escalates when its escalation signal fired."""
    if state == CONFLICTING_EVIDENCE and escalation_required:
        return ROUTE_ESCALATE
    return _ROUTING_BY_STATE[state]


def _m0_confidence_level(state: str, level: str, has_evidence: bool) -> Optional[str]:
    if state == ANSWERABLE:
        return (
            M0_CONFIDENT_AND_SUPPORTED
            if level == CONFIDENCE_HIGH
            else M0_SUPPORTED_BUT_NUANCED
        )
    if state == CONFLICTING_EVIDENCE:
        return M0_CONFLICTING_OR_STALE
    if state == INSUFFICIENT_EVIDENCE:
        return M0_NO_EVIDENCE if not has_evidence else M0_LOW_EVIDENCE
    return None


def _build_confidence(
    state: str, band: str, components: dict[str, Any], has_evidence: bool
) -> ConfidenceAssessment:
    """Categorical confidence; every component stays visible and explainable."""
    if state != ANSWERABLE:
        level = CONFIDENCE_LOW
    elif band != RELEVANCE_HIGH:
        level = CONFIDENCE_MEDIUM
    elif not components["has_current_authoritative_evidence"]:
        level = CONFIDENCE_MEDIUM
    else:
        level = CONFIDENCE_HIGH
    return ConfidenceAssessment(
        level=level,
        relevance_band=band,
        components=components,
        m0_level=_m0_confidence_level(state, level, has_evidence),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _no_results_assessment(query: str) -> EvidenceAssessment:
    """Build an assessment when retrieval returned nothing."""
    empties: list[EvidenceItem] = []
    confidence = ConfidenceAssessment(
        level=CONFIDENCE_LOW,
        relevance_band=RELEVANCE_LOW,
        components={
            "best_semantic_score": 0.0,
            "best_lexical_score": 0.0,
            "relevant_record_count": 0,
            "record_count": 0,
            "has_current_authoritative_evidence": False,
        },
        m0_level=M0_NO_EVIDENCE,
    )
    return EvidenceAssessment(
        query=query,
        state=INSUFFICIENT_EVIDENCE,
        routing=ROUTE_DECLINE,
        confidence=confidence,
        evidence=empties,
        conflicts=[],
        ambiguity=AmbiguityAssessment(
            detected=False, reason="no retrieval results", flags=[], evidence_ids=[]
        ),
        security=SecurityAssessment(
            triggered=False, matched_terms=[], rule="", evidence_ids_with_pii_rules=[]
        ),
                escalation=EscalationAssessment(required=False, reasons=[]),
        coverage=CoverageAssessment(
            query_topics=[],
            covered_topics=[],
            uncovered_topics=[],
            relevant_record_count=0,
            has_current_authoritative_evidence=False,
            stale_only_for_queried_topics=False,
        ),
        explanation=f"query: {query!r}\nstate: {INSUFFICIENT_EVIDENCE} | no retrieval results",
        results=[],
    )


def assess_evidence(query: str, results: Sequence[RetrievalResult]) -> EvidenceAssessment:
    """Assess retrieved evidence for a query (M3 public entry point).

    Takes a user query and the M2 RetrievalResult list, and returns a structured
    EvidenceAssessment describing what the evidence covers, conflicts with, escalates
    to, or fails to support.

    This layer does NOT generate a customer-facing answer, does NOT call an LLM, and
    does NOT decide the final answer --- it only classifies the evidence so M4 can
    route and (only if appropriate) generate from it.

    The retrieved results are never mutated or deleted: evidence is preserved for
    M3/M4 inspection, including stale records (M0 "authority != relevance" and
    "freshness" non-decisions).
    """
    if not results:
        return _no_results_assessment(query)

    query_topics = detect_query_topics(query)
    day_counts = extract_day_counts(query)
    items = [_build_evidence_item(result, query_topics) for result in results]
    best_semantic = max((item.semantic_score for item in items), default=0.0)
    best_lexical = max((item.lexical_score for item in items), default=0.0)
    band = _relevance_band(best_semantic)

    conflicts = _detect_conflicts(items, query_topics)
    security = _build_security(query, items)
    ambiguity = _build_ambiguity(query, query_topics, items)
    escalation_reasons = _escalation_reasons(query, query_topics, items, day_counts)
    coverage = _build_coverage(query_topics, items)

    has_items = len(items) > 0
    has_conflicts = bool(conflicts)

    # An escalated ticket in the evidence only escalates when it is query-relevant
    # (shares a contradiction topic with the query); the reason is documented so
    # escalation.required=True always has an explanation (M0 §12.2).
    aligned_escalated = sorted(
        item.source_id
        for item in items
        if item.escalated and item.relevant_topics
    )
    if aligned_escalated:
        escalation_reasons = list(escalation_reasons) + [
            {
                "flag": "escalated_ticket_evidence",
                "reason": (
                    "query-relevant evidence includes ticket(s) that the corpus "
                    "itself escalated; comparable cases need human review"
                ),
                "evidence_ids": aligned_escalated,
            }
        ]
    escalation_required = bool(escalation_reasons)

    components = {
        "best_semantic_score": best_semantic,
        "best_lexical_score": best_lexical,
        "relevant_record_count": len([i for i in items if i.relevant_topics]),
        "record_count": len(items),
        "has_current_authoritative_evidence": coverage.has_current_authoritative_evidence,
    }

    state = _decide_state(
        security_triggered=security.triggered,
        ambiguity_detected=ambiguity.detected,
        band=band,
        has_items=has_items,
        coverage=coverage,
        has_conflicts=has_conflicts,
        escalation_reasons=escalation_reasons,
    )
    routing = _routing_for(state, escalation_required)
    confidence = _build_confidence(state, band, components, has_evidence=has_items)

    esc_assessment = EscalationAssessment(
        required=escalation_required,
        reasons=escalation_reasons,
    )
    explanation = _build_explanation(
        query=query,
        state=state,
        routing=routing,
        confidence=confidence,
        items=items,
        conflicts=conflicts,
        ambiguity=ambiguity,
        security=security,
        escalation=esc_assessment,
        coverage=coverage,
    )

    return EvidenceAssessment(
        query=query,
        state=state,
        routing=routing,
        confidence=confidence,
        evidence=items,
        conflicts=list(conflicts),
        ambiguity=ambiguity,
        security=security,
        escalation=esc_assessment,
        coverage=coverage,
        explanation=explanation,
        results=list(results),
    )


def _build_explanation(
    query: str,
    state: str,
    routing: str,
    confidence: ConfidenceAssessment,
    items: Sequence[EvidenceItem],
    conflicts: Sequence[ConflictIndicator],
    ambiguity: AmbiguityAssessment,
    security: SecurityAssessment,
    escalation: EscalationAssessment,
    coverage: CoverageAssessment,
) -> str:
    """Developer-facing explanation for logs/debugging (NOT a customer answer)."""
    lines: list[str] = []
    lines.append(f"query: {query!r}")
    lines.append(
        f"state: {state} | routing: {routing} | confidence: {confidence.level} (heuristic)"
    )
    lines.append(
        f"relevance band: {confidence.relevance_band} "
        f"(best semantic {confidence.components['best_semantic_score']:.4f}, "
        f"best lexical {confidence.components['best_lexical_score']:.4f}, "
        f"{confidence.components['relevant_record_count']} query-relevant records)"
    )
    lines.append(
        f"authority present: {sorted({item.authority for item in items}, reverse=True)} | "
        f"freshness classes: {sorted({item.freshness_class for item in items})}"
    )
    if coverage.query_topics:
        lines.append(
            f"query topics: {coverage.query_topics} | covered: {coverage.covered_topics} | "
            f"uncovered: {coverage.uncovered_topics} | "
            f"stale-only: {coverage.stale_only_for_queried_topics}"
        )
    else:
        lines.append("query topics: none recognized")
    if conflicts:
        for conflict in conflicts:
            lines.append(f"conflict [{','.join(conflict.classes)}]: {conflict.description}")
    else:
        lines.append("conflicts: none detected among query-relevant evidence")
    lines.append(f"ambiguity: detected={ambiguity.detected} ({ambiguity.reason})")
    lines.append(
        f"security: triggered={security.triggered} terms={security.matched_terms} "
        f"(PII-rule evidence: {security.evidence_ids_with_pii_rules})"
    )
    if escalation.reasons:
        for reason in escalation.reasons:
            lines.append(f"escalation [{reason['flag']}]: {reason['reason']}")
    else:
        lines.append("escalation: none")
    lines.append(f"evidence ids: {', '.join(item.source_id for item in items)}")
    return "\n".join(lines)