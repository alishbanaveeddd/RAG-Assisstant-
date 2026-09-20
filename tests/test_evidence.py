"""Automated tests for the M3 evidence assessment layer (pytest).

Run with::

    python -m pytest tests -q

Scope: deterministic evidence assessment only. These tests assert what M3
*classifies* about retrieved evidence — sufficiency, conflicts, ambiguity,
security, escalation — never what M4 should answer. No LLM/API calls occur.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from learnforge.evidence import (
    ANSWERABLE,
    CLARIFICATION_REQUIRED,
    CONFLICTING_EVIDENCE,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ESCALATION_REQUIRED,
    INSUFFICIENT_EVIDENCE,
    M0_CONFIDENT_AND_SUPPORTED,
    M0_CONFLICTING_OR_STALE,
    M0_LOW_EVIDENCE,
    M0_NO_EVIDENCE,
    M0_SUPPORTED_BUT_NUANCED,
    RELEVANCE_LOW,
    ROUTE_CLARIFY,
    ROUTE_DECLINE,
    ROUTE_ESCALATE,
    ROUTE_GENERATE,
    SECURITY_ESCALATION,
    ASSESSMENT_STATES,
    _item_claims,
    detect_query_ambiguity,
    detect_query_topics,
    extract_day_counts,
    assess_evidence,
)
from learnforge.retrieval import Retriever, RetrievalResult
from learnforge.schema import AUTHORITY_BY_TYPE

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "learnforge-knowledge-base-data" / "learnforge-knowledge-base"
SOURCE_FILES = ("faqs.md", "policies.md", "tickets.md")

RECORDS_PATH = REPO_ROOT / "data" / "processed" / "kb_records.json"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "kb_embeddings.json"

MANUAL_QUERIES = (
    "How long do I have to request a refund?",
    "Can I get a refund after 20 days?",
    "Can I download courses on my laptop?",
    "Does LearnForge support Internet Explorer?",
    "Cancel my LearnForge",
    "Can I send support my CVV?",
    "xyzabc qwerty 123456",
)


def make_result(source_id, text, source_type=None, metadata=None,
                semantic=0.8, lexical=0.0, fused=None):
    """Minimal RetrievalResult matching the M1/M2 schema shape."""
    if source_type is None:
        source_type = (
            "faq" if source_id.startswith("FAQ")
            else "ticket" if source_id.startswith("TICKET")
            else "policy"
        )
    if metadata is None:
        metadata = {
            "freshness": "undated",
            "is_stale": False,
            "stale_reason": None,
            "authority": AUTHORITY_BY_TYPE[source_type],
            "ticket_status": "Resolved" if source_type == "ticket" else None,
        }
    return RetrievalResult(
        record={
            "source_id": source_id,
            "source_type": source_type,
            "title": source_id,
            "chunk_text": text,
            "citation_key": source_id,
            "metadata": metadata,
            "vector": None,
        },
        semantic_score=semantic,
        lexical_score=lexical,
        fused_score=fused if fused is not None else max(semantic, lexical),
    )


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_store(str(RECORDS_PATH), str(EMBEDDINGS_PATH))


def assess(retriever, query, k=5):
    return assess_evidence(query, retriever.search(query, k))


# --------------------------------------------------------------------------- #
# 1: state vocabulary
# --------------------------------------------------------------------------- #

def test_assessment_states_are_defined():
    assert set(ASSESSMENT_STATES) == {
        ANSWERABLE,
        CLARIFICATION_REQUIRED,
        CONFLICTING_EVIDENCE,
        INSUFFICIENT_EVIDENCE,
        ESCALATION_REQUIRED,
        SECURITY_ESCALATION,
    }
    from learnforge.evidence import _ROUTING_BY_STATE
    assert set(_ROUTING_BY_STATE) == set(ASSESSMENT_STATES)


# --------------------------------------------------------------------------- #
# 2-4: authoritative / stale / undated recognition
# --------------------------------------------------------------------------- #

def test_current_authoritative_evidence_is_recognized(retriever):
    a = assess(retriever, "How long do I have to request a refund?")
    policy = next(i for i in a.evidence if i.source_id == "POLICY-02")
    assert policy.authority == AUTHORITY_BY_TYPE["policy"] == 3
    assert "current" in policy.claims.get("refund_window", [])
    assert a.coverage.has_current_authoritative_evidence


def test_stale_evidence_is_recognized_not_deleted(retriever):
    a = assess(retriever, "Does LearnForge support Internet Explorer?")
    policy09 = next(i for i in a.evidence if i.source_id == "POLICY-09")
    assert policy09.freshness_class == "stale"
    assert policy09.is_stale is True
    assert policy09.stale_reason
    assert "POLICY-09" in a.evidence_ids


def test_undated_evidence_is_not_falsely_marked_stale(retriever):
    a = assess(retriever, "How long do I have to request a refund?")
    faq02 = next(i for i in a.evidence if i.source_id == "FAQ-02")
    assert faq02.freshness == "undated"
    assert faq02.is_stale is False
    assert faq02.freshness_class == "undated"


# --------------------------------------------------------------------------- #
# 5: authority preservation, separate from relevance
# --------------------------------------------------------------------------- #

def test_authority_levels_are_preserved_per_source_type(retriever):
    for query in MANUAL_QUERIES[:5]:
        a = assess(retriever, query)
        for item in a.evidence:
            assert item.authority == AUTHORITY_BY_TYPE[item.source_type]


def test_relevant_ticket_does_not_override_policy(retriever):
    a = assess(retriever, "Can I get a refund after 20 days?")
    by_id = {item.source_id: item for item in a.evidence}
    policy, ticket = by_id.get("POLICY-02"), by_id.get("TICKET-03")
    assert policy is not None and ticket is not None
    assert policy.authority > ticket.authority
    conflict = next(c for c in a.conflicts if c.family == "refund_window")
    assert "POLICY-02" in conflict.current_evidence_ids
    assert "TICKET-03" in conflict.exception_evidence_ids


def test_stale_policy_is_not_treated_as_current(retriever):
    a = assess(retriever, "How long do I have to request a refund?")
    policy02 = next(i for i in a.evidence if i.source_id == "POLICY-02")
    assert policy02.authority == 3
    assert policy02.freshness_class == "stale"
    assert "stale" in policy02.claims["refund_window"]
    conflict = next(c for c in a.conflicts if c.family == "refund_window")
    assert "POLICY-02" in conflict.stale_evidence_ids


# --------------------------------------------------------------------------- #
# 6-7: contradiction families, current vs stale distinguishable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query,family", [
    ("How long do I have to request a refund?", "refund_window"),
    ("Can I get a refund after 20 days?", "refund_window"),
    ("Can I download courses on my laptop?", "offline_downloads"),
    ("Does LearnForge support Internet Explorer?", "browser_support"),
    ("Are captions required for every video before publication?", "captions_accessibility"),
    ("Is my course progress saved instantly?", "progress_sync"),
])
def test_conflict_detection_fires_for_known_families(retriever, query, family):
    a = assess(retriever, query)
    conflict = next((c for c in a.conflicts if c.family == family), None)
    assert conflict is not None, (query, [c.family for c in a.conflicts])
    assert conflict.current_evidence_ids, "conflict without a current side"
    assert conflict.stale_evidence_ids or conflict.exception_evidence_ids


def test_refund_conflict_exposes_m0_conflict_classes():
    from learnforge.evidence import _detect_conflicts, _build_evidence_item
    meta = {
        "freshness": "Effective date: January 2026",
        "is_stale": False,
        "stale_reason": None,
        "authority": 3,
        "ticket_status": None,
        "contradiction_topics": ["refund_window"],
    }
    ticket_meta = dict(meta, authority=1, contradiction_topics=["refund_window"])
    items = [
        _build_evidence_item(make_result(
            "POLICY-02",
            "The standard refund window is 14 days. Older documentation "
            "referred to a 7-day refund period; that wording is outdated.",
            metadata=meta,
        ), {"refund_window"}),
        _build_evidence_item(make_result(
            "TICKET-03",
            "User cited a 30-day money-back guarantee shown at purchase.",
            source_type="ticket",
            metadata=ticket_meta,
        ), {"refund_window"}),
    ]
    conflicts = _detect_conflicts(items, {"refund_window"})
    conflict = next(c for c in conflicts if c.family == "refund_window")
    assert "current_vs_stale" in conflict.classes
    assert "user_claim_vs_current_policy" in conflict.classes
    assert conflict.current_evidence_ids == ["POLICY-02"]
    assert conflict.stale_evidence_ids == ["POLICY-02"]
    assert conflict.exception_evidence_ids == ["TICKET-03"]


def test_current_and_stale_sides_are_distinguishable(retriever):
    """FAQ-07 legitimately quotes both the stale laptop claim and the current
    mobile-only guidance (M0 §11.1: one record may carry both sides). The
    ConflictIndicator still separates the sides across records."""
    a = assess(retriever, "Can I download courses on my laptop?")
    conflict = next(c for c in a.conflicts if c.family == "offline_downloads")
    assert conflict.current_evidence_ids == ["FAQ-07", "TICKET-15"]
    assert conflict.stale_evidence_ids == ["FAQ-07", "TICKET-15"]
    faq07 = next(i for i in a.evidence if i.source_id == "FAQ-07")
    assert set(faq07.claims["offline_downloads"]) == {"stale", "current"}


def test_payment_conflict_uses_retired_practice_as_stale_side(retriever):
    a = assess(retriever, "What payment information can support ask me for?")
    conflict = next((c for c in a.conflicts if c.family == "payment_data_collection"), None)
    assert conflict is not None
    assert "POLICY-10" in conflict.stale_evidence_ids
    assert "FAQ-15" in conflict.current_evidence_ids


def test_claims_extraction_for_captions_and_sync():
    assert "stale" in _item_claims("captions for every video before publication")["captions_accessibility"]
    assert "current" in _item_claims("captions whenever practical")["captions_accessibility"]
    assert "stale" in _item_claims("progress saved instantly")["progress_sync"]
    assert "current" in _item_claims("progress automatically synchronized")["progress_sync"]
    assert "stale" in _item_claims("annual subscriptions were billed monthly")["annual_billing"]
    assert "current" in _item_claims("the annual price shown at checkout")["annual_billing"]


def test_query_topic_detection():
    assert detect_query_topics("Can I get a refund after 20 days?") == {"refund_window"}
    assert detect_query_topics("Does LearnForge support Internet Explorer?") == {"browser_support"}
    assert detect_query_topics("Cancel my LearnForge") == set()


def test_day_count_extraction():
    assert extract_day_counts("Can I get a refund after 20 days?") == [20]
    assert extract_day_counts("refund after 7 days or 30 days?") == [7, 30]
    assert extract_day_counts("no numbers here") == []


# --------------------------------------------------------------------------- #
# 8: ambiguity / clarification
# --------------------------------------------------------------------------- #

def test_ambiguous_cancellation_requires_clarification(retriever):
    a = assess(retriever, "Cancel my LearnForge")
    assert a.state == CLARIFICATION_REQUIRED
    assert a.routing == ROUTE_CLARIFY
    assert a.ambiguity.detected is True
    assert "ambiguous_intent_requires_clarification" in a.ambiguity.flags
    assert "TICKET-07" in a.ambiguity.evidence_ids


def test_specific_requests_are_not_ambiguous():
    topics = detect_query_topics("Cancel my annual subscription")
    detected, _ = detect_query_ambiguity("Cancel my annual subscription", topics)
    assert detected is False


def test_informational_questions_are_not_ambiguous():
    detected, _ = detect_query_ambiguity("How do I cancel?", set())
    assert detected is False


def test_ambiguity_requires_evidence_alignment(retriever):
    """A query can look heuristic-ambiguous, but M3 only fires when the
    retrieved evidence itself carries the TICKET-07 classification."""
    ambiguous = "close my stuff"
    topics = detect_query_topics(ambiguous)
    detected, _ = detect_query_ambiguity(ambiguous, topics)
    assert detected is True  # heuristic fires...
    results = [make_result("FAQ-01", "How to contact support", semantic=0.6)]
    a = assess_evidence(ambiguous, results)
    assert a.ambiguity.detected is False  # ...but no flagged evidence -> no state
    assert a.state != CLARIFICATION_REQUIRED


# --------------------------------------------------------------------------- #
# 9: insufficient evidence
# --------------------------------------------------------------------------- #

def test_nonsense_query_returns_insufficient_evidence(retriever):
    a = assess(retriever, "xyzabc qwerty 123456")
    assert a.state == INSUFFICIENT_EVIDENCE
    assert a.routing == ROUTE_DECLINE
    assert a.confidence.relevance_band == RELEVANCE_LOW
    assert a.confidence.m0_level == M0_LOW_EVIDENCE
    assert a.coverage.relevant_record_count == 0


def test_empty_retrieval_returns_insufficient_evidence():
    a = assess_evidence("anything", [])
    assert a.state == INSUFFICIENT_EVIDENCE
    assert a.routing == ROUTE_DECLINE
    assert a.evidence == []
    assert a.confidence.m0_level == M0_NO_EVIDENCE


def test_weak_relevance_is_insufficient_even_with_results():
    results = [make_result("FAQ-01", "How do I reset my password?", semantic=0.2)]
    a = assess_evidence("why is the sky blue", results)
    assert a.state == INSUFFICIENT_EVIDENCE
    assert a.confidence.m0_level == M0_LOW_EVIDENCE


def test_only_stale_evidence_for_a_policy_question_is_flagged():
    """If nothing current supports the queried topic, M3 reports stale-only."""
    stale_meta = {
        "freshness": "Last reviewed: February 2026",
        "is_stale": True,
        "stale_reason": "older wording",
        "authority": 3,
        "ticket_status": None,
        "contradiction_topics": ["refund_window"],
    }
    results = [
        make_result(
            "POLICY-02",
            "The older policy allowed refunds within 7 days of purchase.",
            metadata=stale_meta,
        )
    ]
    a = assess_evidence("How long do I have to request a refund?", results)
    assert a.coverage.stale_only_for_queried_topics is True
    assert a.state == INSUFFICIENT_EVIDENCE


def test_answerable_state_for_clean_supported_evidence():
    """A single current policy record with only a current claim is answerable."""
    results = [
        make_result(
            "POLICY-03",
            "Captions should be provided whenever practical for instructional videos.",
            metadata={
                "freshness": "Updated: March 2026",
                "is_stale": False,
                "stale_reason": None,
                "authority": 3,
                "ticket_status": None,
                "contradiction_topics": ["captions_accessibility"],
            },
            semantic=0.9,
        )
    ]
    a = assess_evidence("Are captions provided whenever practical?", results)
    assert a.state == ANSWERABLE
    assert a.routing == ROUTE_GENERATE
    assert a.confidence.level == CONFIDENCE_HIGH
    assert a.confidence.m0_level == M0_CONFIDENT_AND_SUPPORTED
    assert a.conflicts == []


# --------------------------------------------------------------------------- #
# 10: security / PII
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query,term", [
    ("Can I send support my CVV?", "cvv"),
    ("Should I give support my card number?", "card number"),
    ("Should I send my banking password?", "password"),
    ("Can I provide my authentication code?", "authentication code"),
])
def test_security_queries_trigger_security_escalation(retriever, query, term):
    a = assess(retriever, query)
    assert a.security.triggered is True
    assert a.state == SECURITY_ESCALATION
    assert a.routing == ROUTE_ESCALATE
    assert any(term in t for t in a.security.matched_terms)


def test_asking_what_support_may_request_is_not_security_sensitive(retriever):
    a = assess(retriever, "What payment information can support ask me for?")
    assert a.security.triggered is False
    assert a.state != SECURITY_ESCALATION


def test_security_assessment_names_pii_rule_evidence(retriever):
    a = assess(retriever, "Can I send support my CVV?")
    assert "POLICY-10" in a.security.evidence_ids_with_pii_rules
    assert "FAQ-15" in a.security.evidence_ids_with_pii_rules


# --------------------------------------------------------------------------- #
# 11-13: preservation, determinism, no-LLM
# --------------------------------------------------------------------------- #

def test_evidence_ids_and_citation_keys_are_preserved(retriever):
    for query in MANUAL_QUERIES[:6]:
        a = assess(retriever, query)
        assert a.evidence_ids
        assert a.citation_keys == a.evidence_ids
        for item in a.evidence:
            assert item.citation_key == item.source_id
            assert item.chunk_text


def test_retrieved_results_are_not_modified_or_deleted(retriever):
    query = "Can I get a refund after 20 days?"
    results = retriever.search(query, 10)
    before = [r.to_dict() for r in results]
    a = assess_evidence(query, results)
    after = [r.to_dict() for r in results]
    assert before == after
    assert {i.source_id for i in a.evidence} == {r.source_id for r in results}


def test_assessment_is_deterministic(retriever):
    for query in MANUAL_QUERIES:
        dicts = [assess(retriever, query).to_dict() for _ in range(3)]
        assert dicts[0] == dicts[1] == dicts[2], query


def test_module_imports_are_local_only():
    import ast
    src = (REPO_ROOT / "learnforge" / "evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    forbidden = {"openai", "anthropic", "google", "requests", "httpx", "urllib"}
    assert modules & forbidden == set()


def test_no_network_calls_at_runtime(retriever):
    import socket

    original = socket.socket
    try:
        class Blocked(original):
            def __init__(self, *args, **kwargs):
                raise OSError("network disabled for M3 test")

        socket.socket = Blocked
        a = assess(retriever, "How long do I have to request a refund?")
        assert a.state == CONFLICTING_EVIDENCE
    finally:
        socket.socket = original


# --------------------------------------------------------------------------- #
# 15-16: confidence explainability + explanation
# --------------------------------------------------------------------------- #

def test_confidence_is_categorical_with_exposed_components(retriever):
    a = assess(retriever, "How long do I have to request a refund?")
    assert a.confidence.level in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW}
    assert isinstance(a.confidence.components, dict) and a.confidence.components
    assert "NOT a calibrated" in a.confidence.note
    as_dict = a.confidence.to_dict()
    assert as_dict["level"] == a.confidence.level
    assert as_dict["components"] == a.confidence.components


def test_conflicts_and_confidence_are_independent_signals(retriever):
    a = assess(retriever, "Can I get a refund after 20 days?")
    assert a.conflicts
    assert a.confidence.level == CONFIDENCE_LOW


def test_explanation_is_developer_facing_and_explainable(retriever):
    a = assess(retriever, "Can I get a refund after 20 days?")
    assert a.explanation
    for needed in ("query:", "state:", "conflict", "ambiguity", "security", "escalation"):
        assert needed in a.explanation


def test_m0_confidence_level_is_populated(retriever):
    for query in MANUAL_QUERIES[:5]:
        a = assess(retriever, query)
        assert a.confidence.m0_level in {
            M0_CONFIDENT_AND_SUPPORTED, M0_SUPPORTED_BUT_NUANCED,
            M0_CONFLICTING_OR_STALE, M0_LOW_EVIDENCE, M0_NO_EVIDENCE, None,
        }


# --------------------------------------------------------------------------- #
# 14: dataset-specific regression checks
# --------------------------------------------------------------------------- #

def test_annual_subscription_conflict_and_alignment(retriever):
    a = assess(retriever, "Are refunds available for annual subscriptions?")
    assert any(c.family == "refund_window" for c in a.conflicts)
    flags = {r["flag"] for r in a.escalation.reasons}
    assert flags & {"policy_wording_ambiguity", "escalated_ticket_evidence"}


def test_offline_download_contradiction_is_recognized(retriever):
    a = assess(retriever, "Can I download courses on my laptop?")
    conflict = next(c for c in a.conflicts if c.family == "offline_downloads")
    assert "FAQ-07" in conflict.stale_evidence_ids
    assert a.coverage.query_topics == ["offline_downloads"]


def test_unresolved_and_escalated_flags_survive(retriever):
    a = assess(retriever, "Can I get a refund after 20 days?")
    ticket03 = next(i for i in a.evidence if i.source_id == "TICKET-03")
    assert ticket03.escalated is True
    assert ticket03.unresolved is True
    assert ticket03.ticket_status == "Escalated"


def test_security_rule_metadata_survives_into_assessment(retriever):
    a = assess(retriever, "What payment information can support ask me for?")
    assert {"FAQ-04", "FAQ-15", "POLICY-10"} <= set(a.security.evidence_ids_with_pii_rules)


def test_day_driven_promotional_escalation_fires_only_outside_standard(retriever):
    inside = assess(retriever, "Can I get a refund after 10 days?")
    outside = assess(retriever, "Can I get a refund after 20 days?")
    inside_flags = {r["flag"] for r in inside.escalation.reasons}
    outside_flags = {r["flag"] for r in outside.escalation.reasons}
    assert "promotional_terms_may_differ" in outside_flags
    assert "promotional_terms_may_differ" not in inside_flags


def test_source_kb_files_still_exist_and_are_nonempty(retriever):
    for name in SOURCE_FILES:
        path = KB_DIR / name
        assert path.exists() and path.stat().st_size > 0





