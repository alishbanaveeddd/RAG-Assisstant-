"""Automated tests for the M5 multi-turn conversation context layer (pytest).

Run with::

    python -m pytest tests/test_conversation.py -q

All tests are deterministic and require no API key or network access. They
verify that conversation history is used only as context for query preparation
and prompt construction -- never as authoritative KB evidence or citation source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from learnforge.conversation import (
    DEFAULT_MAX_TURNS,
    Conversation,
    EmptyMessageError,
    InvalidRoleError,
    Turn,
    new_conversation,
)
from learnforge.evidence import (
    ANSWERABLE,
    CLARIFICATION_REQUIRED,
    CONFLICTING_EVIDENCE,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    ESCALATION_REQUIRED,
    INSUFFICIENT_EVIDENCE,
    SECURITY_ESCALATION,
    EvidenceAssessment,
)
from learnforge.generation import (
    FakeProvider,
    PromptBundle,
    build_prompt,
    generate,
)
from learnforge.generation import SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retriever():
    from learnforge.retrieval import Retriever

    return Retriever.from_store(
        str(REPO_ROOT / "data" / "processed" / "kb_records.json"),
        str(REPO_ROOT / "data" / "processed" / "kb_embeddings.json"),
    )


def make_assessment(
    query, state=ANSWERABLE, routing=None, items=None, **kw
):
    """Build a minimal EvidenceAssessment for controlled prompt tests."""
    from learnforge.evidence import (
        AmbiguityAssessment,
        ConfidenceAssessment,
        CoverageAssessment,
        EscalationAssessment,
        SecurityAssessment,
        ROUTE_GENERATE,
    )

    if items is None:
        items = []
    if routing is None:
        routing = ROUTE_GENERATE
    return EvidenceAssessment(
        query=query,
        state=state,
        routing=routing,
        confidence=ConfidenceAssessment(
            level=kw.get("level", CONFIDENCE_HIGH),
            relevance_band=kw.get("band", "high"),
            components={
                "best_semantic_score": 0.9,
                "best_lexical_score": 0.0,
                "relevant_record_count": len(items),
                "record_count": len(items),
                "has_current_authoritative_evidence": bool(items),
            },
            m0_level=kw.get("m0_level"),
        ),
        evidence=list(items),
        conflicts=[],
        ambiguity=AmbiguityAssessment(
            detected=False, reason="test", flags=[], evidence_ids=[]
        ),
        security=SecurityAssessment(
            triggered=False, matched_terms=[], rule="",
            evidence_ids_with_pii_rules=[],
        ),
        escalation=EscalationAssessment(required=False, reasons=[]),
        coverage=CoverageAssessment(
            query_topics=[], covered_topics=[], uncovered_topics=[],
            relevant_record_count=len(items),
            has_current_authoritative_evidence=bool(items),
            stale_only_for_queried_topics=False,
        ),
        explanation="test explanation",
    )




# ---- Basic behavior tests ---- #


def test_conversation_creation():
    c = new_conversation()
    assert c.turn_count == 0
    assert c.get_history() == []


def test_add_user_turn():
    c = new_conversation()
    turn = c.add_user("What is the refund policy?")
    assert turn.role == "user"
    assert turn.content == "What is the refund policy?"
    assert c.turn_count == 1


def test_add_assistant_turn():
    c = new_conversation()
    c.add_assistant("Here is what I found...")
    assert c.turn_count == 1
    assert c.history[0]["role"] == "assistant"


def test_chronological_ordering():
    c = new_conversation()
    c.add_user("first question")
    c.add_assistant("first answer")
    c.add_user("second question")
    c.add_assistant("second answer")
    history = c.get_history()
    assert len(history) == 4
    assert history[0]["content"] == "first question"
    assert history[1]["role"] == "assistant"
    assert history[2]["content"] == "second question"
    assert history[3]["role"] == "assistant"


def test_retrieve_history():
    c = new_conversation()
    c.add_user("Q1")
    c.add_assistant("A1")
    history = c.get_history()
    assert len(history) == 2
    assert {"role": "user", "content": "Q1"} in history
    assert {"role": "assistant", "content": "A1"} in history


# ---- Validation ---- #


def test_invalid_role_rejected():
    c = new_conversation()
    with pytest.raises(InvalidRoleError):
        c.add_turn("system", "you are a helpful bot")
    with pytest.raises(InvalidRoleError):
        c.add_turn("tool", "x")
    assert c.turn_count == 0


def test_turn_invalid_role_rejected():
    with pytest.raises(InvalidRoleError):
        Turn(role="system", content="nope")


def test_empty_message_rejected():
    c = new_conversation()
    for bad in ["", "   ", "\t\n  "]:
        with pytest.raises(EmptyMessageError):
            c.add_user(bad)
    assert c.turn_count == 0


def test_none_and_non_str_message_rejected():
    c = new_conversation()
    with pytest.raises(EmptyMessageError):
        c.add_user(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        c.add_user(123)  # type: ignore[arg-type]


def test_message_is_stripped():
    c = new_conversation()
    turn = c.add_user("  padded question  ")
    assert turn.content == "padded question"


def test_invalid_max_turns_rejected():
    with pytest.raises(ValueError):
        Conversation(max_turns=1)
    with pytest.raises(ValueError):
        Conversation(max_turns="lots")  # type: ignore[arg-type]


# ---- Bounded history ---- #


def test_history_is_capped():
    c = new_conversation(max_turns=4)
    for i in range(10):
        c.add_user(f"q{i}")
    assert c.turn_count == 4


def test_newest_turns_retained():
    c = new_conversation(max_turns=4)
    for i in range(6):
        c.add_user(f"q{i}")
    contents = [t["content"] for t in c.get_history()]
    assert contents == ["q2", "q3", "q4", "q5"]


def test_bound_keeps_chronological_order():
    c = new_conversation(max_turns=3)
    c.add_user("q1")
    c.add_assistant("a1")
    c.add_user("q2")
    c.add_assistant("a2")
    assert [t["content"] for t in c.get_history()] == ["a1", "q2", "a2"]


def test_default_max_turns_is_20():
    assert DEFAULT_MAX_TURNS == 20
    c = new_conversation()
    for i in range(30):
        c.add_user(f"q{i}")
    assert c.turn_count == DEFAULT_MAX_TURNS


# ---- Context construction ---- #


def test_prepare_query_preserves_prior_topic():
    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("The knowledge base contains conflicting refund terms.")
    prepared = c.prepare_query("What about 20 days?")
    assert "refund" in prepared.lower()
    assert "20 days" in prepared
    # the assistant answer is context, never part of the prepared retrieval query
    assert "conflicting refund terms" not in prepared


def test_prepare_query_empty_history_returns_query():
    c = new_conversation()
    assert c.prepare_query("How long do I have to request a refund?") == (
        "How long do I have to request a refund?"
    )


def test_prepare_query_is_bounded():
    c = new_conversation(max_turns=20)
    for i in range(10):
        c.add_user("x" * 4000)
    prepared = c.prepare_query("final question")
    assert len(prepared) <= 2000 + len("final question")


def test_context_block_delimits_and_labels_history():
    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("There are conflicting refund terms.")
    block = c.build_context_block()
    assert block.startswith("<conversation_context>")
    assert block.endswith("</conversation_context>")
    assert "NOT KB evidence" in block
    assert "[user] What is the refund policy?" in block
    assert "[assistant] There are conflicting refund terms." in block


def test_context_block_empty_when_no_history():
    c = new_conversation()
    assert c.build_context_block() == ""


def test_context_block_truncates_long_messages():
    c = new_conversation()
    c.add_user("y" * 5000)
    block = c.build_context_block(max_chars_per_message=100)
    assert "y" * 101 not in block


# ---- No evidence leakage ---- #


def test_history_is_not_a_citation_source():
    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("See [FAQ-99] for details.")
    block = c.build_context_block()
    # history is fenced as context only, explicitly not evidence
    assert "NOT KB evidence" in block
    assert "<evidence>" not in block


# ---- Session isolation ---- #


def test_sessions_do_not_share_state():
    a = new_conversation()
    b = new_conversation()
    a.add_user("What is the refund policy?")
    a.add_assistant("conflicting refund terms")
    assert b.turn_count == 0
    assert b.prepare_query("What about 20 days?") == "What about 20 days?"


def test_independent_instances_do_not_mutate_each_other():
    a = new_conversation()
    b = new_conversation()
    b.add_user("How do I download videos?")
    assert a.first_user_message is None
    assert b.first_user_message == "How do I download videos?"


# ---- Reset ---- #


def test_reset_clears_history():
    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("conflicting refund terms")
    c.clear()
    assert c.turn_count == 0
    assert c.get_history() == []
    assert c.build_context_block() == ""


def test_reset_removes_prior_context_from_followup():
    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("conflicting refund terms")
    c.reset()
    prepared = c.prepare_query("What about 20 days?")
    assert "refund" not in prepared.lower()
    assert prepared == "What about 20 days?"


# ---- M3 routing preservation ---- #


def test_conversation_block_does_not_change_answerable_routing(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("(previous answer)")
    prepared = c.prepare_query("How long do I have to request a refund?")
    results = retriever.search(prepared, k=5)
    assessment = assess_evidence(prepared, results)
    block = c.build_context_block()
    with_ctx = build_prompt(assessment, conversation_block=block)
    without_ctx = build_prompt(assessment)
    # The M3-selected task instruction is identical with/without conversation context.
    assert with_ctx.state_instruction == without_ctx.state_instruction
    # The context block is appended to the user message, outside the evidence packet.
    assert block in with_ctx.user
    assert "<conversation_context>" in with_ctx.user
    assert with_ctx.user.index("<evidence>") < with_ctx.user.index("<conversation_context>")
    assert with_ctx.allowed_citations == without_ctx.allowed_citations


def test_ambiguous_followup_stays_clarification(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    c.add_user("I want to cancel.")
    c.add_assistant("Could you clarify what you want to cancel?")
    prepared = c.prepare_query("Do it now.")
    results = retriever.search(prepared, k=5)
    assessment = assess_evidence(prepared, results)
    # history must not resolve the ambiguity: M3 remains authoritative
    assert assessment.state == CLARIFICATION_REQUIRED


def test_security_followup_stays_security_escalation(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    c.add_user("I need help with my payment.")
    c.add_assistant("I can help with billing questions.")
    prepared = c.prepare_query("Can I send you my CVV?")
    results = retriever.search(prepared, k=5)
    assessment = assess_evidence(prepared, results)
    assert assessment.state == SECURITY_ESCALATION


def test_insufficient_followup_stays_insufficient(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    c.add_user("Do you support feature X?")
    c.add_assistant("I don't have that information.")
    prepared = c.prepare_query("What about on the enterprise plan?")
    results = retriever.search(prepared, k=5)
    assessment = assess_evidence(prepared, results)
    assert assessment.state == INSUFFICIENT_EVIDENCE


# ---- Integration: Conversation -> M2 -> M3 -> M4 ---- #


def test_full_pipeline_uses_current_evidence_not_history(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    # Turn 1
    c.add_user("What is the refund policy?")
    c.add_assistant("The KB contains conflicting refund terms.")
    # Turn 2: prepared retrieval query keeps the refund topic
    prepared = c.prepare_query("What about if I bought it 20 days ago?")
    assert "refund" in prepared.lower()

    # M2 retrieval runs fresh against the KB
    results = retriever.search(prepared, k=5)
    assert results

    # M3 assessment is authoritative
    assessment = assess_evidence(prepared, results)
    assert assessment.state in {
        ANSWERABLE,
        CLARIFICATION_REQUIRED,
        CONFLICTING_EVIDENCE,
        INSUFFICIENT_EVIDENCE,
        ESCALATION_REQUIRED,
        SECURITY_ESCALATION,
    }

    # M4 generation with a deterministic fake provider
    fake = FakeProvider(default_response="Grounded response [POLICY-02]")
    block = c.build_context_block()
    result = generate(
        prepared, assessment, provider=fake, conversation_block=block
    )
    assert result.answer
    # citations come only from M3-approved KB evidence
    for cite in result.citations_used:
        assert cite in result.allowed_citations
    # conversation history is never injected into the answer text
    assert "conversation_context" not in result.answer
    assert "<conversation_context>" in fake.last_user_prompt


def test_pipeline_insufficient_produces_no_fabricated_citations(retriever):
    from learnforge.evidence import assess_evidence

    c = new_conversation()
    c.add_user("Do you support feature X?")
    c.add_assistant("I don't have that information.")
    prepared = c.prepare_query("What about on the enterprise plan?")
    results = retriever.search(prepared, k=5)
    assessment = assess_evidence(prepared, results)
    assert assessment.state == INSUFFICIENT_EVIDENCE
    # A model reply that invents a citation must be stripped: TICKET-03 is not
    # among the M3-approved evidence for this query.
    fake = FakeProvider(default_response="You definitely get 30 days [TICKET-03].")
    result = generate(
        prepared, assessment, provider=fake,
        conversation_block=c.build_context_block(),
    )
    # M3 keeps the retrieved (weak) evidence in the packet, so allowed citations
    # come only from that packet -- never from conversation history.
    assert "TICKET-03" not in result.allowed_citations
    assert result.citations_used == []
    assert "TICKET-03" in result.invalid_citations
    for cite in result.allowed_citations:
        assert cite.startswith(("FAQ-", "POLICY-", "TICKET-"))


# ---- Security content must never become carried context ---- #


def test_prepare_query_excludes_security_sensitive_history():
    """A prior CVV turn is dropped from the carried context of a later turn."""
    from learnforge.evidence import detect_security

    c = new_conversation()
    c.add_user("what's cvv can u give me my cvv")
    c.add_assistant("For your safety, please don't share CVV/CVC codes.")

    prepared = c.prepare_query(
        "What browsers are supported?",
        exclude=lambda message: detect_security(message)[0],
    )

    # the sensitive prior turn is not carried forward ...
    assert "cvv" not in prepared.lower()
    # ... but the current question is always kept verbatim.
    assert prepared.endswith("What browsers are supported?")


def test_prepare_query_without_exclude_still_carries_benign_topics():
    """Regression guard: the exclusion predicate must not drop benign context."""
    from learnforge.evidence import detect_security

    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("Refunds follow the policy.")

    prepared = c.prepare_query(
        "What about 20 days?",
        exclude=lambda message: detect_security(message)[0],
    )

    assert "refund" in prepared.lower()
    assert "20 days" in prepared


def test_context_block_excludes_security_sensitive_turns():
    """The model-visible context block must not contain the prior CVV exchange."""
    from learnforge.evidence import detect_security

    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("Refunds follow the policy.")
    c.add_user("Can I send you my CVV?")
    c.add_assistant("For your safety, never share card numbers, CVV/CVC codes or PINs.")

    block = c.build_context_block(exclude=lambda text: detect_security(text)[0])

    assert "cvv" not in block.lower()
    assert "card numbers" not in block.lower()
    # benign history survives, and the block stays properly delimited
    assert block.startswith("<conversation_context>")
    assert block.endswith("</conversation_context>")
    assert "refund policy" in block.lower()


# ---- New topics must not inherit unrelated context ---- #


def test_new_topic_question_drops_unrelated_context():
    """A question naming a new topic must not inherit unrelated prior context."""
    from learnforge.evidence import detect_query_topics

    c = new_conversation()
    c.add_user("can i download courses on my laptop?")
    c.add_assistant("Downloads are covered by the offline-access policy.")

    prepared = c.prepare_query("what browsers are supported?", topics_of=detect_query_topics)

    # the laptop/download turn is a different topic: nothing is carried
    assert prepared == "what browsers are supported?"
    assert "download" not in prepared.lower()
    assert "laptop" not in prepared.lower()


def test_related_topic_followup_still_carries_context():
    """A related follow-up keeps the prior topic (refund -> 'after 20 days')."""
    from learnforge.evidence import detect_query_topics

    c = new_conversation()
    c.add_user("What is the refund policy?")
    c.add_assistant("Refunds follow the policy.")

    prepared = c.prepare_query("what about after 20 days?", topics_of=detect_query_topics)

    assert "refund" in prepared.lower()
    assert "20 days" in prepared


def test_current_query_is_never_dropped_by_context_gating():
    """Gating applies to prior context only: the current query always survives."""
    from learnforge.evidence import detect_query_topics, detect_security

    c = new_conversation()
    c.add_user("What browsers are supported?")
    c.add_assistant("Chrome, Edge, Firefox and Safari.")

    prepared = c.prepare_query(
        "Can I send you my CVV?",
        exclude=lambda message: detect_security(message)[0],
        topics_of=detect_query_topics,
    )

    assert "cvv" in prepared.lower()
    assert prepared.strip().endswith("Can I send you my CVV?")


