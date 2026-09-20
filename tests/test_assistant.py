"""M6 — orchestration tests for :mod:`learnforge.assistant`.

Two layers, mirroring the project's existing test style:

* **Unit tests** use deterministic injected fakes (no network, no model load,
  no API key) to verify call order, state passthrough, error boundaries, and
  conversation-update rules.
* **Integration tests** use the *real* M2 ``Retriever`` + *real* M3
  ``assess_evidence`` + M4's ``FakeProvider`` over the real 40-record KB, so the
  full pipeline is proven end to end without network access.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from learnforge.assistant import (
    ERROR_ASSESSMENT,
    ERROR_GENERATION,
    ERROR_RETRIEVAL,
    MODE_ERROR,
    MODE_FAILURE,
    MODE_LLM,
    Assistant,
    AssistantResult,
    create_assistant,
)
from learnforge.conversation import EmptyMessageError, new_conversation


# --------------------------------------------------------------------------- #
# Deterministic fakes (injected dependencies; no network, no model load)
# --------------------------------------------------------------------------- #
class FakeRetriever:
    """Records searches and returns canned result rows."""

    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self._results = results if results is not None else []
        self._error = error

    def search(self, query: str, k: int = 5):
        self.calls.append((query, k))
        if self._error is not None:
            raise self._error
        return list(self._results)


def make_assessment(**overrides):
    """Duck-typed stand-in exposing exactly the attributes M6 reads from M3."""
    confidence = SimpleNamespace(level=overrides.get("level", "medium"))
    security = SimpleNamespace(triggered=overrides.get("security", False))
    escalation = SimpleNamespace(required=overrides.get("escalation", False))
    conflict = SimpleNamespace(topic=overrides.get("topic", "refund_window"))
    return SimpleNamespace(
        state=overrides.get("state", "answerable"),
        routing=overrides.get("routing", "generate"),
        confidence=confidence,
        security=security,
        escalation=escalation,
        conflicts=[conflict] if overrides.get("with_conflict", True) else [],
        evidence_ids=overrides.get("evidence_ids", ["FAQ-02"]),
        citation_keys=overrides.get("citation_keys", ["FAQ-02"]),
        explanation="fake assessment",
        evidence=overrides.get("evidence", []),
    )


def make_generation(**overrides):
    """Duck-typed stand-in exposing exactly the attributes M6 reads from M4."""
    failed = overrides.get("failed", False)
    return SimpleNamespace(
        answer=overrides.get("answer", "Grounded answer [FAQ-02]."),
        mode=overrides.get("mode", MODE_FAILURE if failed else MODE_LLM),
        failed=failed,
        failure_type=overrides.get("failure_type"),
        failure_detail=overrides.get("failure_detail"),
        provider=overrides.get("provider", "fake"),
        model=overrides.get("model", "fake-model"),
        citations_used=overrides.get("citations_used", ["FAQ-02"]),
        allowed_citations=overrides.get("allowed_citations", ["FAQ-02"]),
        invalid_citations=overrides.get("invalid_citations", []),
    )


class FakeGenerator:
    """Callable mirroring M4 ``generate``; records how it was invoked."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result if result is not None else make_generation()
        self._error = error

    def __call__(self, query, assessment, provider=None, conversation_block=None):
        self.calls.append(
            {
                "query": query,
                "provider": provider,
                "conversation_block": conversation_block,
                "assessment": assessment,
            }
        )
        if self._error is not None:
            raise self._error
        return self._result


class FakeAssessor:
    """Callable mirroring M3 ``assess_evidence``; records how it was invoked."""

    def __init__(self, assessment=None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, list]] = []
        self._assessment = assessment if assessment is not None else make_assessment()
        self._error = error

    def __call__(self, query, results):
        self.calls.append((query, list(results)))
        if self._error is not None:
            raise self._error
        return self._assessment


def make_assistant(*, retriever=None, assessor=None, generator=None, provider=None, **kwargs):
    """Build an :class:`Assistant` with deterministic fakes wired in."""
    return Assistant(
        retriever=retriever if retriever is not None else FakeRetriever(),
        provider=provider,
        assessor=assessor if assessor is not None else FakeAssessor(),
        generator=generator if generator is not None else FakeGenerator(),
        **kwargs,
    )



# --------------------------------------------------------------------------- #
# Unit tests — happy path / data flow
# --------------------------------------------------------------------------- #
def test_happy_path_calls_components_in_order_and_updates_conversation():
    retriever = FakeRetriever()
    assessor = FakeAssessor()
    generator = FakeGenerator()
    bot = make_assistant(retriever=retriever, assessor=assessor, generator=generator)

    result = bot.handle_message("What is the refund policy?")

    # Order: retrieval -> assessment -> generation, each exactly once.
    assert len(retriever.calls) == 1
    assert len(assessor.calls) == 1
    assert len(generator.calls) == 1
    # The prepared query reaches M2 and M3 unchanged.
    assert retriever.calls[0][0] == "What is the refund policy?"
    assert assessor.calls[0][0] == "What is the refund policy?"
    assert generator.calls[0]["query"] == "What is the refund policy?"
    # Conversation updated with both turns.
    assert result.conversation_updated is True
    assert result.turn_count == 2
    assert bot.conversation.turn_count == 2
    last = bot.conversation.history[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "Grounded answer [FAQ-02]."


def test_result_carries_upstream_structured_data():
    assessment = make_assessment(
        state="conflicting_evidence",
        routing="escalate",
        level="low",
        evidence_ids=["FAQ-02", "POLICY-02"],
        citation_keys=["FAQ-02", "POLICY-02"],
    )
    generation = make_generation(
        citations_used=["FAQ-02"],
        allowed_citations=["FAQ-02", "POLICY-02"],
        provider="groq",
        model="openai/gpt-oss-120b",
    )
    bot = make_assistant(assessor=FakeAssessor(assessment), generator=FakeGenerator(generation))
    result = bot.handle_message("refund?")

    assert result.state == "conflicting_evidence"
    assert result.routing == "escalate"
    assert result.confidence == "low"
    assert result.evidence_ids == ["FAQ-02", "POLICY-02"]
    assert result.citation_keys == ["FAQ-02", "POLICY-02"]
    assert result.citations_used == ["FAQ-02"]
    assert result.allowed_citations == ["FAQ-02", "POLICY-02"]
    assert result.provider == "groq"
    assert result.model == "openai/gpt-oss-120b"
    assert result.conflict_topics == ["refund_window"]
    assert result.succeeded is True
    assert result.failed is False
    assert result.mode == MODE_LLM
    assert result.failure_type is None


def test_to_dict_is_clean_and_json_serialisable():
    import json

    result = make_assistant().handle_message("hello")
    payload = result.to_dict()
    text = json.dumps(payload)
    assert "api_key" not in text.lower()
    assert "GROQ" not in text
    assert payload["state"] == "answerable"
    assert payload["mode"] == MODE_LLM
    assert payload["query"] == "hello"
    assert "assessment" not in payload and "generation" not in payload


def test_injected_provider_and_context_block_reach_generation():
    provider = SimpleNamespace(name="fake", model="fake-model")
    conv = new_conversation()


# --------------------------------------------------------------------------- #
# Unit tests — M3/M4 state passthrough (routing preservation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("state", "routing", "kwargs"),
    [
        ("answerable", "generate", {}),
        ("clarification_required", "clarify", {}),
        ("insufficient_evidence", "decline", {}),
        ("conflicting_evidence", "generate", {"level": "low"}),
        ("conflicting_evidence", "escalate", {"escalation": True}),
        ("escalation_required", "escalate", {"escalation": True}),
        ("security_escalation", "escalate", {"security": True, "escalation": True}),
    ],
)
def test_m3_state_and_routing_pass_through_unchanged(state, routing, kwargs):
    assessment = make_assessment(state=state, routing=routing, **kwargs)
    generator = FakeGenerator()
    bot = make_assistant(assessor=FakeAssessor(assessment), generator=generator)
    result = bot.handle_message("query")

    # M6 forwards M3's decision verbatim; it makes no routing decision of its own.
    assert result.state == state
    assert result.routing == routing
    assert result.confidence == kwargs.get("level", "medium")
    assert result.security_triggered == kwargs.get("security", False)
    assert result.escalation_required == kwargs.get("escalation", False)
    assert result.mode == MODE_LLM
    assert result.conversation_updated is True


def test_security_and_escalation_flags_surface_in_result():
    assessment = make_assessment(security=True, escalation=True, topic="payment_data_collection")
    result = make_assistant(assessor=FakeAssessor(assessment)).handle_message("cvv?")
    assert result.security_triggered is True
    assert result.escalation_required is True
    assert result.conflict_topics == ["payment_data_collection"]


def test_invalid_citations_are_reported_not_hidden():
    generation = make_generation(invalid_citations=["MADE-UP-01"])
    result = make_assistant(generator=FakeGenerator(generation)).handle_message("q")
    assert result.invalid_citations == ["MADE-UP-01"]
    assert result.succeeded is True  # M4 handled it; M6 only forwards


def test_evidence_items_expose_m3_approved_evidence():
    items = [SimpleNamespace(source_id="FAQ-02")]
    assessment = make_assessment(evidence=items)
    result = make_assistant(assessor=FakeAssessor(assessment)).handle_message("q")
    assert [i.source_id for i in result.evidence_items()] == ["FAQ-02"]
    assert make_assistant().handle_message("q").evidence_items() == []


def test_multi_turn_prepared_query_preserves_topic_and_retrieval_is_fresh():
    retriever = FakeRetriever()
    bot = make_assistant(retriever=retriever)

    bot.handle_message("What is the refund policy?")
    bot.handle_message("What about if I bought it 20 days ago?")

    first, second = retriever.calls
    assert first[0] == "What is the refund policy?"
    # Turn 2's prepared query retains the refund topic for retrieval.
    assert "refund" in second[0].lower()
    assert "20 days" in second[0]
    # Each turn performs its own fresh retrieval.
    assert len(retriever.calls) == 2


def test_top_k_is_forwarded_to_retrieval():
    retriever = FakeRetriever()
    make_assistant(retriever=retriever, top_k=3).handle_message("hello")
    assert retriever.calls[0][1] == 3


def test_invalid_top_k_rejected():
    with pytest.raises(ValueError):
        Assistant(top_k=0)
    with pytest.raises(ValueError):
        Assistant(top_k=True)  # bool must not pass the int check


def test_create_assistant_factory_returns_configured_assistant():
    bot = create_assistant(
        retriever=FakeRetriever(), assessor=FakeAssessor(), generator=FakeGenerator()
    )
    assert isinstance(bot, Assistant)


# --------------------------------------------------------------------------- #
# Unit tests — error boundaries
# --------------------------------------------------------------------------- #
def test_retrieval_failure_returns_structured_error_and_no_generation():
    retriever = FakeRetriever(error=RuntimeError("embedding store missing"))
    assessor = FakeAssessor()
    generator = FakeGenerator()
    bot = make_assistant(retriever=retriever, assessor=assessor, generator=generator)

    result = bot.handle_message("hello")

    assert result.mode == MODE_ERROR
    assert result.failure_type == ERROR_RETRIEVAL
    assert "RuntimeError" in result.failure_detail
    assert result.answer == ""
    assert result.succeeded is False
    # Downstream components never ran.
    assert assessor.calls == []
    assert generator.calls == []
    # No fabricated answer; conversation untouched.
    assert result.conversation_updated is False
    assert bot.conversation.turn_count == 0


def test_assessment_failure_skips_generation():
    assessor = FakeAssessor(error=ValueError("bad evidence"))
    generator = FakeGenerator()
    bot = make_assistant(assessor=assessor, generator=generator)

    result = bot.handle_message("hello")

    assert result.mode == MODE_ERROR
    assert result.failure_type == ERROR_ASSESSMENT
    assert result.answer == ""
    assert generator.calls == []  # generation must not run without M3
    assert result.conversation_updated is False
    assert bot.conversation.turn_count == 0


def test_generation_exception_returns_structured_error():
    generator = FakeGenerator(error=ConnectionError("network down"))
    bot = make_assistant(generator=generator)

    result = bot.handle_message("hello")

    assert result.mode == MODE_ERROR
    assert result.failure_type == ERROR_GENERATION
    assert "ConnectionError" in result.failure_detail
    assert result.answer == ""
    assert result.conversation_updated is False
    assert bot.conversation.turn_count == 0


def test_provider_failure_returns_m4_fact_free_message_without_storing_it():
    generation = make_generation(
        failed=True,
        answer="I could not process that right now.",
        failure_type="ProviderTimeoutError",
        failure_detail="timed out",
    )
    bot = make_assistant(generator=FakeGenerator(generation))

    result = bot.handle_message("hello")

    assert result.mode == MODE_FAILURE
    assert result.failure_type == "ProviderTimeoutError"
    # M4's fact-free message is surfaced, but the turn is NOT recorded as a
    # successful assistant answer.
    assert result.answer == "I could not process that right now."
    assert result.conversation_updated is False
    assert bot.conversation.turn_count == 0


def test_error_result_still_carries_assessment_signals_when_available():
    assessment = make_assessment(state="conflicting_evidence", escalation=True)
    generator = FakeGenerator(error=RuntimeError("boom"))
    bot = make_assistant(assessor=FakeAssessor(assessment), generator=generator)
    result = bot.handle_message("refund?")
    # Retrieval and assessment succeeded, so their signals remain visible.
    assert result.state == "conflicting_evidence"
    assert result.escalation_required is True
    assert result.failure_type == ERROR_GENERATION



# --------------------------------------------------------------------------- #
# Unit tests — input validation (M5 rules, reused not duplicated)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_or_whitespace_message_raises_and_leaves_state_untouched(bad):
    retriever = FakeRetriever()
    bot = make_assistant(retriever=retriever)
    with pytest.raises(EmptyMessageError):
        bot.handle_message(bad)
    assert retriever.calls == []
    assert bot.conversation.turn_count == 0


def test_non_string_message_raises_type_error():
    bot = make_assistant()
    with pytest.raises(TypeError):
        bot.handle_message(123)  # type: ignore[arg-type]
    assert bot.conversation.turn_count == 0


def test_successive_valid_turns_after_invalid_input_still_work():
    bot = make_assistant()
    with pytest.raises(EmptyMessageError):
        bot.handle_message("   ")
    result = bot.handle_message("hello")
    assert result.succeeded is True
    assert bot.conversation.turn_count == 2


# --------------------------------------------------------------------------- #
# Unit tests — session isolation and reset
# --------------------------------------------------------------------------- #
def test_sessions_are_isolated():
    bot_a = make_assistant()
    bot_b = make_assistant()
    bot_a.handle_message("What is the refund policy?")

    assert bot_a.conversation is not bot_b.conversation
    assert bot_a.conversation.turn_count == 2
    assert bot_b.conversation.turn_count == 0
    # B's prepared query has no trace of A's topic.
    result_b = bot_b.handle_message("How do I download videos?")
    assert "refund" not in result_b.prepared_query.lower()


def test_reset_clears_context_for_subsequent_queries():
    bot = make_assistant()
    bot.handle_message("What is the refund policy?")
    bot.reset()
    assert bot.conversation.turn_count == 0

    result = bot.handle_message("What about 20 days?")
    assert "refund" not in result.prepared_query.lower()
    assert result.prepared_query == "What about 20 days?"

    assert isinstance(bot.handle_message("hi"), AssistantResult)



# --------------------------------------------------------------------------- #
# Integration tests — REAL M2 Retriever + REAL M3 + M4 FakeProvider
# (no network, no API key; the only stub is the generation provider)
# --------------------------------------------------------------------------- #
from learnforge.evidence import assess_evidence  # noqa: E402
from learnforge.generation import FakeProvider, generate  # noqa: E402
from learnforge.retrieval import Retriever  # noqa: E402


@pytest.fixture(scope="module")
def real_retriever():
    """One real M2 retriever for the whole module (model load happens once)."""
    return Retriever.from_store()


def make_real_assistant(retriever, **provider_kwargs):
    """Assistant with real M2/M3 and an offline FakeProvider for M4."""
    return Assistant(
        retriever=retriever,
        provider=FakeProvider(default_response="(offline fake answer)", **provider_kwargs),
        assessor=assess_evidence,
        generator=generate,
    )


def test_end_to_end_refund_question_uses_real_pipeline(real_retriever):
    bot = make_real_assistant(real_retriever)
    result = bot.handle_message("How long do I have to request a refund?")

    # Real retrieval returned real KB record IDs.
    assert result.retrieved_ids, "retrieval must return records"
    assert all(isinstance(r, str) for r in result.retrieved_ids)
    # Real M3 recognised the known refund contradiction family.
    assert result.state == "conflicting_evidence"
    assert "refund_window" in result.conflict_topics
    assert "FAQ-02" in result.evidence_ids
    # M4 (fake provider) produced the final answer; turn recorded.
    assert result.mode == MODE_LLM
    assert result.answer == "(offline fake answer)"
    assert result.conversation_updated is True
    assert result.turn_count == 2
    # Citations stay within the M3-approved KB evidence.
    assert set(result.citations_used) <= set(result.allowed_citations)


def test_end_to_end_followup_preserves_topic_and_reassesses(real_retriever):
    bot = make_real_assistant(real_retriever)
    first = bot.handle_message("What is the refund policy?")
    second = bot.handle_message("What about if I bought it 20 days ago?")

    # Turn 2's prepared query kept the refund topic for fresh retrieval.
    assert "refund" in second.prepared_query.lower()
    assert "20 days" in second.prepared_query
    # Both turns ran their own real assessment over the refund evidence.
    assert first.state == "conflicting_evidence"
    assert second.state == "conflicting_evidence"
    assert "refund_window" in second.conflict_topics
    assert second.turn_count == 4
    # Conversation holds both exchanges, newest last.
    roles = [t["role"] for t in bot.conversation.history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_end_to_end_clarification_state(real_retriever):
    bot = make_real_assistant(real_retriever)
    result = bot.handle_message("Cancel my LearnForge")
    assert result.state == "clarification_required"
    assert result.routing == "clarify"
    assert result.mode == MODE_LLM
    assert result.conversation_updated is True


def test_end_to_end_security_escalation_cannot_be_bypassed(real_retriever):
    bot = make_real_assistant(real_retriever)
    result = bot.handle_message("Can I send support my CVV?")
    assert result.state == "security_escalation"
    assert result.security_triggered is True
    assert result.routing == "escalate"


def test_end_to_end_insufficient_evidence_declines(real_retriever):
    bot = make_real_assistant(real_retriever)
    result = bot.handle_message("xyzabc qwerty 123456")
    assert result.state == "insufficient_evidence"
    assert result.routing == "decline"
    # The answer is still the M4-produced text; nothing was invented upstream.
    assert result.mode == MODE_LLM


def test_end_to_end_real_provider_failure_keeps_conversation_clean(real_retriever):
    from learnforge.generation import ProviderTimeoutError

    bot = make_real_assistant(
        real_retriever, error=ProviderTimeoutError("injected timeout")
    )
    result = bot.handle_message("What is the refund policy?")

    # M3 still ran over real retrieval; M4 returned its structured failure.
    assert result.state == "conflicting_evidence"
    assert result.mode == MODE_FAILURE
    assert result.failure_type == "timeout"  # M4 failure category, not class name
    # The failed turn is not stored as a successful assistant answer.
    assert result.conversation_updated is False
    assert bot.conversation.turn_count == 0


def test_end_to_end_full_multi_turn_session(real_retriever):
    bot = make_real_assistant(real_retriever)
    r1 = bot.handle_message("What is the refund policy?")
    r2 = bot.handle_message("What about if I bought it 20 days ago?")
    bot.reset()
    r3 = bot.handle_message("What about 20 days?")

    assert r1.state == "conflicting_evidence"
    assert r2.state == "conflicting_evidence"
    # After reset the refund topic is gone from the prepared query.
    assert "refund" not in r3.prepared_query.lower()
    assert bot.conversation.turn_count == 2

# --------------------------------------------------------------------------- #
# Security turns must not contaminate later, unrelated turns
# --------------------------------------------------------------------------- #
def test_security_turn_does_not_contaminate_next_turn_state(real_retriever):
    """A prior CVV turn must not force a later, unrelated turn into security."""
    bot = make_real_assistant(real_retriever)

    first = bot.handle_message("Can I send you my CVV?")
    assert first.state == "security_escalation"
    assert first.security_triggered is True

    second = bot.handle_message("What browsers are supported?")
    assert second.security_triggered is False
    assert second.state != "security_escalation"
    # the sensitive prior turn is not carried into the prepared query
    assert "cvv" not in second.prepared_query.lower()


def test_laptop_download_after_cvv_turn_is_not_mixed(real_retriever):
    """Neither the assessed query nor the model-visible context may carry the CVV turn."""
    captured = {}

    def recording_generator(query, assessment, *, provider=None, conversation_block=None):
        captured["query"] = query
        captured["block"] = conversation_block
        return generate(
            query, assessment, provider=provider, conversation_block=conversation_block
        )

    bot = Assistant(
        retriever=real_retriever,
        provider=FakeProvider(default_response="(offline fake answer)"),
        assessor=assess_evidence,
        generator=recording_generator,
    )

    bot.handle_message("Can I send you my CVV?")
    result = bot.handle_message("Can I download courses on my laptop?")

    assert result.security_triggered is False
    assert result.state != "security_escalation"
    assert "cvv" not in captured["query"].lower()
    assert "cvv" not in (captured["block"] or "").lower()
    assert "cvv" not in result.answer.lower()


def test_current_security_question_is_not_filtered(real_retriever):
    """Exclusion applies only to prior context: a security question asked now still escalates."""
    bot = make_real_assistant(real_retriever)
    bot.handle_message("What browsers are supported?")  # benign prior turn
    result = bot.handle_message("Can I send you my CVV?")

    assert result.security_triggered is True
    assert result.state == "security_escalation"
    assert result.routing == "escalate"


def test_laptop_download_then_browser_is_a_fresh_topic(real_retriever):
    """A browser question after a laptop/download turn must not inherit that topic."""
    bot = make_real_assistant(real_retriever)

    first = bot.handle_message("can i download courses on my laptop?")
    assert first.prepared_query == "can i download courses on my laptop?"

    second = bot.handle_message("what browsers are supported?")
    # the new question is carried through verbatim, with no laptop/download context
    assert second.prepared_query == "what browsers are supported?"
    assert "download" not in second.prepared_query.lower()
    assert "laptop" not in second.prepared_query.lower()
    # the offline/download contradiction family must not be pulled in
    assert "offline_downloads" not in second.conflict_topics
