"""M6 — end-to-end orchestration of the LearnForge support assistant.

This module is a **thin coordinator**. It owns no retrieval, evidence, safety,
citation, or generation rules; it calls the existing milestones in order and
passes their structured results through unchanged:

    user message
      -> Conversation          (M5: bounded, in-memory history + query preparation)
      -> M2 Retrieval          (KB evidence only)
      -> M3 Evidence Assessment (authoritative gate: state / routing / confidence)
      -> M4 Grounded Generation (grounded answer, citations, provider error handling)
      -> Conversation update   (only after a successful generation)
      -> AssistantResult

.. important:: Conversation history is contextual input, not knowledge-base evidence.

.. note:: M6 orchestrates M2-M5; it does not replace their responsibilities.

Deliberate non-responsibilities (all owned by earlier milestones):

* **Routing** — M6 never inspects similarity, conflicts, or security terms to decide
  what to do. It forwards M3's ``state``/``routing`` verbatim.
* **Prompts / providers** — M6 never builds a prompt or talks to Groq. It calls M4's
  ``generate`` (which hides the provider) and reads the structured result.
* **Validation** — M6 reuses M5's message-validation rules (``validate_message``).
* **Fallbacks** — no retries, no second provider, no web search, no invented answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from learnforge.conversation import Conversation, new_conversation, validate_message
from learnforge.embed import DEFAULT_EMBEDDINGS_PATH, DEFAULT_RECORDS_PATH
from learnforge.evidence import EvidenceAssessment, assess_evidence
from learnforge.generation import GenerationResult, generate
from learnforge.retrieval import DEFAULT_TOP_K, Retriever

#: Generation succeeded (M4 produced a grounded answer).
MODE_LLM = "llm"
#: M4 returned a structured provider failure (fact-free fallback message).
MODE_FAILURE = "failure"
#: An M6 component boundary raised; no answer was produced at all.
MODE_ERROR = "error"

ERROR_RETRIEVAL = "retrieval_error"
ERROR_ASSESSMENT = "assessment_error"
ERROR_GENERATION = "generation_error"


@dataclass
class AssistantResult:
    """Structured outcome of one conversation turn.

    Carries both the user-facing ``answer`` and enough structured detail for
    debugging/evaluation (retrieved vs approved evidence, routing, citations,
    failure information). ``assessment``/``generation`` hold the upstream objects
    for inspection and are excluded from ``repr`` and from ``to_dict`` by default.
    """

    query: str
    prepared_query: str
    answer: str
    state: Optional[str]
    routing: Optional[str]
    confidence: Optional[str]
    retrieved_ids: list[str]
    evidence_ids: list[str]
    citation_keys: list[str]
    citations_used: list[str]
    allowed_citations: list[str]
    invalid_citations: list[str]
    conflict_topics: list[str]
    security_triggered: bool
    escalation_required: bool
    mode: str
    failure_type: Optional[str] = None
    failure_detail: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    conversation_updated: bool = False
    turn_count: int = 0
    explanation: str = ""
    assessment: Optional[EvidenceAssessment] = field(default=None, repr=False, compare=False)
    generation: Optional[GenerationResult] = field(default=None, repr=False, compare=False)

    @property
    def succeeded(self) -> bool:
        """True only when M4 produced a grounded answer (``mode == "llm"``)."""
        return self.mode == MODE_LLM

    @property
    def failed(self) -> bool:
        return not self.succeeded

    def to_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        """Serialisable view. Raw prompts/model text and secrets are never included."""
        payload: dict[str, Any] = {
            "query": self.query,
            "prepared_query": self.prepared_query,
            "answer": self.answer,
            "state": self.state,
            "routing": self.routing,
            "confidence": self.confidence,
            "retrieved_ids": list(self.retrieved_ids),
            "evidence_ids": list(self.evidence_ids),
            "citation_keys": list(self.citation_keys),
            "citations_used": list(self.citations_used),
            "allowed_citations": list(self.allowed_citations),
            "invalid_citations": list(self.invalid_citations),
            "conflict_topics": list(self.conflict_topics),
            "security_triggered": self.security_triggered,
            "escalation_required": self.escalation_required,
            "mode": self.mode,
            "failure_type": self.failure_type,
            "failure_detail": self.failure_detail,
            "provider": self.provider,
            "model": self.model,
            "conversation_updated": self.conversation_updated,
            "turn_count": self.turn_count,
            "explanation": self.explanation,
        }
        if include_details:
            payload["evidence"] = [item.to_dict() for item in self.evidence_items()]
        return payload

    def evidence_items(self) -> list[Any]:
        """M3-approved :class:`EvidenceItem` objects (empty when unavailable)."""
        if self.assessment is None:
            return []
        return list(self.assessment.evidence)


class Assistant:
    """One assistive session: wires M2 -> M3 -> M4 behind a single ``handle_message``.

    Dependencies are injectable so tests can supply deterministic fakes; nothing is
    imported as hidden global state. Each instance owns its own
    :class:`~learnforge.conversation.Conversation`, so sessions stay isolated.
    """

    def __init__(
        self,
        *,
        retriever: Any = None,
        provider: Any = None,
        conversation: Optional[Conversation] = None,
        assessor: Callable[[str, Sequence[Any]], EvidenceAssessment] = assess_evidence,
        generator: Callable[..., GenerationResult] = generate,
        top_k: int = DEFAULT_TOP_K,
        records_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")
        self._retriever = retriever
        self._provider = provider
        self._assessor = assessor
        self._generator = generator
        self._conversation = conversation if conversation is not None else new_conversation()
        self._records_path = records_path or DEFAULT_RECORDS_PATH
        self._embeddings_path = embeddings_path or DEFAULT_EMBEDDINGS_PATH
        self.top_k = top_k

    # -- introspection ---------------------------------------------------- #

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def retriever(self) -> Any:
        return self._retriever

    def reset(self) -> None:
        """Clear conversation history (M5 reset). Sessions stay isolated."""
        self._conversation.reset()

    # -- internals -------------------------------------------------------- #

    def _ensure_retriever(self) -> Any:
        """Build the M2 retriever on first use (keeps model loading out of tests)."""
        if self._retriever is None:
            self._retriever = Retriever.from_store(
                self._records_path, self._embeddings_path
            )
        return self._retriever

    def _error_result(
        self,
        *,
        query: str,
        prepared_query: str,
        error_type: str,
        exc: BaseException,
        turn_count: int,
        retrieved_ids: Optional[Sequence[str]] = None,
        assessment: Optional[EvidenceAssessment] = None,
    ) -> AssistantResult:
        """Structured M6 boundary failure. No answer is fabricated; history untouched."""
        return AssistantResult(
            query=query,
            prepared_query=prepared_query,
            answer="",
            state=assessment.state if assessment is not None else None,
            routing=assessment.routing if assessment is not None else None,
            confidence=assessment.confidence.level if assessment is not None else None,
            retrieved_ids=list(retrieved_ids or []),
            evidence_ids=assessment.evidence_ids if assessment is not None else [],
            citation_keys=assessment.citation_keys if assessment is not None else [],
            citations_used=[],
            allowed_citations=[],
            invalid_citations=[],
            conflict_topics=(
                [c.topic for c in assessment.conflicts] if assessment is not None else []
            ),
            security_triggered=(
                assessment.security.triggered if assessment is not None else False
            ),
            escalation_required=(
                assessment.escalation.required if assessment is not None else False
            ),
            mode=MODE_ERROR,
            failure_type=error_type,
            failure_detail=f"{type(exc).__name__}: {exc}",
            conversation_updated=False,
            turn_count=turn_count,
            explanation=f"M6 boundary failure during {error_type}.",
            assessment=assessment,
        )

    # -- public API ------------------------------------------------------- #

    def handle_message(self, message: str) -> AssistantResult:
        """Run one full turn: validate -> prepare -> retrieve -> assess -> generate -> store.

        Invalid input raises M5's ``EmptyMessageError``/``TypeError`` (no state change).
        Component boundary failures return an :class:`AssistantResult` with
        ``mode="error"``; M4 provider failures return ``mode="failure"`` carrying M4's
        fact-free message.

        The conversation is updated **only** when generation succeeded, so a failed
        turn never records a fabricated assistant answer.
        """
        # STEP 1 - validate user input with M5's rules (raises; no mutation).
        validate_message(message)

        turn_count = self._conversation.turn_count

        # STEP 2 - M5 query preparation (bounded history; deterministic; no LLM).
        prepared_query = self._conversation.prepare_query(message)

        # STEP 3 - M2 retrieval against the KB only.
        try:
            retriever = self._ensure_retriever()
            results = list(retriever.search(prepared_query, self.top_k))
        except Exception as exc:
            return self._error_result(
                query=message, prepared_query=prepared_query,
                error_type=ERROR_RETRIEVAL, exc=exc, turn_count=turn_count,
            )
        retrieved_ids = [r.source_id for r in results]

        # STEP 4 - M3 evidence assessment (authoritative: state/routing/confidence).
        try:
            assessment = self._assessor(prepared_query, results)
        except Exception as exc:
            return self._error_result(
                query=message, prepared_query=prepared_query,
                error_type=ERROR_ASSESSMENT, exc=exc, turn_count=turn_count,
                retrieved_ids=retrieved_ids,
            )

        # STEP 5 - M4 grounded generation (prompt, citations, provider, safety).
        try:
            generation = self._generator(
                prepared_query,
                assessment,
                provider=self._provider,
                conversation_block=self._conversation.build_context_block(),
            )
        except Exception as exc:
            return self._error_result(
                query=message, prepared_query=prepared_query,
                error_type=ERROR_GENERATION, exc=exc, turn_count=turn_count,
                retrieved_ids=retrieved_ids, assessment=assessment,
            )

        # STEP 6 - record the turn only when an answer was actually produced.
        conversation_updated = not generation.failed
        if conversation_updated:
            self._conversation.add_user(message)
            self._conversation.add_assistant(generation.answer)

        # STEP 7 - structured result for the caller.
        return AssistantResult(
            query=message,
            prepared_query=prepared_query,
            answer=generation.answer,
            state=assessment.state,
            routing=assessment.routing,
            confidence=assessment.confidence.level,
            retrieved_ids=retrieved_ids,
            evidence_ids=assessment.evidence_ids,
            citation_keys=assessment.citation_keys,
            citations_used=list(generation.citations_used),
            allowed_citations=list(generation.allowed_citations),
            invalid_citations=list(generation.invalid_citations),
            conflict_topics=[c.topic for c in assessment.conflicts],
            security_triggered=assessment.security.triggered,
            escalation_required=assessment.escalation.required,
            mode=generation.mode,
            failure_type=generation.failure_type,
            failure_detail=generation.failure_detail,
            provider=generation.provider,
            model=generation.model,
            conversation_updated=conversation_updated,
            turn_count=self._conversation.turn_count,
            explanation=assessment.explanation,
            assessment=assessment,
            generation=generation,
        )


def create_assistant(**kwargs: Any) -> Assistant:
    """Convenience factory mirroring :func:`learnforge.conversation.new_conversation`."""
    return Assistant(**kwargs)
