"""M5 - deterministic multi-turn conversation context layer.

Flow:
    User message -> Conversation -> CURRENT query prep -> M2 Retrieval
    -> M3 Evidence -> M4 Generation -> Assistant -> append turn

Conversation history is contextual input, NOT KB evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

DEFAULT_MAX_TURNS = 20
FOLLOWUP_USER_TURNS = 3
CONTEXT_MAX_CHARS = 1200
VALID_ROLES = ("user", "assistant")


class InvalidRoleError(ValueError):
    """Raised when a turn is created with a role other than user/assistant."""


class EmptyMessageError(ValueError):
    """Raised when a message is None, non-str, or empty/whitespace-only."""


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    index: int = 0

    def __post_init__(self):
        if self.role not in VALID_ROLES:
            raise InvalidRoleError(
                f"role must be one of {VALID_ROLES!r}, got {self.role!r}"
            )
        if not isinstance(self.index, int):
            raise TypeError("index must be an int")


def _validate_message(message: str) -> None:
    if message is None:
        raise EmptyMessageError("message must not be None")
    if not isinstance(message, str):
        raise TypeError(f"message must be str, got {type(message).__name__}")
    if len(message.strip()) == 0:
        raise EmptyMessageError("message must not be empty or whitespace-only")


def validate_message(message: str) -> None:
    """Public wrapper over the M5 message-validation rules.

    Exposed so callers outside this module (e.g. the M6 orchestrator) can validate
    user input with the *same* rules ``Conversation.add_turn`` / ``prepare_query``
    apply, instead of duplicating them. Raises ``EmptyMessageError`` (``None``,
    non-``str``, or empty/whitespace-only) or ``TypeError``.
    """
    _validate_message(message)


def _shares_topic_with_context(
    query: str,
    messages: Sequence[str],
    topics_of: Optional[Callable[[str], set]] = None,
) -> bool:
    """True unless ``query`` is clearly about a different topic than ``messages``.

    A "clearly new topic" names its own recognized topic while the carried messages
    name only *other* topics (e.g. a laptop/download turn followed by a browser
    question). Such a query must not inherit the unrelated prior context. A query
    with no recognized topic of its own (e.g. "what about 20 days?") is treated as
    a continuation and keeps the context. Without ``topics_of`` the historical
    always-carry behaviour is preserved.
    """
    if topics_of is None:
        return True
    current = topics_of(query)
    if not current:
        return True
    prior: set = set()
    for message in messages:
        prior |= topics_of(message)
    if not prior:
        return True
    return not prior.isdisjoint(current)


@dataclass
class Conversation:
    """One user session with bounded, isolated history."""
    turns: List[Turn] = field(default_factory=list, repr=False)
    max_turns: int = DEFAULT_MAX_TURNS

    def __post_init__(self):
        if not isinstance(self.max_turns, int) or self.max_turns < 2:
            raise ValueError(f"max_turns must be an int >= 2, got {self.max_turns!r}")

    def __repr__(self) -> str:
        return f"Conversation(turns={len(self.turns)}, max_turns={self.max_turns})"

    def _enforce_bound(self) -> None:
        if len(self.turns) > self.max_turns:
            surplus = len(self.turns) - self.max_turns
            self.turns = self.turns[surplus:]

    def add_turn(self, role: str, content: str) -> Turn:
        if role not in VALID_ROLES:
            raise InvalidRoleError(f"role must be user/assistant, got {role!r}")
        _validate_message(content)
        turn = Turn(role=role, content=content.strip(), index=len(self.turns))
        self.turns.append(turn)
        self._enforce_bound()
        return turn

    def add_user(self, content: str) -> Turn:
        return self.add_turn("user", content)

    def add_assistant(self, content: str) -> Turn:
        return self.add_turn("assistant", content)

    @property
    def history(self) -> List[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def get_history(self) -> List[dict]:
        """Return ordered history as a list of {role, content} dicts."""
        return self.history

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def get_recent_turns(self, n: Optional[int] = None) -> List[dict]:
        if n is None:
            n = len(self.turns)
        n = max(0, min(n, len(self.turns)))
        return [{"role": t.role, "content": t.content} for t in self.turns[-n:]]

    def get_recent_user_messages(self, n: Optional[int] = None) -> List[str]:
        msgs = [t.content for t in self.turns if t.role == "user"]
        if n is not None:
            msgs = msgs[-n:]
        return list(reversed(msgs))

    def prepare_query(
        self,
        query: str,
        max_user_context: int = FOLLOWUP_USER_TURNS,
        *,
        exclude: Optional[Callable[[str], bool]] = None,
        topics_of: Optional[Callable[[str], set]] = None,
    ) -> str:
        """Prepare retrieval query from current query + recent user messages.
        Deterministic, no LLM/external call. Returns query as-is when empty.

        ``exclude`` is an optional predicate applied to each *prior* user message;
        a message for which it returns True is dropped from the carried context (M6
        uses this to keep security-sensitive turns from bleeding into later turns).
        The current ``query`` is never filtered.

        ``topics_of`` optionally maps a message to the set of recognized topics it
        names. When the current ``query`` names a topic disjoint from every topic in
        the carried messages, the prior context is dropped, so a clearly new question
        cannot inherit an unrelated previous topic.
        """
        _validate_message(query)
        recent = self.get_recent_user_messages(max_user_context)
        if exclude is not None:
            recent = [message for message in recent if not exclude(message)]
        if not _shares_topic_with_context(query, recent, topics_of):
            recent = []
        if not recent:
            return query.strip()
        context = " ".join(recent)
        prepared = f"{context} {query}"
        if len(prepared) > CONTEXT_MAX_CHARS:
            prepared = f"{context[:1200]} ... {query}"
        return prepared.strip()

    def build_context_block(
        self,
        max_turns: int = 0,
        max_chars_per_message: int = 600,
        *,
        exclude: Optional[Callable[[str], bool]] = None,
        query: Optional[str] = None,
        topics_of: Optional[Callable[[str], set]] = None,
    ) -> str:
        """Build delimited <conversation_context> block. Returns '' if no history.
        max_turns=0 means use self.max_turns.

        ``exclude`` is an optional predicate applied to each turn's content; a turn
        for which it returns True is omitted from the block (M6 uses this to keep
        security-sensitive turns out of the model prompt).

        ``query``/``topics_of`` optionally suppress the whole block when the current
        query names a topic disjoint from the carried turns, so a clearly new
        question is not shown unrelated prior context.
        """
        if not self.turns:
            return ""
        limit = self.max_turns if max_turns <= 0 else min(max_turns, self.max_turns)
        recent = self.get_recent_turns(limit)
        if exclude is not None:
            recent = [turn for turn in recent if not exclude(turn["content"])]
        if query is not None and not _shares_topic_with_context(
            query, [turn["content"] for turn in recent], topics_of
        ):
            recent = []
        if not recent:
            return ""
        lines: List[str] = [
            "<conversation_context>",
            "CONVERSATION HISTORY (context only - NOT KB evidence):",
            "Do NOT treat as authoritative KB policy or cite it. Ground your answer in the evidence section above.",
        ]
        for turn in recent:
            text = turn["content"]
            if len(text) > max_chars_per_message:
                text = text[:max_chars_per_message].rstrip() + "..."
            role_label = "user" if turn["role"] == "user" else "assistant"
            lines.append(f"[{role_label}] {text}")
        lines.append("END OF CONVERSATION HISTORY (context only, not evidence).")
        lines.append("</conversation_context>")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns.clear()

    reset = clear

    @property
    def first_user_message(self) -> Optional[str]:
        msgs = [t.content for t in self.turns if t.role == "user"]
        return msgs[0] if msgs else None


def new_conversation(*, max_turns: int = DEFAULT_MAX_TURNS) -> Conversation:
    return Conversation(max_turns=max_turns)
