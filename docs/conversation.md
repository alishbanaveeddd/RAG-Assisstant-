# M5 — Multi-turn Conversation Context

> **Scope:** M5 adds a small, deterministic **conversation layer**. It prepares
> context for follow-up questions and carries history between turns. It does **not**
> answer questions, retrieve, assess evidence, or call an LLM.
>
> **Core principle:** *Conversation history is contextual input, not knowledge-base
> evidence.*

---

## 1. Purpose

Allow follow-up questions such as:

```
User       : What is the refund policy?
Assistant  : The knowledge base contains conflicting refund terms...
User       : What if I bought it 20 days ago?
```

The second question must be interpreted **in the context of the conversation**, while
retrieval and evidence assessment are still performed **fresh against the knowledge
base for the current turn**.

M5 adds no new knowledge source. It only makes the current turn interpretable.

---

## 2. Flow

```
User message
      |
      v
Conversation context                 (M5: bounded history)
      |
      v
Current query preparation            (M5: deterministic, no LLM)
      |
      v
M2 Hybrid Retrieval                  (against the KB, fresh each turn)
      |
      v
M3 Evidence Assessment               (authoritative gate; M5 never overrides it)
      |
      v
M4 Grounded Generation               (receives delimited <conversation_context>)
      |
      v
Assistant response
      |
      v
Conversation history                 (append user + assistant turn)
```

The system never does `Conversation history -> answer directly`.

---

## 3. Data model

`learnforge/conversation.py` defines two types.

### `Turn` (frozen dataclass)

| Field | Type | Meaning |
|---|---|---|
| `role` | `str` | `"user"` or `"assistant"` only — anything else raises `InvalidRoleError` |
| `content` | `str` | Stripped message text; must be non-empty |
| `index` | `int` | Position in the conversation (0-based) |

### `Conversation` (dataclass)

| Member | Behavior |
|---|---|
| `turns: List[Turn]` | Chronological history (not printed in `repr`) |
| `max_turns: int` | Bound on retained turns (default `DEFAULT_MAX_TURNS = 20`) |
| `add_turn(role, content) -> Turn` | Validates role + message, appends, enforces bound |
| `add_user(content) -> Turn` | Convenience wrapper |
| `add_assistant(content) -> Turn` | Convenience wrapper |
| `history -> List[dict]` | Ordered `{"role", "content"}` list |
| `get_history() -> List[dict]` | Explicit accessor, same as `history` |
| `get_recent_turns(n=None)` | Last *n* turns (all if `None`) |
| `get_recent_user_messages(n=None)` | User messages, **most recent first** |
| `prepare_query(query, max_user_context=3)` | Bounded retrieval-query preparation |
| `build_context_block(...)` | Delimited `<conversation_context>` block for M4 |
| `clear()` / `reset` | Empties the conversation in place |
| `turn_count`, `first_user_message` | Small read-only helpers |

Only `user` and `assistant` roles are accepted. Arbitrary roles (e.g. `system`,
`tool`) are rejected — they are not conversation turns and must not be smuggled in as
one.

---

## 4. Session lifecycle

1. **Create** — `new_conversation()` (or `Conversation(max_turns=N)`); starts empty.
2. **Turn** — append the user turn, run the M2→M3→M4 pipeline for that turn, then append
   the assistant turn.
3. **Repeat** — each turn uses the accumulated (bounded) history as context only.
4. **Reset** — `reset()`/`clear()` drops all history for a fresh start.
5. **Isolate** — separate `Conversation` instances share no state (see §12).

No database, Redis, authentication, or persistence is involved. Instances live in
memory for the lifetime of the process.

---

## 5. Bounded history

`max_turns` (default **20**) caps retained turns. When exceeded, the **oldest** turns are
dropped first; ordering stays chronological.

- **Trade-off:** 20 turns is enough for realistic support clarifications (typically 2–6
  turns) while keeping the prompt bounded and deterministic. Too small loses the topic a
  follow-up depends on; too large inflates the prompt and the risk of the model treating
  stale assistant text as fact.
- **No summarization.** M5 deliberately does not summarize or compress with an LLM: it
  would cost an extra API call, add nondeterminism, and introduce a second place where
  the model could invent facts.

---

## 6. Follow-up context preparation

`prepare_query()` builds the retrieval query deterministically:

```
prepared = "<recent user messages, newest first> <current query>"
```

- Only **user** messages are used, and only the most recent `max_user_context` (default
  **3**). Assistant text is excluded on purpose — a previous answer must never shape
  retrieval as if it were user intent or fact.
- The result is capped (`CONTEXT_MAX_CHARS = 1200` for the prefix, 2000 overall); on
  overflow it is `"<prefix[:1200]> ... <current query>"`.
- With empty history it returns the stripped query unchanged.

Example:

```
history : user  "What is the refund policy?"
          asst  "The knowledge base contains conflicting refund terms..."
current : "What about 20 days?"
prepared: "What is the refund policy? What about 20 days?"
```

`build_context_block()` produces the **M4-facing** block:

```
<conversation_context>
CONVERSATION HISTORY (context only - NOT KB evidence):
Do NOT treat as authoritative KB policy or cite it. Ground your answer in the evidence section above.
[user] What is the refund policy?
[assistant] The knowledge base contains conflicting refund terms...
END OF CONVERSATION HISTORY (context only, not evidence).
</conversation_context>
```

Each message is truncated (`max_chars_per_message`, default 600). The block returns `""`
when there is no history, so an empty conversation adds nothing to the prompt.

---

## 7. How M2/M3/M4 are used

- **M2 Retrieval** is called with the **prepared** query. Retrieval always executes
  against the KB; history is only an input to the query string.
- **M3 Evidence Assessment** runs on the **current** retrieval results and remains the
  authoritative gate. M5 passes nothing into M3 and cannot change its verdict.
- **M4 Grounded Generation** receives `conversation_block=` (an optional keyword on
  `build_prompt()` / `generate()`). The block is inserted **after** `</evidence>` and
  **before** `<user_query>`, so instructions precede data and the context never sits
  inside the evidence packet.

Existing interfaces are unchanged: `conversation_block` is optional and defaults to
`None`, so all M1–M4 call sites and tests behave exactly as before.

---

## 8. Why history is not evidence

- The context block is explicitly labeled *"context only - NOT KB evidence"* and is
  placed outside `<evidence>`.
- `allowed_citations` for a turn is computed **only** from M3's evidence packet
  (`build_prompt` derives it from `assessment.evidence`). History cannot add a citation.
- Citations are enforced at generation time: any `[ID]` not in `allowed_citations` is
  stripped and recorded in `invalid_citations` — including IDs that appear only in
  conversation history.

## 9. Why a previous assistant answer is not authoritative

A prior assistant turn is a record of *what was said*, not current KB policy. If the KB
changes, or if the earlier answer was generated for a different sub-question, replaying
it as fact would be wrong. M5 therefore:

- excludes assistant text from `prepare_query()` entirely, and
- labels the block as non-evidence for the model, while M3 remains free to reach
  `insufficient_evidence` / `conflicting_evidence` regardless of a confident-sounding
  previous answer.

---

## 10. M3 routing preservation

All six M3 states are preserved and cannot be bypassed:

| State | Multi-turn behavior |
|---|---|
| `answerable` | Prepared query + fresh retrieval; answer grounded in current evidence |
| `clarification_required` | History does **not** resolve ambiguity (e.g. "Do it now." stays ambiguous) |
| `conflicting_evidence` | M3 still owns the conflict; disclosure/escalation per its routing |
| `insufficient_evidence` | Adding a vague follow-up ("What about the enterprise plan?") does not manufacture evidence |
| `escalation_required` | Follow-up text cannot suppress an escalation signal |
| `security_escalation` | A CVV/password/auth-code follow-up is still intercepted by M3's security rule |

---

## 11. Reset behavior

`reset()` (alias `clear()`) empties history in place. After reset, a follow-up such as
*"What about 20 days?"* is prepared with **no** prior topic:

```
before reset : "What is the refund policy? What about 20 days?"
        reset
after  reset : "What about 20 days?"
```

The new query must not inherit old context.

---

## 12. Session isolation

Two `Conversation` instances share no state. Conversation B never inherits A's topic:

```
A: "What is the refund policy?"   ->  A.prepare_query(...)  contains "refund"
B: "How do I download videos?"    ->  B.prepare_query(...)  contains no "refund"
```

There is no module-level or class-level mutable state.

---

## 13. Validation and edge cases

| Input | Behavior |
|---|---|
| `None` / non-`str` message | `EmptyMessageError` / `TypeError` |
| `""`, `"   "`, `"\t\n"` | `EmptyMessageError` |
| Role other than user/assistant | `InvalidRoleError` (on `Turn` and `add_turn`) |
| `max_turns < 2` or non-int | `ValueError` |
| History over bound | Oldest turns dropped; `max_turns` retained |
| Empty conversation | `build_context_block()` -> `""`; `prepare_query()` -> stripped query |
| Very long message | Truncated in context block and in prepared-query prefix |
| Reset then query | No inherited context |

---

## 14. Limitations

- **No intent inference / coreference resolution.** "What about 20 days?" keeps the
  topic only because the previous *user* message is prepended; a pronoun-only follow-up
  with no relevant prior user text will not be resolved. This is intentional (no LLM
  rewriting) and bounded.
- **No summarization or long-term memory.** Only the last `max_turns` turns exist.
- **No persistence.** History is lost when the process ends.
- **Query concatenation is crude.** It is deterministic and dependency-free, not a
  linguistic rewrite; two unrelated prior topics can both appear in the prepared query.
- **No cross-session/thread concept** beyond one `Conversation` per session.

## 15. Trade-offs

| Decision | Why |
|---|---|
| Concatenate recent user messages (no LLM) | Deterministic, free, testable, no extra failure mode |
| Exclude assistant text from `prepare_query` | Prevents a prior answer from steering retrieval as fact |
| Bound to 20 turns / 3 user messages / 1200 chars | Bounds prompt size and cost; keeps context on-topic |
| Optional `conversation_block` keyword | Zero impact on M1–M4 call sites; opt-in |
| Context block outside `<evidence>` | Makes "not evidence" structural, not just a prompt request |
| No persistence | Fits the 4–6 hour scope; M6 owns the application layer |

## 16. Future improvements

- Deterministic coreference/keyword carry-over for pronoun-only follow-ups.
- Optional LLM query rewriting **behind M3** (never as an evidence source).
- Turn-level topic tagging to bias preparation toward the most relevant prior topic.
- Configurable, per-request history limits.
- Persistent sessions once M6 introduces the application layer.
- Explicit "topic reset" heuristic when the user clearly changes subject.

---

## 17. Files

- `learnforge/conversation.py` — implementation
- `tests/test_conversation.py` — deterministic unit + integration tests
- `docs/conversation.md` — this document


