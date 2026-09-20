# End-to-End Assistant (M6)

M6 connects the approved M0–M5 components into **one coherent, runnable pipeline**.
It is a thin coordination layer: it owns no retrieval, evidence, safety, citation,
or generation rules.

> **Conversation history is contextual input, not knowledge-base evidence.**
>
> **M6 orchestrates M2–M5; it does not replace their responsibilities.**

## 1. Component responsibilities

| Layer | Module | Responsibility (unchanged) |
|---|---|---|
| M5 | `conversation.py` | Bounded in-memory history, validation, query preparation, context block |
| M2 | `retrieval.py` | Hybrid (dense + BM25, RRF) retrieval over the 40 normalized KB records |
| M3 | `evidence.py` | Authoritative evidence gate: state, routing, confidence, conflicts, security |
| M4 | `generation.py` | Grounded generation, citations, prompt-injection defence, provider errors |
| M6 | `assistant.py` | Orchestration of one turn; structured result; error boundaries |
| CLI | `cli.py` | Interactive/single-shot demo runner exercising the real pipeline |

## 2. Complete request flow

```text
User
 ↓
Conversation (M5)            validate_message — raises on empty/invalid input
 ↓
Query preparation (M5)       prepare_query(): recent user turns + current query
 ↓
Hybrid Retrieval (M2)        KB ONLY — fresh retrieval every turn
 ↓
Evidence Assessment (M3)     authoritative: state / routing / confidence
 ↓
Grounded Generation (M4)     prompt + evidence + citations; provider hidden
 ↓
Assistant Response
 ↓
Conversation update          ONLY after a successful generation
```

## 3. Orchestrator interface

```python
from learnforge.assistant import Assistant, create_assistant

bot = create_assistant(
    retriever=...,        # M2 Retriever (built lazily from the local store if omitted)
    provider=...,         # M4 provider (e.g. FakeProvider or the configured Groq one)
    conversation=...,     # M5 Conversation (a fresh one is created if omitted)
    assessor=assess_evidence,   # M3 (injectable for tests)
    generator=generate,         # M4 (injectable for tests)
    top_k=5,
)
result = bot.handle_message("What is the refund policy?")
```

`AssistantResult` (structured, `to_dict()` for logs/evaluation) exposes:
`query`, `prepared_query`, `answer`, `state`, `routing`, `confidence`,
`retrieved_ids`, `evidence_ids`, `citation_keys`, `citations_used`,
`allowed_citations`, `invalid_citations`, `conflict_topics`,
`security_triggered`, `escalation_required`, `mode`, `failure_type`,
`failure_detail`, `provider`, `model`, `conversation_updated`, `turn_count`,
`explanation`. No secrets and no raw prompts/responses are exposed.

Turn modes: `llm` (M4 grounded answer), `failure` (M4 structured provider
failure carrying its fact-free message), `error` (an M6 component boundary
raised; empty answer, nothing stored).

## 4. Conversation integration

M5 owns validation (`EmptyMessageError`/`TypeError` propagate — M6 does not
duplicate the rules), query preparation (`prepare_query`), and the non-evidence
context block (`build_context_block`) forwarded to M4. The user turn and the
assistant turn are appended **only** after a successful generation, so a failed
turn never records a fabricated answer.

## 5–7. Retrieval / evidence / generation integration

M6 calls `retriever.search(prepared_query, top_k)` — the prepared query from M5,
never raw history concatenation. It passes the **current** retrieval results into
`assess_evidence` unchanged; M3 alone decides state/routing/confidence. It calls
M4's `generate(prepared_query, assessment, provider=…, conversation_block=…)`;
no second prompt, no direct Groq call, no provider selection in M6.

## 8. Error boundaries

| Failure | Behaviour |
|---|---|
| Retrieval raises | `mode="error"`, `failure_type="retrieval_error"`; M3/M4 never run; no answer; history untouched |
| Assessment raises | `mode="error"`, `failure_type="assessment_error"`; generation never called |
| Generation raises | `mode="error"`, `failure_type="generation_error"` |
| M4 provider failure | `mode="failure"` with M4's fact-free message and failure category; **not** stored as an assistant turn |
| Invalid input | M5's exception propagates; no state change |
| Insufficient evidence | M3's normal `insufficient_evidence` path runs — no fallback knowledge is invented |

No retries, no fallback provider, no web search (predictable failure behaviour,
per the assignment).

## 9. CLI usage

```bash
# offline, deterministic (no API key needed):
python -m learnforge.cli --provider fake --once "What is the refund policy?"

# real provider (requires GROQ_API_KEY; never printed or logged):
python -m learnforge.cli --provider real --show-details

# session mode: :reset :details :quit
```

The provider choice is **explicit**. `--provider real` without a key exits with a
clear message — the CLI never silently substitutes fake generation. Flags:
`--k`, `--records`, `--embeddings`, `--json`, `--fake-response`, `--fake-error`.

## 10. Testing strategy

`tests/test_assistant.py` — 37 tests: unit tests on injected fakes (call order,
state passthrough for all six M3 states, error boundaries, validation, session
isolation, reset) plus integration tests over the **real** retriever + **real**
M3 with M4's `FakeProvider` (no network, no key). M1–M5 suites are unchanged
regressions.

## 11. Limitations / trade-offs

* In-memory sessions only (no persistence by design; M0 scope).
* `prepare_query` is deterministic text preparation, not LLM rewriting — cheap
  and predictable, but it cannot paraphrase ("cancel"→"refund") beyond prefixing
  recent user turns.
* A boundary error collapses to a single structured result; no partial streaming.
* The CLI is a demo surface, not a product UI.

## 12. Future improvements

Session persistence, API server, evaluation harness (M7), richer context
preparation (topic extraction), and streaming responses.
