# LearnForge RAG Support Assistant

A **retrieval-augmented generation (RAG)** customer-support assistant for the
fictional ed-tech company **LearnForge**, built as the Applied AI/LLM Engineer
take-home assignment. It answers user questions from a supplied knowledge base
(15 FAQs, 10 policy documents, 15 historical support tickets), is deliberately
conservative in the face of the KB's **intentional contradictions and outdated
documents**, cites every KB-backed claim, and **escalates to a human instead of
guessing**.

The guiding principle of the whole design:

> **The LLM is only a language generator. It never decides whether the evidence
> is trustworthy — a deterministic evidence-assessment layer does that.**
> **Conversation history is contextual input, not knowledge-base evidence.**

## What this is about

The supplied LearnForge KB deliberately contains conflicting and outdated
information (e.g. a stale 7-day refund policy alongside the current 14-day
policy and a historical 30-day promotional claim; "download courses on your
laptop" advice that is no longer true; retired card-number-handling rules).
A naive RAG system would confidently echo whichever contradictory chunk it
retrieved first. This project demonstrates the engineering that prevents that:

1. **Deterministic ingestion** preserves every record verbatim with its ID,
   source type, freshness dates (never invented), stale flags, and ticket status.
2. **Hybrid retrieval** (local semantic embeddings + BM25 lexical, fused with
   Reciprocal Rank Fusion) surfaces *both* sides of a contradiction — stale
   records are never deleted, because they are the evidence of the conflict.
3. **Deterministic evidence assessment** classifies every turn into one of six
   states (`answerable`, `clarification_required`, `insufficient_evidence`,
   `conflicting_evidence`, `escalation_required`, `security_escalation`) and
   detects the KB's seven known contradiction families.
4. **Grounded generation** (Groq, `llama-3.3-70b-versatile`, free tier) may only
   use the evidence M3 approved, must cite supplied record IDs, treats KB text
   as data (prompt-injection defence), and follows M3's routing.
5. **Multi-turn context** is bounded, deterministic, and never authoritative.
6. **Everything is offline-testable**: the full test suite (297 tests) and the
   evaluation benchmark run with no network and no API key.

## Architecture diagram

```text
                            ┌──────────────────────────────────────────┐
                            │                USER                      │
                            └────────────────────┬─────────────────────┘
                                                 │ user message
                                                 ▼
                      ┌─────────────────────────────────────────────┐
                      │      M5 Conversation (in-memory)            │
                      │  validate • bounded history (20 turns)      │
                      │  prepare_query(): recent user turns + query │
                      │  build_context_block(): NON-evidence context│
                      └────────────────────┬────────────────────────┘
                                           │ prepared query
                                           ▼
 ┌──────────────────────────┐   ┌─────────────────────────────────────┐
 │  OFFLINE DATA PIPELINE   │   │   M2 Hybrid Retrieval  (KB ONLY)    │
 │                          │   │                                     │
 │  faqs/policies/tickets   │   │  semantic: all-MiniLM-L6-v2 (local) │
 │        .md (SOURCE,      │   │  lexical : Okapi BM25 (stdlib)      │
 │        never modified)   │   │  fusion  : Reciprocal Rank Fusion   │
 │            │             │   │           (RRF, k=60) + top-k=5     │
 │            ▼             │   └──────────────────┬──────────────────┘
 │  M1 Ingestion (determ.)  │                      │ 40 records, one
 │  parse•validate•normalize│                      │ chunk per record
 │            │             │                      │ + full metadata
 │            ▼             │                      ▼
 │  kb_records.json (40)    │   ┌─────────────────────────────────────┐
 │  kb_embeddings.json      │──▶│   M3 Evidence Assessment            │
 │  (40 × 384, local store) │   │   (THE AUTHORITATIVE GATE)          │
 └──────────────────────────┘   │  • state + routing + confidence     │
                                │  • 7 contradiction families         │
                                │  • stale / current / exception split│
                                │  • security (PII) detection         │
                                │  • escalation + ambiguity signals   │
                                └──────────────────┬──────────────────┘
                                                   │ approved evidence
                                                   │ approved evidence
                                                   ▼
                        ┌─────────────────────────────────────────────┐
                        │   M4 Grounded Generation                    │
                        │  SYSTEM RULES                               │
                        │  <evidence> KB text + metadata </evidence>  │
                        │  conversation context (NOT evidence)        │
                        │  <user_query>                               │
                        │  provider adapter: Groq / FakeProvider      │
                        │  citation sanitising • structured failures  │
                        └────────────────────┬────────────────────────┘
                                             │ AssistantResult
                                             ▼
                        ┌─────────────────────────────────────────────┐
                        │  M6 Orchestration (assistant.py + cli.py)   │
                        │  wires M2→M3→M4, structured result,         │
                        │  error boundaries, conversation update      │
                        └─────────────────────────────────────────────┘
```

## Data schema diagram

```text
SUPPLIED MARKDOWN (immutable source of truth)
├── faqs.md      ──► 15 records  (FAQ-01 … FAQ-15)
├── policies.md  ──► 10 records  (POLICY-01 … POLICY-10)
└── tickets.md   ──► 15 records  (TICKET-01 … TICKET-15)
            │
            ▼  M1 ingest.py  (deterministic; heading-based segmentation;
            │                 SHA-256 of each source recorded; malformed
            │                 records rejected, never silently accepted)
            ▼
kb_records.json ─── 40 normalized records ───┐
            │                                │  one record = one chunk
            ▼  M2 embed.py (local model)     │  (auditable citations)
kb_embeddings.json ─── 40 × 384 vectors ──────┘
```

Every normalized record (JSON):

| Field | Type | Meaning |
|---|---|---|
| `source_id` | str | Stable record ID, e.g. `POLICY-02` |
| `source_type` | str | `faq` \| `policy` \| `ticket` |
| `title` | str | Record heading/question/subject |
| `chunk_text` | str | **Verbatim** source text (incl. heading) — never rewritten |
| `citation_key` | str | Equals `source_id`; the only citable key in answers |
| `metadata.authority` | int | `policy=3`, `faq=2`, `ticket=1` (source-type property) |
| `metadata.freshness` | str | Parsed `Last reviewed/Effective/Updated` date, else `"undated"` — **dates are never invented** |
| `metadata.is_stale` | bool | True only for explicitly archived/"older version" language |
| `metadata.stale_reason` | str\|null | **Verbatim** source sentence justifying the stale flag |
| `metadata.ticket_status` | str\|null | Verbatim ticket resolution status |
| `metadata.escalated` / `unresolved` | bool | Ticket escalation / unresolved flags |
| `metadata.ambiguity_flags` | list[str] | e.g. `ambiguous_intent_requires_clarification` |
| `metadata.contradiction_topics` | list[str] | Retrieval hints: `refund_window`, `annual_billing`, `offline_downloads`, `browser_support`, `captions_accessibility`, `progress_sync`, `payment_data_collection` |
| `vector` | null | Filled at embedding time in `kb_embeddings.json` |

Example record (abbreviated):

```json
{
  "source_id": "POLICY-02",
  "source_type": "policy",
  "title": "Refund Policy",
  "chunk_text": "### POLICY-02 — Refund Policy\nLearnForge offers a 7-day refund window ... [verbatim] ...",
  "citation_key": "POLICY-02",
  "metadata": {
    "authority": 3,
    "freshness": "Last reviewed: February 2026",
    "is_stale": true,
    "stale_reason": "This older wording is archived; the current refund window is 14 days.",
    "contradiction_topics": ["refund_window"]
  },
  "vector": null
}
```

## How routing, safety and escalation work

M3 maps every turn to exactly one of six states and M6/M4 must follow it:

| State | Routing | Example |
|---|---|---|
| `answerable` | generate with citations | "How do I reset my password?" |
| `conflicting_evidence` | generate **with disclosure**, or escalate | "Can I get a refund after 20 days?" (14-day policy vs 30-day historical claim → escalate) |
| `insufficient_evidence` | decline, no citations | "Do you support feature X?" |
| `clarification_required` | ask for clarification | "Cancel my LearnForge" (subscription? course? account?) |
| `security_escalation` | escalate, never request PII | "Can I send you my CVV?" |
| `escalation_required` | escalate to human review | outdated-doc reliance with unresolved tickets |

Confidence is **categorical** (`high`/`medium`/`low`) with all component signals
exposed — never a pseudo-calibrated probability. Payment credentials (CVV,
card numbers, passwords, auth codes) are never solicited. Failed turns are
never recorded into history and never produce fabricated answers.

## Setup & running

```bash
# 1. install (Python 3.11+ recommended)
pip install -r requirements.txt        # numpy, sentence-transformers, pytest, groq

# 2. run the full test suite (offline, no API key, ~20 s)
python -m pytest tests -q              # 297 tests

# 3. run the deterministic evaluation benchmark (28 cases)
python -m learnforge.evaluation
python -m learnforge.evaluation --json-out results.json

# 4. try the assistant — offline deterministic mode (no API key)
python -m learnforge.cli --provider fake --once "What is the refund policy?"

# 5. real LLM mode (interactive multi-turn session; needs GROQ_API_KEY)
#    set it via the environment — NEVER hard-code or commit it
setx GROQ_API_KEY "your-key-here"      # Windows (or $env:GROQ_API_KEY=... / export ...)
python -m learnforge.cli --provider real          # session: :reset :details :quit
python -m learnforge.cli --provider real --once "What is the refund policy?" --show-details
```

Notes:

* The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) to the
  local HuggingFace cache. **No KB content ever leaves your machine for
  retrieval** — embeddings are computed locally.
* `--provider real` without `GROQ_API_KEY` exits with explicit guidance; the CLI
  never silently substitutes the fake provider. `.env` files are gitignored.

## Tests — what is tested and what it tells you

The suite has **297 offline tests** (no network, no API key; M4's `FakeProvider`
is used wherever an LLM would be called). Run them all with
`python -m pytest tests -q`, or one file at a time. Each suite verifies one
milestone and gives a reviewer a specific guarantee:

| Suite | # | What it verifies | The guarantee it gives you |
|---|---|---|---|
| `tests/test_ingest.py` | 43 | Parses all 3 markdown files into exactly 15+10+15 records; IDs unique/preserved; verbatim content; required fields; **no dates invented for undated records**; stale set + verbatim stale reasons for all 10 stale records; per-contradiction metadata (refund windows, annual billing, offline downloads, browsers, captions, sync, card digits); ambiguous/unresolved tickets keep status; duplicate/missing/malformed records rejected; byte-identical JSON across runs; **source KB files unmodified (SHA-256)** | The data foundation is faithful to the supplied KB — nothing rewritten, invented, or dropped |
| `tests/test_retrieval.py` | 47 | Retriever indexes all 40 records; embeddings generated **locally/offline**; semantic, lexical (BM25) and hybrid (RRF) retrieval each return relevant records for realistic queries; rare terms and exact record IDs (`FAQ-08`) found lexically; deterministic tie-breaking; **stale records not deleted**; tickets remain identifiable with status/ambiguity flags; authority metadata intact; empty/whitespace/nonsense queries safe; source KB untouched | Retrieval surfaces the *right evidence including contradictory/stale sides*, without authority-based filtering, deterministically |
| `tests/test_evidence.py` | 49 | All six M3 states; current-vs-stale distinction; undated ≠ falsely stale; authority preserved (never a filter); **all 7 contradiction families detected** (with current/stale/exception evidence IDs split); ambiguity → clarification; nonsense → insufficient; CVV/password/auth-code → security escalation; day-driven escalation (20 days escalates, 10 doesn't); deterministic assessments; **no LLM/network imports at runtime** | The *reliability gate* works: the system knows what it knows, what conflicts, and when to ask a human — before any LLM is involved |
| `tests/test_generation.py` | 72 | Prompt construction (system rules → delimited `<evidence>` → context → query); grounding rules (no outside knowledge, no fabricated policies/dates/eligibility); **citations restricted to M3-approved IDs** (invented `[TICKET-03]` is stripped and reported); tickets not presented as policy; stale evidence not presented as current; **all six state behaviors incl. the exact security constraints**; prompt-injection instruction present; provider errors (missing key, timeout, rate limit, auth, server, malformed, empty) → structured failures, no invented answers | The LLM *cannot* overstep: it is a generator inside a fence, and every failure mode degrades safely |
| `tests/test_conversation.py` | 32 | Turn model (user/assistant only); invalid role/empty message rejected; bounded history (20 turns, newest kept); follow-up prepared query preserves topic ("refund…20 days"); history never becomes a citation; **session isolation**; reset removes context; M3 routing preserved across turns | Multi-turn context is bounded, deterministic, and can never corrupt evidence or routing |
| `tests/test_assistant.py` | 37 | Orchestration order (retrieve → assess → generate); prepared query reaches M2/M3; all six states pass through **verbatim**; retrieval/assessment/generation failures → structured `error`, downstream never runs, history untouched; provider failure → M4's fact-free message, turn not recorded; empty/non-string input via M5 rules; session isolation; reset | The end-to-end pipeline wires the right components in the right order and fails safely at every boundary |
| `tests/test_evaluation.py` | 17 | The benchmark harness itself: case schema validation, malformed cases rejected; **the harness can detect a regression** (wrong expectation → fails); determinism across repeats; all 28 cases pass with full dimension rollups | The evaluation is trustworthy — it proves behavior, and it would notice breakage |

Total: **297 tests, all passing in ~20 s, zero network access.**

## Evaluation benchmark & results

`python -m learnforge.evaluation` runs **28 behavioral cases** (24 pipeline +
4 failure) through the **real** retriever and assessor with the offline
`FakeProvider`. Expectations are behavioral (state/routing/flags/required
evidence), never reference wording; exact record IDs are asserted only where
genuinely required (current-vs-stale and contradiction cases).

Actual results:

```
category          passed  /total          dimension        passed  /total
answerable             3/3               routing               66/66
conflict               4/4               retrieval             17/17
stale                  4/4               grounding             51/51
insufficient           2/2               conversation           8/8
security               3/3               failure_safety        16/16
ambiguous              2/2
ticket_evidence        2/2               cases: 28  passed: 28
multi_turn             4/4               deterministic: True
failure                4/4               ALL PASSED: True
```

These are **grounding / unsupported-answer behavioral checks**, not a calibrated
hallucination rate: free-form answer wording and faithfulness require human
review or a separately validated LLM-judge evaluation with real provider
responses (citation *presence* is only observable with a real provider, because
`FakeProvider` deliberately emits no citation markers). Methodology:
`docs/evaluation.md`.

## Failure handling summary

| Situation | Behavior |
|---|---|
| Low confidence / insufficient evidence | decline transparently; no citations; no invented answer |
| Stale data | retrieved, disclosed, never presented as current |
| Bad retrieval (component error) | structured `error`; no answer; history untouched |
| Provider failure (timeout/rate limit/auth/malformed) | structured `failure` with fact-free message; turn not recorded |
| No API key | explicit exit with guidance; no silent fake swap |
| Ambiguity | clarification request; nothing assumed |

## Trade-offs (why X over Y)

- **Hybrid retrieval + RRF, no tuned weights** — robust to exact terms and
  paraphrase without inventing score weights. With more time: cross-encoder
  re-ranking and paragraph-level chunking.
- **One record = one chunk** — fully auditable citations at 40 records; real
  scale would need finer chunks + metadata filters.
- **Local MiniLM embeddings** — free, offline, reproducible; a provider-grade
  embedding would score higher (the store records the model name for a swap).
- **Deterministic query preparation** — no extra LLM call, fully testable; an
  LLM rewriter would resolve pronouns better.
- **Rule-based contradiction families** — explainable and aligned to this tiny
  dataset instead of a brittle general NLI model.
- **No RAGAS / external eval framework** — the decisive behaviors are
  deterministic structured outputs; an LLM judge adds nondeterminism and cost
  for marginal value at this scope.
- **Groq free tier behind a one-class adapter** — satisfies the assignment
  constraint; the provider is swappable by design.

With more time/budget: persistent sessions + an API server, KB versioning with
ingestion timestamps, LLM-judge evaluation over a golden answer set, feedback
into retrieval thresholds, and human-agent handoff integration.

## Limitations

In-memory conversations only (no persistence by design); small hand-curated
benchmark; M3 thresholds are documented heuristics; no deployment story — this
is a design-quality prototype scoped to the assignment's ~4–6 effective hours.

## Repo layout & documentation

```
learnforge/            schema, ingest, embed, lexical, retrieval, evidence,
                       generation, conversation, assistant, cli, evaluation
tests/                 297 offline tests (7 suites, see table above)
data/processed/        kb_records.json (40 records), kb_embeddings.json (40×384)
data/eval/             cases.json (28 benchmark cases)
docs/architecture.md           root design document (M0)
docs/data-schema.md            record schema + ingestion rules (M1)
docs/retrieval.md              hybrid retrieval design (M2)
docs/evidence-assessment.md    states, conflicts, confidence (M3)
docs/generation.md             prompts, grounding, provider adapter (M4)
docs/conversation.md           bounded multi-turn context (M5)
docs/end-to-end.md             orchestration + CLI (M6)
docs/evaluation.md             benchmark methodology + results (M7)
```

Each milestone document lists its design decisions, parameters, and explicit
assumptions; `docs/architecture.md` is the root.




