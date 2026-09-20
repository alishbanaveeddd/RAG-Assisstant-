# Evidence Assessment Layer (M3)

> Milestone: M3 — deterministic evidence assessment between M2 retrieval and M4 generation.
> Flow: `query → M2 retrieval → retrieved evidence → M3 assessment → {sufficient, insufficient, conflicting, ambiguous, escalation_required} → M4 (later)`.
> M3 does **not** generate answers, does **not** call an LLM, and does **not** implement chat history, API, or UI.

## 1. Public interface

```python
from learnforge.evidence import assess_evidence
assessment = assess_evidence(query, results)   # results: list[RetrievalResult] from M2
```

`assess_evidence` is deterministic: same query + same results ⇒ identical
`EvidenceAssessment.to_dict()` output (asserted by test). It never mutates or deletes
the retrieved `results` — all evidence, including stale records, stays available to M4.

## 2. Assessment states and routing

| State | Constant | Routing (M0 §8.5) | Meaning |
|---|---|---|---|
| `answerable` | `ANSWERABLE` | `generate` | Current, non-contradictory, query-relevant evidence covers the question. |
| `clarification_required` | `CLARIFICATION_REQUIRED` | `clarify` | Underspecified action request **and** retrieved evidence carries the TICKET-07 ambiguity classification. The clarification *question* is M4's job. |
| `conflicting_evidence` | `CONFLICTING_EVIDENCE` | `generate` (disclosure required), or `escalate` if an escalation signal also fired | Query-relevant evidence contains both a current claim and a stale/historical/exception claim (M0 §11). |
| `insufficient_evidence` | `INSUFFICIENT_EVIDENCE` | `decline` | No results, low relevance band, no query-aligned coverage, or only stale evidence for the queried topics. |
| `escalation_required` | `ESCALATION_REQUIRED` | `escalate` | Query-aligned escalation flags from the corpus (see §7). |
| `security_escalation` | `SECURITY_ESCALATION` | `escalate` | Query names sensitive payment/identity data (M0 §12.4, FR-17). |

State precedence (first match wins, `_decide_state`): `security_escalation` →
`clarification_required` → `insufficient_evidence` (no results / low band / uncovered
topics / stale-only) → `conflicting_evidence` → `escalation_required` → `answerable`.

## 3. Evidence dimensions (kept separate and explainable)

1. **Relevance** — categorical band from the best semantic score: `high ≥ 0.45`,
   `medium ≥ 0.30`, else `low`. Raw similarity is a *signal*, never proof of sufficiency.
2. **Authority** — lifted verbatim from M1 metadata (`policy=3, faq=2, ticket=1`) plus a
   **role** per record (`authoritative_policy` / `operational_guidance` /
   `historical_evidence`). Authority never filters or reorders evidence (M0 §9).
3. **Freshness** — three buckets, no invented dates: `current` (dated and not stale),
   `stale` (M1 `is_stale`), `undated` (FAQs/tickets without a date line).
4. **Contradiction/conflict** — deterministic family detection (§4).
5. **Ambiguity** — query-shape heuristic *and* evidence alignment (§6).
6. **Coverage** — which family topics the query names, which are covered, whether any
   current authoritative record supports them, whether support is stale-only.
7. **Security/PII** — regex-based sensitive-data detection (§8).
8. **Escalation indicators** — query-aligned lifts of M1 ticket flags (§7).

## 4. Contradiction detection (deterministic, corpus-specific)

Seven families mirror M0 §6.4. Each defines regexes for `stale_claims`, `current_claims`,
and (refunds only) `exception_claims`, matched against the **verbatim** `chunk_text`.
A record that quotes an outdated claim *and* corrects it (FAQ-07, POLICY-02) legitimately
contributes to both sides — exactly the contradiction M0 §11.1 requires exposing.

| Family | Stale side | Current side | Exception side |
|---|---|---|---|
| `refund_window` | 7-day wording | 14-day wording | 30-day / money-back guarantee |
| `annual_billing` | "billed monthly" | "annual price / shown at checkout" | — |
| `offline_downloads` | desktop/laptop download | mobile app only / "does not currently provide" | — |
| `browser_support` | Internet Explorer | Chrome/Edge/Firefox/Safari | — |
| `captions_accessibility` | captions for every video | "whenever practical" | — |
| `progress_sync` | "instantly" | "automatically synchronized" | — |
| `payment_data_collection` | "first six and last four" card digits | minimum-information rules | — |

A family fires only when **(a)** the query is about that topic
(`detect_query_topics`) and **(b)** query-relevant evidence contains both a current and a
stale/exception side. Each conflict is a `ConflictIndicator` with separate
`current_evidence_ids` / `stale_evidence_ids` / `exception_evidence_ids`, M0 §11.2
classes, and a readable description. **M3 never decides which side is true** (M0 §11.4).

## 5. Current vs stale

`EvidenceItem.freshness_class` distinguishes `current` / `stale` / `undated`, and
`EvidenceItem.claims` maps each family to the claim sides found in that record. For
"Can I download courses on my laptop?" the assessment reports FAQ-07 `stale=True` with
*both* sides, TICKET-15 historical with the outdated-documentation flag, plus an
`offline_downloads` conflict — instead of a bare high confidence.

## 6. Ambiguity

`detect_query_ambiguity` mirrors the TICKET-07 pattern: a short action request
(`cancel`, `close`, `transfer`, …) with no specific object, no recognized topic, and not
a question. Detection alone is not enough — `detected` also requires the retrieved
evidence to carry M1's `ambiguous_intent_requires_clarification` flag, so a
heuristic-ambiguous query over unambiguous evidence does not trigger clarification.
State: `clarification_required`, routing `clarify`. **No clarification question is
generated** (M4's job).

## 7. Escalation (query-aligned only)

| Flag (from M1 metadata) | Fires when | Corpus basis |
|---|---|---|
| `promotional_terms_may_differ` | refund query citing N days with `14 < N ≤ 30` | TICKET-03 (30-day guarantee vs 14-day standard) |
| `policy_wording_ambiguity` | subscription + refund query | TICKET-08 (contradictory cancellation-page wording) |
| `outdated_documentation_reliance` | offline-download query | TICKET-15 (material reliance on outdated guide) |
| `identity_verification_required` | account/transfer query naming an account/email | TICKET-10 / TICKET-13 |
| `escalated_ticket_evidence` | query-relevant evidence includes a ticket the corpus itself escalated | TICKET-03 / 08 / 11 |

A flag on a retrieved ticket describes *that ticket's* handling; it must not escalate an
unrelated query. `escalation.required=True` always carries at least one documented
reason (each reason names its `flag`, `reason`, and `evidence_ids`).

## 8. Security / PII

`detect_security` marks a query security-sensitive when it names sensitive data outright
(`cvv`, `cvc`, card number, banking password, authentication/auth code, PIN, government
ID, SSN, passport) or offers to share a password/credential. Asking *what* support may
request (without naming sensitive data) is **not** security-sensitive — that is
answerable from policy. State: `security_escalation`, routing `escalate`. M3 generates
no security advice and requests no sensitive information; it also lists the evidence IDs
carrying the corpus's PII rules (`evidence_ids_with_pii_rules`) so M4 can ground a safe
response.

## 9. Confidence approach

Categorical (`high` / `medium` / `low`) **plus** every component that produced it — no
unexplained magic number, never presented as a calibrated probability:

- Non-`answerable` states → `low`.
- `answerable` with a `high` relevance band and current authoritative evidence → `high`.
- Otherwise → `medium`.

Exposed components: `best_semantic_score`, `best_lexical_score`,
`relevant_record_count`, `record_count`, `has_current_authoritative_evidence`. Each
`ConfidenceAssessment` carries the note: *"Internal heuristic for routing/debugging
only; NOT a calibrated probability of correctness."* A cross-check to M0 §12.1's five
levels is included as `m0_level`.

## 10. Output object

`EvidenceAssessment` (`.to_dict()` for logging): `query`, `state`, `routing`,
`confidence`, `evidence` (per-record dimensions incl. role, authority, freshness class,
claims, flags, retrieval scores), `evidence_ids`, `citation_keys`, `conflicts`,
`ambiguity`, `security`, `escalation`, `coverage`, and `explanation` — a multi-line
**developer-facing** summary for logs/debugging (explicitly *not* a customer answer).

## 11. Why the LLM does not decide sufficiency

M0 §14: the generator only sees evidence chosen *after* a deterministic assessment gate.
Letting the LLM self-assess trustworthiness would reintroduce hallucination pressure —
the model cannot be relied on to refuse its own confident answer. M3's rules are
inspectable, testable, and reproducible; M4 receives an explicit machine-readable state
and may only generate when the state permits it.

## 12. Why raw similarity is not treated as probability

Embedding cosine similarity measures surface/topic overlap, not answerability. High
similarity to a *stale* record is precisely how naive RAG systems confidently repeat
outdated policy. M3 therefore combines similarity with authority, freshness, coverage,
conflict, and security signals, and refuses to bless evidence below the medium band even
when records exist.

## 13. Limitations

- Topic/conflict detection is regex-based over the *known* families; new families need
  new rules (deliberate M0 §11.1 trade-off for a 4–6 h scope).
- Relevance bands use fixed thresholds (0.45 / 0.30) on all-MiniLM-L6-v2 scores —
  heuristics, not calibrated probabilities.
- Ambiguity detection is lexical; paraphrases without action verbs are not detected.
- `escalated_ticket_evidence` requires topic overlap; escalation-worthy tickets without
  overlap are surfaced as evidence but do not force escalation (conservative choice).
- Security detection is English-only regex matching — defense-in-depth, not a
  replacement for M4's output-side PII checks.

## 14. Testing

`tests/test_evidence.py` — 49 pytest tests: state vocabulary; current/stale/undated
recognition; authority preservation and separation from relevance; all seven
contradiction families (incl. synthetic both-sides construction); current-vs-stale
distinguishability; ambiguity + evidence alignment; insufficiency (nonsense / empty /
weak / stale-only); security states (CVV, card number, banking password, auth code, plus
the negative "what may support ask" case); ID/citation preservation; non-mutation of
results; determinism; no-LLM (static import scan + runtime socket-blocking); confidence
explainability; and dataset-specific regressions (annual-billing, offline,
captions/sync, payment-data, day-driven promotional escalation).
Full suite: **139 passed** (43 M1 + 47 M2 + 49 M3).


