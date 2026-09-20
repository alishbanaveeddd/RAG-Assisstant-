# LearnForge Support Assistant — Architecture & Requirements (Milestone M0)

> **Milestone:** M0 — Architecture & Requirements
> **Status:** documentation only. **Nothing has been implemented in this milestone**: no code (beyond this analysis), no dependency installation, no embeddings created, no vector database created, and no LLM APIs called.
> **Assignment source:** `Applied AI-LLM Engineer - Take-Home Assignment.docx` — an AI-powered customer-support assistant that answers questions from an internal KB of FAQs, help-center/policy docs, and past support tickets; ~4–6 hours; free-tier LLM APIs allowed.
> **Corpus inspected:** `learnforge-knowledge-base-data/learnforge-knowledge-base/` — `README.md`, `faqs.md` (FAQ-01…15), `policies.md` (POLICY-01…10), `tickets.md` (TICKET-01…15).

---

## 1. Problem statement

LearnForge must be able to answer customer-support questions by retrieving from and
generating over an internal knowledge base composed of three source types: **FAQs**
(curated Q&A), **policies / help-center excerpts**, and **past support-ticket
transcripts**. The assignment requires a retrieval-augmented-generation (RAG) loop that
**reduces hallucination**, **handles multi-turn conversations**, and **escalates to a
human when it is not confident** — and the supplied corpus is *intentionally realistic
and intentionally messy* (per the corpus `README.md`): it contains contradictions,
outdated policy language, retired internal instructions, stale help articles, and
ambiguous tickets.

A naive “retrieve some chunks, prompt an LLM, return the answer” design would
synthesize wrong answers from stale or conflicting chunks (e.g. assert a single refund
window when three different numbers appear in the corpus), would treat past ticket
outcomes as current policy, would answer questions the data does not actually support,
and would ignore the security instructions to never request or repeat payment-card /
identity data. The architecture must be specifically shaped around the LearnForge
dataset’s known problems, not generic RAG boilerplate.

---

## 2. Goals

1. Answer LearnForge support questions with a **retrieval + generation** pipeline over the supplied corpus.
2. Treat **retrieved evidence as separate from the LLM**: answers are grounded *only*
   in retrieved evidence (quoted, cited), not in the model’s parametric knowledge.
3. Preserve and surface **source metadata** — document ID, source type, and
   dates/version/freshness markers — for every cited claim.
4. **Recognize insufficient evidence** and decline/clarify/escalate rather than fabricate.
5. **Recognize conflicting / stale evidence** (e.g. the 7-day vs 14-day vs 30-day
   refund wording; the “annual billed monthly” note; the IE browser note) and handle it
   without silently merging or overwriting.
6. **Escalate** uncertain, ambiguous, or high-risk cases (especially financial/refund,
   security/fraud, account-transfer, and outdated-document-that-influenced-a-purchase
   cases) rather than over-committing.
7. Keep **conversation context separate from authoritative KB content** — chat history
   informs disambiguation, not facts.
8. Fit the **~4–6 hour** scope: small, demonstrable, locally runnable.

---

## 3. Non-goals

- Not a production support-ticketing system. No live account/order lookup, no real
  payment handling, no billing-system integration.
- Not a KB cleanup tool. The supplied data files are **not modified, rewritten,
  deleted, or moved**; chunking/restructuring is a derived, reversible representation.
- Not a generic framework demo. The design must specifically handle the LearnForge
  contradictions, outdated docs, and ambiguous tickets.
- Not open-ended chit-chat. Multi-turn is for support clarification only.

## 4. Functional requirements

### 4.1 Source model
- **FR-1.** The system models exactly three source types: `faq` (FAQ-01…15), `policy` (POLICY-01…10), and `ticket` (TICKET-01…15). Each is a separately addressable record with a stable `source_id` used for every citation.
- **FR-2.** Each record carries metadata: `source_id`, `source_type`, `title`, and — where present — `freshness` (the review/effective/updated date string or note), plus an extracted **`is_stale`** flag when the record text explicitly says an older version is wrong/outdated/retired/obsolete (see §6.4).
- **FR-3.** Ticket transcripts are **evidence of handling**, not authoritative policy. They carry a `status` (Resolved / Pending monitoring / Escalated / Awaiting … / Refund review requested / …) and are used to recognize ambiguity, escalation precedent, and what information agents ask for — never as the source of “what the current policy is.”

### 4.2 Ingestion / representation (no data modification)
- **FR-4.** The system parses the supplied markdown into per-record chunks preserving original `source_id`/`source_type`/metadata. It does **not** modify source files.
- **FR-5.** Where a policy record contains an explicit “older version said X — that is outdated” note, that note is preserved so the response layer can prefer the current version and disclose the conflict rather than average them.

### 4.3 Retrieval (evidence kept separate from generation)
- **FR-6.** For a user query the system retrieves a small, ranked set of **evidence objects** (records + metadata + score), never the whole corpus as one block.
- **FR-7.** Retrieval must be able to surface both the **current** and the **stale** version of a topic when both exist (e.g. current 14-day refund policy vs. the archived 7-day article; mobile-only offline vs. the stale desktop-download article), so the evidence-evaluation layer — not retrieval — decides which version is authoritative.
- **FR-8.** Retrieval favors `faq` and `policy` records for factual questions and also considers `ticket` records as precedent/escalation exemplars (clearly tagged as non-authoritative).

### 4.4 Response generation
- **FR-9.** The generator receives an **evidence packet** (quoted excerpts with `source_id`, `source_type`, `freshness`, `is_stale`) and must ground every factual claim in it. It must not use parametric knowledge of LearnForge policy.
- **FR-10.** The system must classify each turn at response time as: `supported` / `conflicting/stale` / `low-evidence` / `no-evidence`, and behave accordingly.
- **FR-11.** On conflicting/stale/low-evidence, prefer caution: disclose the conflict or uncertainty, ask a clarifying question, or escalate — do not synthesize a confident blended answer.
- **FR-12.** The system must ask clarifying questions for ambiguous requests (mirroring TICKET-07) instead of guessing intent.

### 4.5 Conversation
- **FR-13.** Support multi-turn for a single request needing clarification.
- **FR-14.** Conversation history is a **separate channel** from the KB; it supplies user-provided facts and intent, never authoritative policy.
- **FR-15.** The system can incorporate user-provided details from prior turns (e.g. order number, course name, “I bought it 8 months ago”) without treating them as KB truth.

### 4.6 Escalation and safety
- **FR-16.** The system must escalate rather than over-answer for these corpus-derived classes (see §6.5):
  - promotional/contractual refund exceptions differing from the 14-day standard (TICKET-03);
  - refund-window / cancellation-page wording ambiguity with no verifying artifact (TICKET-08);
  - individual-purchase vs subscription-enrollment identity conflict with no resolved receipt (TICKET-11);
  - account/identity/enrollment/transfer questions requiring verification or admin action (TICKET-10, TICKET-13);
  - outdated official documentation that appears to have influenced a purchase decision (TICKET-15);
  - billing/chargeback/duplicate-payment/pending-authorization needing transaction references (TICKET-02, TICKET-05, TICKET-06, TICKET-14);
  - accessibility barriers needing content-team reporting + an alternative resource (TICKET-12).
- **FR-17.** The system must **never ask for, and must not repeat**, sensitive payment/identity data. The corpus repeatedly says not to request full card number, CVV, PIN, banking password, authentication code, or government ID; only minimum info (transaction date/amount/last-four, order number, course name) is acceptable for transaction identification.

### 4.7 Citations and metadata in output
- **FR-18.** Any factual claim in the answer must map to a cited `source_id` + `source_type`; freshness shown where available; ticket records labeled precedent (not policy).
- **FR-19.** Where a source says an older version is wrong, the system prefers current wording, notes the outdated article, and — when reliance was material — points to the review/escalation path (TICKET-15 pattern).

---

## 5. Non-functional requirements

- **NFR-1 (Groundedness over fluency).** A cautious, attributed answer is acceptable; a fluent but fabricated answer is a failure.
- **NFR-2 (Inspectable evidence).** The retrieved source set for a query must be inspectable so an evaluator can confirm the answer did not “just know” it.
- **NFR-3 (Small scope).** Achievable in ~4–6 hours. Minimal, well-delimited pipeline.
- **NFR-4 (Locally runnable).** Prefer local/no external calls by default; don’t assume managed services unless explicitly chosen later.
- **NFR-5 (Visible uncertainty).** Uncertainty, conflict, staleness must be visible in the response.
- **NFR-6 (Data integrity).** Source files unmodified.

## 6. Dataset / source analysis

### 6.1 Corpus composition

| File | Records | Nature | Notes |
|---|---|---|---|
| `faqs.md` | FAQ-01 … FAQ-15 | Curated Q&A | Mostly platform guidance; some note older help articles may no longer apply (e.g. FAQ-07 offline; FAQ-13 subscription vs owned). |
| `policies.md` | POLICY-01 … POLICY-10 | Policy / help-center excerpts | Many carry explicit `Last reviewed / Effective / Updated / Reviewed` dates and explicit “older version … is outdated/retired/obsolete” callouts. Highest-density current-vs-stale source. |
| `tickets.md` | TICKET-01 … TICKET-15 | Past support transcripts w/ STATUS | Mix of Resolved **and non-resolved** outcomes (Pending monitoring, Escalated, Awaiting transaction reference, Awaiting verification, Refund review requested, Transfer request pending, Pending bank authorization, Cancellation completed; refund review requested). Historical handling, not policy authority. |

Total: **40 records**, a tiny corpus (a few pages). Natural unit of chunking = one record
(FAQ-XX, POLICY-XX, TICKET-XX) because IDs and metadata already align to these units.

### 6.2 Source authority (data-derived)

- **Policies are the strongest current-policy source**, especially those with explicit review/effective dates and with explicit “older version is wrong” notes.
- **FAQs are authoritative for the behaviors they describe**, but FAQ answers are not always dated; where a policy excerpt is more specific or more recent, policy should win for policy facts (FAQ-02 and POLICY-02 both state the 14-day standard, which is consistent).
- **Tickets are evidence, not authority.** Use for recognizing ambiguity, escalation patterns, disambiguation phrasing, and what info agents request — not as the answer to “what is the policy.”

### 6.3 Dates / version / freshness markers actually present

Captured from the files (the system should extract these, not invent others):

- POLICY-01 — “Last reviewed: February 2026.” **Explicit outdated note**: older version said annual subscriptions were billed monthly — “That wording is outdated.”
- POLICY-02 — “Effective date: January 2026.” **Explicit outdated note**: older help-center documentation referenced a 7-day refund period; “remains accessible in some archived search results but should not be treated as the current standard policy.”
- POLICY-03 — “Updated: March 2026.” **Explicit outdated note**: previous instructor handbook required captions for every video before publication; current guidance says “whenever practical,” though accessibility team continues to encourage captions for all.
- POLICY-04 — “Last updated: May 2026.” **Explicit outdated note**: previous mobile article recommended downloading over cellular data; recommendation removed.
- POLICY-05 — **Explicit outdated note**: older instructor guide required ≥5 quizzes; no longer universal.
- POLICY-06 — “Reviewed: April 2026.” **Explicit outdated note**: archived doc claimed progress saved “instantly”; wording replaced with “automatically synchronized.”
- POLICY-07 — “Effective: December 2025.” (Security/PII focus — no sharing passwords, full card/CVV, auth codes; accounts may be temporarily restricted on suspected fraud.)
- POLICY-08 — **Explicit outdated note**: older documentation referred to a universal “five-user family plan”; no longer offered universally, availability varies by region.
- POLICY-09 — “Last reviewed: June 2026.” **Explicit outdated note**: older 2024 article listed Internet Explorer as a supported browser — “obsolete and should not be used.”
- POLICY-10 — **Explicit outdated/retired note**: older internal billing document instructed agents to ask for first six and last four digits of the card; “that instruction has been retired.” Also: contact Support before chargebacks; don’t request complete card/CVV/PIN/banking passwords/auth codes.

FAQs and tickets have **no hard review dates** in the text (tickets use relative times like “21 days ago,” “September 10,” “8 months ago”); they should be treated as undated, which lowers their authority for current-policy questions.

### 6.4 Contradictions in the corpus (explicit, verbatim-derived)

1. **Refund window — 7-day vs 14-day vs 30-day.**
   - 7-day: older help article (POLICY-02, explicitly archived/outdated).
   - 14-day: current standard for individual course purchases (FAQ-02, POLICY-02).
   - 30-day: user-visible “30-day money-back guarantee” shown at purchase (TICKET-03 screenshot; TICKET-08).
   → Must not be collapsed into one number; 7-day is explicitly non-current; 30-day is a user-visible claim to be verified, not asserted.

2. **Annual billing wording.**
   - Old: “annual subscriptions were billed monthly” (POLICY-01, outdated).
   - Current: annual price shown at checkout is what’s charged (POLICY-01).

3. **Family plan size.**
   - Old: universal “five-user family plan” (POLICY-08, outdated).
   - Current: family plans “when available,” vary by region (POLICY-08).

4. **Browser support.**
   - Old: Internet Explorer supported (POLICY-09 2024 article, obsolete; FAQ-08 doesn’t list it).
   - Current: Chrome/Edge/Firefox/Safari (POLICY-09, FAQ-08).

5. **Offline downloads.**
   - Current: offline download is mobile-app-only, selected courses (FAQ-07, POLICY-04).
   - Old/incorrect: “download courses on your laptop” (FAQ-07 note + TICKET-15 “Offline Learning Guide” article; TICKET-15 — user relied on it, refund review requested).

6. **Progress-save wording.**
   - Old: progress saved “instantly” (POLICY-06 archived doc, outdated).
   - Current: “automatically synchronized,” subject to network delays (POLICY-06; FAQ-03; TICKET-01/TICKET-04).

7. **Caption requirement.**
   - Old: captions required for every video before publication (POLICY-03 old instructor handbook).
   - Current: “whenever practical,” though still encouraged for all (POLICY-03).

8. **Agent payment-data collection.**
   - Old: ask for first six + last four digits of the card (POLICY-10, retired internal doc).
   - Current: request only the minimum needed for transaction identification; never full card/CVV/PIN (POLICY-10).

These are the cases a naive system would mishandle by averaging or by picking the stale version.

### 6.5 Ambiguous / escalation-worthy tickets

- **TICKET-07** — “Cancel my LearnForge.” Ambiguous intent (subscription vs course vs account vs payment); only resolved after agent clarification → refund request. *Pattern: disambiguate before answering.*
- **TICKET-08** — Annual subscription refund; user cites contradictory cancel-page wording; no screenshot; **Escalated due to policy ambiguity.** *Pattern: policy ambiguity + no artifact → escalate.*
- **TICKET-11** — “My course disappeared.” Paid $89 but system shows subscription enrollment; receipt ambiguous; **Escalated** to determine whether receipt is an individual purchase vs another transaction. *Pattern: purchase-type conflict, no resolution → escalate.*
- **TICKET-03** — Refund 21 days out; user claims 30-day guarantee w/ screenshot; standard is 14 days; **Escalated to billing** because promotional terms may differ. *Pattern: standard vs promotional → escalate.*
- **TICKET-15** — User relied on outdated “Offline Learning Guide” (download on laptop); article reported for correction and **refund review requested** because outdated docs were material to a purchase. *Pattern: outdated official doc + material reliance → review/escalation path.*
- **TICKET-13** — Family account, certificate under wrong name; **Transfer request pending.** *Pattern: identity/transfer verification required.*
- **TICKET-10** — Purchased under work email, account under Gmail; **Awaiting verification.** *Pattern: identity verification required before action.*
- **TICKET-02 / TICKET-05 / TICKET-06 / TICKET-14** — pending/authorization/third-party-processor situations needing transaction references or bank timelines; explicitly **not** auto-resolved. *Pattern: billing/chargeback/pending-authorization → need references, escalate/monitoring.*
- **TICKET-12** — Accessibility barrier (no captions); agent reports to content team and provides an alternative transcript. *Pattern: accessibility → content-team reporting + alternative resource; time-sensitive user (“I need to study it today”).*

### 6.6 Security / PII constraints in the data

Repeated across sources (FAQ-15, TICKET-05, TICKET-12, POLICY-07, POLICY-10): do **not**
request or handle full card number, CVV, PIN, banking password, authentication code, or
government ID numbers in support tickets. Acceptable minimum info for transaction
identification: order number, transaction date/amount/last-four, course name. The
assistant must enforce the same constraints and must never ask for the sensitive fields.




## 7. High-level architecture

Components and data flow (diagram in §7.1).

### 7.1 System diagram

```
                         +-----------------------+
   Ingestion (offline):  Corpus reader/parser    |
   parses faqs/policies/tickets.md                  |
   -> per-record chunks with metadata               |
   -> stale-flag + freshness extraction            |
   -> (next milestone) embed & index               |
   +------------------------|----------------------+
                              | derived records + metadata
                              v
                     Index / Evidence Store
   (records + metadata + [later] embeddings)
                              |
   Query path (online)        v
 +------+   +---------------+   +-----------+   +-----------------+
 | User |-->| Conversation  |-->| Query     |-->| Retrieval       |
 | msg  |   | Manager       |   | Understanding| (KB evidence objs)|
 +------+   +---------------+   +---------------+   +-----------------+
                              |                                   |
                              |                                   v
                              |                       +-----------------+
                              |                       | Evidence        |
                              |                       | Assessment      |
                              |                       | (sufficiency,   |
                              |                       | conflict,       |
                              |                       | staleness)      |
                              |                       +-----------------+
                              |                                   |
                              |  decide: GENERATE / CLARIFY / ESCALATE / DECLINE
                              |                                   v
                              |                       +-----------------+
                              |                       | Generator (LLM) |
                              |                       | (evidence packet |
                              |                       | + conversation   |
                              |                       | context, NOT KB) |
                              |                       +-----------------+
                              |                                   |
                              |                                   v
                              |                       +-----------------+
                              |                       | Post-generation  |
                              |                       | verification     |
                              |                       | (citations, PII, |
                              |                       | ungrounded       |
                              |                       | claims)          |
                              |                       +-----------------+
                              |                                   |
                              v                                   v
                        +-------------------+
                        | Response + state   |
                        | update (remember    |
                        | user facts,         |
                        | clarifications,     |
                        | escalation flags)   |
                        +-------------------+
```

**Key architectural decision:** retrieval returns evidence objects; a separate
**Evidence Assessment** step classifies them (sufficiency, conflict/staleness); only
then does generation happen, and only against the evidence packet + explicitly-labeled
conversation context. The generator never sees ticket records as “the answer,” never
sees stale wording as current, and never sees chat history as policy.

### 7.2 Component roles

- **Conversation Manager** — per-session store, **separate from the KB**. Holds user
  facts, pending clarifications, detected intent, and an escalation flag.
- **Query Understanding** — normalizes the query, extracts entities/identifiers
  (order number, course name, dates, transaction amount), classifies intent
  (refund / cancellation / technical / billing / account / accessibility / ambiguous).
- **Retrieval** — returns a small ranked set of evidence objects with metadata.
- **Evidence Assessment (gate before generation)** — deterministic checks: sufficiency,
  conflict/staleness, authority ranking. Yields a decision label passed to generation.
- **Generator (LLM)** — produces text from the evidence packet + conversation context
  + decision label + strict system instructions.
- **Post-generation Verification** — every factual claim must cite evidence; no
  sensitive data; no policy fact not in cited evidence.
- **Response Formatter** — renders the answer with citations; or a clarification
  question; or escalation guidance.

## 8. End-to-end query flow

For each user turn:

1. **Receive message.** Store in Conversation Manager as a user turn (not KB).
2. **Query understanding.** Normalize; extract entities/identifiers; classify intent. If
   intent is unresolvable from this turn + context (e.g. “Cancel my LearnForge” →
   TICKET-07 pattern), **ask a clarifying question and stop** (do not guess).
3. **Retrieve.** Fetch a small ranked set of evidence objects (FAQ/policy as facts;
   ticket as precedent/escalation exemplar) with metadata + `is_stale`/freshness.
4. **Assess evidence (before generation):**
   - sufficiency (does retrieved evidence address the question?),
   - conflict/staleness (current vs stale both retrieved? two records contradict?),
   - authority (policy > faq > ticket; fresh > stale).
   - Produce an **evidence decision**: `supported / conflicting-stale / low-evidence /
     no-evidence`, plus an **escalation signal**.
5. **Route:** GENERATE only if `supported` and non-escalation; CLARIFY if intent
   ambiguous; ESCALATE if signal set; DECLINE if insufficient/no evidence.
6. **Generate** from evidence packet + labeled conversation context + decision + system
   instructions (ground in evidence; cite source_id+source_type+freshness; do not state
   facts absent from evidence; never request/repeat sensitive payment/identity data;
   disclose conflicts/staleness; do not claim unavailable capabilities like live
   account look-ups).
7. **Verify.** Every factual claim must map to a cited source; no sensitive data; no
   policy fact not in cited evidence. On failure: fall back to summarize-evidence-and-
   escalate/decline (never patch with hallucination).
8. **Format + return** (answer w/ citations; or clarification; or escalation guidance
   mirroring the corpus’s outcome types). Update Conversation Manager state for
   follow-up.

---

## 9. Source authority strategy

Authority is tiered and metadata-aware — not “most-chunk-wins”:

1. **Highest:** current **policy** records with an explicit fresh review/effective date
   and no stale flag (e.g. POLICY-01 Feb 2026, POLICY-02 Jan 2026, POLICY-06 Apr 2026,
   POLICY-09 Jun 2026).
2. **Medium:** **FAQ** records (operational guidance, generally current, but mostly
   undated — verify against policy where they conflict; FAQ-02 and POLICY-02 agree on
   the 14-day standard).
3. **Lower:** **ticket** transcripts — precedent/illustration of handling, **never**
   current policy authority. A ticket’s resolving statement is a record that a human did
   X, not a guarantee that X is policy today.
4. **Lowest:** records explicitly flagged **stale/outdated/retired/obsolete**. Usable to
   *acknowledge* an outdated article the user may have seen (and to correct it), but
   never to assert its content as current policy.

**Tie-breaking rules (data-derived):**
- When a user cites an outdated official article (TICKET-15 pattern): do **not** dismiss
  them and do **not** accept it as definitive — state current guidance, note the older
  article is outdated, and offer the review/escalation path when reliance was material.
- When 7-day vs 14-day both appear: treat 14-day as the current standard (POLICY-02
  explicitly excludes the 7-day article from “current standard policy”) and treat 30-day
  as a user-visible promotional/contractual claim to be verified, not asserted
  (TICKET-03, TICKET-08 pattern → escalate to verify promotion terms at purchase date).
- When subscription refund wording conflicts with course-refund wording: treat them as
  different product types (FAQ-02/POLICY-02 distinguish individual courses vs
  subscriptions) — do not average.

---

## 10. Freshness / stale-data strategy

1. **Capture freshness metadata at ingestion** from explicit markers: `Last reviewed`,
   `Effective date`, `Updated`, `Reviewed`, and any explicit “older version …
   outdated/retired/obsolete” sentence (see §6.3).
2. **Prefer current records** when a current + an explicitly-stale record exist on the
   same topic. The stale note is evidence, not noise.
3. Do **not** assume unversioned = current or stale. FAQs and tickets are undated; treat
   them as lower-confidence on freshness. Prefer dated policy excerpts for policy facts.
4. When a user references outdated official docs, respond like the corpus agents do
   (TICKET-15): state current guidance, note the older material is outdated, and — where
  reliance was material — point to the review/escalation path instead of a definitive
  resolution.
5. **Surface freshness in citations** so evaluators/users can see what they’re relying on
   (e.g. “POLICY-02, Effective date: January 2026”; or “FAQ-07 (undated); prefer
   POLICY-02”).
6. For records with no date, rely on authority tiering (policy > faq > ticket) and, when
   they conflict with dated policy, disclose the conflict rather than resolve silently.

---

## 11. Contradiction handling

The architecture must make contradictions visible rather than resolving them silently.

1. **Detect** contradictions via:
   - rule-based matching of the known dataset pairs (§6.4: refund windows; annual
     billing; family-plan size; browser/IE; offline downloads; progress wording;
     caption requirement; retired card-digit instruction), and
   - a light LLM comparison of claims across retrieved records to catch any additional pairs.
2. **Classify** each conflict:
   - `current-vs-stale` (one record explicitly invalidates the other — e.g. 7-day article
     excluded by POLICY-02; annual-billing note in POLICY-01; IE note in POLICY-09);
   - `user-claim-vs-current-policy` (e.g. 30-day guarantee vs 14-day standard;
     cancellation-page wording vs policy);
   - `policy-vs-policy ambiguity` (e.g. subscription refund vs course-refund wording;
     TICKET-08 cancel-page wording).
3. **Respond by class:**
   - current-vs-stale → use current, disclose the stale version is outdated, cite both.
   - user-claim-vs-current-policy → present current standard, note the user-visible claim
     may be promotional/contractual and needs verification, escalate (TICKET-03,
     TICKET-08) — do not assert the user’s number as universal truth.
   - policy-vs-policy ambiguity → disclose the ambiguity and escalate for review of terms
     tied to the user’s purchase/plan/date (TICKET-08, TICKET-11).
4. **Never synthesize a compromise fact or number.** If the corpus is contradictory and
   the applicable terms can’t be determined from freshness/authority alone, the correct
   behavior is disclosure + escalation, not a single confident answer.

---

## 12. Confidence and escalation strategy

### 12.1 Confidence levels

Classify each turn as:
- **Confident & supported** — current, non-contradictory evidence covers the question.
- **Supported but nuanced** — evidence supports an answer with important caveats
  (“may,” “generally,” “where available,” product-specific, depends on plan/purchase
  date). These caveats must appear in the answer.
- **Conflicting/stale** — evidence conflicts or is explicitly outdated.
- **Low-evidence** — relevant material exists but is not conclusive.
- **No-evidence** — nothing useful retrieved.

Only the first is appropriate for a straightforward authoritative-sounding answer.

### 12.2 Escalation classes (corpus-derived)

Escalate rather than fully answer for:
- promotional/contractual refund exceptions differing from the 14-day standard (TICKET-03);
- refund-window / cancellation-page wording ambiguity with no verifying artifact (TICKET-08);
- individual-purchase vs subscription-enrollment identity conflicts with no resolved receipt (TICKET-11);
- account/identity/enrollment/transfer needing verification or admin action (TICKET-10, TICKET-13);
- outdated official documentation that may have influenced a purchase decision (TICKET-15);
- billing/chargeback/duplicate-payment/pending-authorization needing transaction references (TICKET-02, TICKET-05, TICKET-06, TICKET-14);
- accessibility barriers needing content-team reporting + alternative resource (TICKET-12).

“Escalate” in this scope means the assistant’s response reflects that the case needs
human review / more information / a specialized team (and states what is and isn’t known
from the corpus). It does **not** mean fabricating a resolution.

### 12.3 Clarification before answer

For ambiguous/unresolved-intent requests, ask targeted clarifying questions
(subscription vs course vs account vs payment; which course; which order/transaction;
article URL + order number when reliance on outdated docs). Escalation and
clarification are preferred over guessing.

### 12.4 Sensitive-data guard

| government ID numbers. Only minimum info for transaction identification is acceptable.

## 13. Multi-turn conversation strategy

1. **Separate store.** Conversation state (user facts, pending clarifications, detected
   intent, escalation flag) lives separately from the KB and is explicitly labeled
   “user-provided” when passed to generation.
2. **Use context to disambiguate/clarify/follow-up**, e.g. remember “Biology Essentials,”
   “8 months ago,” “$89 receipt,” “don’t have a screenshot,” and use them to ask the
   next useful question.
3. **Never let conversation history override retrieved evidence.** If a user asserts
   something in chat that conflicts with current policy, disclose the conflict and defer
   to current policy / escalation rather than accepting the user’s chat statement as
   truth.
4. **Bounded turns.** Favor a small number of disciplined clarification turns over
   open-ended chit-chat.
5. **Don’t fake capabilities.** The assistant should not imply it can look up the user’s
   real account/orders unless that integration exists (it doesn’t in this scope).
   Where the corpus agents say “I can look up using your email,” the assistant in this
   scope frames that as what Support can do, not as something this assistant can
   actually do against a live account.

*Implemented in M5:* `learnforge/conversation.py` (see `docs/conversation.md`).
The layer keeps a bounded, per-session history, prepares the current retrieval
query deterministically (recent **user** messages + current query — never the
assistant's previous answers), and passes the history to M4 as a delimited,
explicitly non-evidence `<conversation_context>` block. All six M3 states remain
authoritative: conversation context cannot add citations, resolve ambiguity,
suppress escalation, or override `stale`/conflicting evidence.

---

## 14. Hallucination prevention

1. **Evidence-first generation.** The generator only sees quoted evidence excerpts with
   `source_id`+`source_type`+freshness+`is_stale`; system instructions forbid inventing
   policy numbers/terms/dates/fees/contact methods not in that packet.
2. **No parametric policy knowledge as authority.** Even if the model “knows” a
   LearnForge policy from training data, that is not a valid source; only retrieved
   evidence is cited.
3. **Evidence decision gate before generation.** The assessment step classifies
   supported / conflicting-stale / low-evidence / no-evidence, forcing cautious
   language and preventing the generator from smoothing over contradictions.
4. **Citation requirement.** Every factual claim must map to a cited `source_id`.
   Uncited factual claims fail post-generation verification.
5. **Stale-content suppression.** Chunks flagged explicitly outdated are not presented
   as current; surfaced only (a) to correct a user who referenced them, or (b) as an
   explicitly-labeled historical note.
6. **Smaller high-signal retrieval set.** Limiting retrieved context reduces the chance
   the model latches onto a stale chunk and presents it as current.
7. **Number/date verification.** If the generator states a specific policy number,
   refund window, or date, post-verification checks it appears in cited evidence;
   mismatches trigger fallback to cautious/disclose/escalate.

This is not “trust the LLM to be careful.” It is (a) retrieval discipline, (b) a separate
evidence-evaluation gate, and (c) prompt constraints that make grounding and uncertainty
mandatory.

---

## 15. Failure handling

1. **Retrieval returns nothing useful / low relevance.**
   Decline politely, state what (if anything) was found, and route to a clarifying
   question or escalation — never guess. (Mirrors: “I don’t have that in our docs; can
   you share X so I can help?” behavior in the corpus.)
2. **Stale data.** Acknowledge, present current guidance, flag the stale source, and —
   where reliance was material — point to the review/escalation path (TICKET-15). Do
   not present stale specifics as current.
3. **Conflicting evidence.** Disclose; cite both; escalate where applicable terms can’t
   be determined (TICKET-03, TICKET-08, TICKET-11).
4. **Bad retrieval (irrelevant top results).** Treat as insufficient evidence:
   decline/clarify/escalate, and log for evaluation.
5. **LLM slips** (uncited claims, sensitive data, ungrounded policy). Post-generation
   verification catches them; on failure, fall back to a safe summarize-evidence-and-
   escalate response rather than patching with hallucination.
6. **Retrieval/index unavailable (prototype).** Degrade to a safe “I couldn’t find
   authoritative documentation for that” + next step; do not fabricate.
7. **Rate limits / model unavailable.** Retry with backoff + a fallback provider
   (assignment allows multiple free-tier providers); if all fail, degrade as above.
8. **Out-of-scope queries** (e.g. “write my essay”). Decline as out of scope.
9. **Partially-resolvable tickets.** Mirror the corpus outcome type (pending, awaiting
   reference, refund review requested, escalated) instead of inventing a closed
   resolution.
10. **Sensitive data received.** Do not store/forward full card/CVV/PIN/passwords/
   auth-codes/govt ID; redact from any internal logging; instruct the user not to share
   them. (Real handling of received sensitive data is out of scope; the requirement is
       behavioral consistency with the corpus.)

## 16. Proposed technology choices

> Proposals only — to be confirmed at implementation. This milestone performs none of
> these (no installation/embeddings/vector-DB/LLM calls).

- **Language:** Python (natural fit; easy to run/evaluate on the small corpus).
- **Generator (LLM):** a **free-tier** LLM API (assignment allows Groq / Gemini /
  OpenRouter). Propose a single provider interface so the choice is swappable; not
  dependent on a paid tier.
- **Embeddings:** a free-tier embedding model from the same provider family (e.g.
  `nomic-embed-text`-style or the provider’s embeddings endpoint).
- **Index / store:** ~40 records, so the simplest robust choice is a small **local
  embedding store** (single JSON/Parquet file or a lightweight local DB) with
  cosine-similarity retrieval. A full managed vector DB is explicitly **out** for M0;
  a tiny local vector library is optional. Keeps the prototype locally runnable and
  dependency-light.
- **Retrieval:** dense semantic search over per-record chunks (one record = one chunk,
  since IDs/metadata already align to these units); optionally a light lexical fallback
  for exact IDs like “FAQ-08” or the cancel-page wording “cancel within 14 days.”
- **Evidence assessment + post-verification checks:** rule-based + small prompts (stale
  flag, contradiction pairs listed in §6.4, citation check, PII guard) — deterministic
  and reviewable.
- **Conversation state:** in-memory per session (one session = one conversation context);
  sufficient for the demo.
- **Framework:** a thin custom pipeline (not LangChain/LlamaIndex) to keep control over
  evidence-separation and guardrails and to stay within scope. (Alternative noted;
  argued against below.)

### Proposed data schema (record)

One record per source entry:

```
{
  "source_id": "POLICY-02",          # e.g. FAQ-08, POLICY-09, TICKET-15
  "source_type": "faq"|"policy"|"ticket",
  "title": "Cancellation and Refund Policy",
  "chunk_text": "<verbatim source text>",
  "metadata": {
    "freshness": "Effective date: January 2026",   # or "undated"
    "is_stale": true,                 # true if text explicitly says older version is wrong/outdated/retired/obsolete
    "stale_reason": "references archived 7-day refund article",
    "authority": 2,                   # policy=>3, faq=>2, ticket=>1
    "ticket_status": "Escalated due to policy ambiguity"  # tickets only
  },
  "vector": [ ... ]   # embeddings added at ingest (next milestone)
}
```

Chunking: **one record = one chunk** (already ID-aligned; small corpus).
Tickets may optionally get a one-line issue summary for better retrieval of relevant
precedents without altering the transcript.

---

## 17. Testing strategy

Focus on the **failure modes the corpus is designed to expose**, not just happy-path
FAQ answering.

- **Retrieval tests.** Given queries, assert the top-k includes the expected
  `source_id`(s) — incl. current vs stale. E.g. a refund-window query must retrieve
  FAQ-02 + POLICY-02 and the stale POLICY-02 7-day note.
- **Evidence-assessment tests (deterministic):**
  - low/no-sufficiency on out-of-corpus or unanswerable queries (e.g. competitor pricing);
  - conflict detection returns `conflicting-stale` for refund-window and annual-billing
    queries (both current and stale retrieved);
  - staleness detection asserts the 7-day article, IE article, “billed monthly” note,
    five-user plan, desktop-download article, “instantly,” “every video,” and retired
    card-digit instruction are flagged `is_stale`.
- **Generation/grounding tests.** For each query class, assert behavior:
  - supported → answer cites `source_id`+`source_type` (+ freshness);
  - conflicting-stale → answer discloses conflict/staleness, cites both, escalates for
    refund-window/promotional cases;
  - ambiguous intent → asks a clarifying question, does not guess (TICKET-07);
  - out-of-scope → declines;
  - stale-relied-on (TICKET-15) → corrects + offers review/escalation path.
- **Safety/PII tests.** Assert no request for or echo of full card/CVV/PIN/passwords/
  auth codes/govt ID.
- **No-fabrication tests.** Assert policy numbers/dates/windows stated in the answer
  appear in cited evidence; assert nothing is asserted when evidence is `no-evidence`.
- **Conversation tests.** Multi-turn: disambiguate first turn, then answer using KB
  evidence while treating user facts as context; assert a chat assertion that conflicts
  with policy does not override it.
- **Integrity test.** Assert source files unchanged after a run.

Testable against the small corpus with no live account access needed; many checks can
be deterministic.

## 18. Evaluation strategy

Evaluation measures whether the system handles the corpus’s intentional hard parts.

- **Primary: groundedness/attribution.** Share of factual claims backed by a cited
  `source_id` (and source_type); and whether any asserted policy fact is absent from
  cited evidence (a hallucination). This dominates fluency/brevity.
- **Secondary metrics:**
  - stale-data handling: current preferred, outdated flagged, review/escalation path
    offered where relevant;
  - contradiction handling: 7-vs-14-vs-30-day refund and subscription-vs-course wording
    disclosed instead of averaged;
  - escalation vs over-answering: cases that escalated in the corpus (TICKET-03,
    TICKET-08, TICKET-11, TICKET-15, billing-pending cases) should escalate/hedge,
    not be confidently resolved;
  - ambiguity handling: ambiguous requests (TICKET-07) should clarify, not guess;
  - citation completeness: claims cite source_id+type + freshness where available;
    ticket sources labeled precedent;
  - conversation isolation: chat assertions not treated as policy;
  - safety: zero PII/security slips.
- **Method:** a curated eval set drawn from the corpus — representative factual queries
  (FAQ-derived) + each contradiction/stale zone + each escalation class + ambiguous
  requests + out-of-scope/unsupportable queries. Human rubric scoring (and, where
  available, an LLM-as-judge with the above rubric; no large external benchmark needed
  at this scope).
- **Failure prioritization.** A fluent wrong answer is worse than a cautious “I don’t
  have enough to answer / this needs review.” Penalize confident fabrication heavily,
  especially on contradiction and stale-data cases.

---

## 19. Trade-offs

- **Thin custom pipeline vs framework (LangChain/LlamaIndex).** Custom gives direct
  control over evidence separation and guardrails and fewer “magic” failures; a
  framework starts faster but is harder to constrain. For a ~40-record corpus with a
  need to *explicitly* not treat chat history/the LLM as authority, custom is worth the
  small extra wiring.
- **Local/small store vs managed vector DB.** ~40 records make a tiny local store plenty;
  a managed DB adds failure surface and dependencies with zero benefit at this scale.
  Trade simplicity/locality for scalability here.
- **Dense retrieval vs lexical fallback.** Dense retrieval can over-match stale docs;
  that is why the Evidence Assessment gate exists. A lexical fallback (for exact IDs /
  quoted policy strings) improves precision on the dataset’s ID-referencing cases.
- **Rule-based stale/conflict flags vs LLM-only judgment.** Rules on the known stale
  markers are cheap and reliable (§6.4); LLM judgment is more general but can miss or
  invent conflicts. Use rules for the known cases, LLM only to catch *additional*
  pairs, and keep the answer cautious regardless.
- **Escalation conservatism vs decisiveness.** Escalating and expressing uncertainty
  looks less “helpful,” but for this corpus the confident answer is frequently the
  wrong one. Favor honesty/correctness.
- **One-chunk-per-record vs fine sub-chunking.** One-per-record preserves clean IDs and
  citations and is plenty for 40 small records; fine chunking would fragment
  authority/freshness metadata and complicate citation. Keep it whole-record.
- **Not implementing in M0.** Delivering only architecture.md (no code/embeddings/vector
  DB/LLM calls) is a deliberate scope call so the design can be reviewed before
   engineering. Trades speed-of-iteration for up-front clarity on the hardest parts.

---

## 20. Future improvements

- Hybrid retrieval (BM25 + dense) with re-ranking; query rewriting/multiquery.
- A dedicated vector DB / managed index only if the corpus or QPS grows beyond the
  assignment scale.
- Richer freshness pipeline: versioned records with explicit current/archived status
  rather than relying on inline “older version” sentences.
- Structured slot-filling to collect order number, course, transaction date/amount, and
  to refuse to even prompt for sensitive PII.
- Grounding via constrained/structured generation (forced-citation / retrieval-augmented
  verification pass that re-checks claims against evidence).
- A learned (small) escalation classifier trained on the escalation-worthy tickets,
  with the rule-based classes as a high-precision backbone.
- Regression eval set that grows from the 8 ambiguity/escalation tickets + FAQ-derived
  factual cases, with a small LLM-as-judge rubric.
- Conversation memory with coreference resolution + entity extraction.
- Per-course / per-product exception modeling (the corpus already shows variation:
  certificates, offline eligibility, captions, assessments — currently over-generalized).
- Explicit PII redaction of any ticket text containing even partial card digits before
  it reaches the generator (defense in depth on POLICY-10’s retired practice).
- Telemetry/logging of declined/escalated/uncertain turns for continuous improvement
  (with privacy safeguards), and integrations with real account/order/support systems
  (out of scope here).

---

## Assumptions requiring your approval

1. **Scope of this milestone.** Only `docs/architecture.md` is produced here — no code,
   no dependencies, no embeddings, no vector DB, no LLM calls. Implementation
   (including the data schema + diagram the assignment also asks for) follows next.
2. **Assignment requirements source.** Taken from the supplied `.docx` (build a retrieval
   + generation support assistant that reduces hallucination, handles multi-turn, and
   escalates when not confident; ~4–6 hours; free-tier LLM APIs OK) plus the corpus
   `README.md`’s explicit statement that contradictions/outdated/ambiguous data are
   intentional. Confirm nothing additional is intended.
3. **Tickets = evidence, not authority.** Past tickets are triage precedents, not
   current policy.
4. **Freshness interpretation.** I rely on the explicit `Last reviewed / Effective /
   Updated / Reviewed` date lines and the explicit “older version is wrong/outdated/
   retired/obsolete” callouts in policy text to determine current vs stale; I have
   **not** invented dates for unversioned records (FAQs; tickets; POLICY-05/POLICY-10
   which show no hard date line).
5. **Local/small stack preference.** Propose a small, locally-runnable pipeline
   (Python + local embedding store, dense+lexical retrieval) to fit the 4–6 hour scope
   and stay dependency-light. Confirm whether the evaluator expects a specific embedding
   model, vector store, or cloud LLM, or “small and locally runnable” is the right
   target.
6. **Out-of-scope items.** Authentication, live account/order lookup, real ticketing/
   payment systems, and KB authoring are all assumed out of scope.
7. **TICKET-05, TICKET-10/11 fully read.** Now captured in §6.5: TICKET-05 = unrecognized
   charge that was a subscription renewal, cancellation completed + refund review
   requested; TICKET-10 = course bought under work email with account under Gmail,
   awaiting verification; TICKET-11 = “course disappeared,” $89 receipt ambiguous
   between individual purchase and subscription enrollment, escalated. All confirm the
      ambiguity/escalation design; no change to requirements.
8. **Word-document deliverables.** The .docx also asks for a data schema, a system
   design diagram, and a README (failure-handling / eval / trade-offs). The schema
   (§16) and the diagram (§7.1) are provided here so the architecture is
   self-documenting; the full README + working prototype follow in implementation.
   Confirm whether the M0 level of detail is sufficient.

## Summary

### Files inspected (this milestone)
- `Applied AI-LLM Engineer - Take-Home Assignment.docx` — assignment instructions
  (read via XML extraction; requirements captured above).
- `learnforge-knowledge-base-data/learnforge-knowledge-base/README.md`
- `learnforge-knowledge-base-data/learnforge-knowledge-base/faqs.md` (FAQ-01…15)
- `learnforge-knowledge-base-data/learnforge-knowledge-base/policies.md` (POLICY-01…10)
- `learnforge-knowledge-base-data/learnforge-knowledge-base/tickets.md` (TICKET-01…15)

### Important requirements discovered
- Retrieval + generation assistant over FAQs / policy docs / past tickets; reduce
  hallucination; handle multi-turn; escalate to human when not confident (.docx).
- Retrieved evidence treated separately from the LLM (.docx + README).
- Recognize insufficient evidence; recognize conflicting/stale evidence; escalate
  uncertain cases rather than hallucinate (.docx).
- Conversation context separated from authoritative KB content (.docx).
- Sources retain metadata: document ID, source type, dates/version/freshness (.docx +
  dated policy excerpts).
- ~4–6 hour scope; free-tier LLM APIs; deliver public Git repo + README
  (failure-handling / eval / trade-offs) + data schema + system design diagram (.docx).
- Do not modify/rewrite/delete/move the supplied data (README).

### Important dataset issues discovered
- 40 discrete ID-aligned records across three source types; IDs (FAQ-XX / POLICY-XX /
  TICKET-XX) are natural citation keys.
- Intentional, source-spanned contradictions: refund window 7-day (stale, POLICY-02) vs
  14-day (current standard, FAQ-02 + POLICY-02) vs 30-day (promotional claim,
  TICKET-03/TICKET-08); annual billing “billed monthly” (stale, POLICY-01) vs “annual
  price at checkout” (current); “five-user family plan universal” (stale, POLICY-08) vs
  “may vary by region” (current); IE supported (stale 2024 article, POLICY-09 + FAQ-08)
  vs modern browsers only; desktop offline-download article (stale, FAQ-07 + TICKET-15)
  vs mobile-only offline (current, FAQ-07/POLICY-04); progress “instantly” (stale,
  POLICY-06) vs “automatically synchronized” (current); captions “every video” (stale,
  POLICY-03) vs “whenever practical” (current); agent card-digit collection “first six
  and last four” (retired, POLICY-10) vs “minimum needed only” (current).
- Policies carry explicit freshness metadata + overt “older version is wrong” callouts
  that must be honored, not averaged.
- Many tickets are unresolved/pending/escalated (TICKET-02 pending monitoring,
  TICKET-06 awaiting transaction reference, TICKET-08 escalated, TICKET-11 escalated,
  TICKET-13 transfer pending, TICKET-14 pending bank auth, TICKET-15 refund review
  requested). They are triage precedents the assistant should mirror, not resolve.
- Ambiguous tickets requiring disambiguation/escalation: TICKET-03, TICKET-07,
  TICKET-08, TICKET-11, TICKET-15 (+ TICKET-13, TICKET-10 on verification).
- Repeated explicit PII/security rules: never request/handle full card/CVV/PIN,
  banking password, auth code, or government ID (FAQ-15, TICKET-05/12, POLICY-07,
  POLICY-10) — assistant must enforce this.

### Files created (this milestone)
- `docs/architecture.md` — the primary deliverable (this document).
- `extract_docx.py` and `assignment_docx_text.txt` — diagnostic artifacts used only to
  read the `.docx` assignment requirements into plain text; **not** part of the
  deliverable. These should be removed (or moved to a scratch location) before the
  prototype commit.
