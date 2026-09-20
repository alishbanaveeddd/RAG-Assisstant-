# LearnForge KB Data Schema & Ingestion (Milestone M1)

> **Milestone:** M1 — Data Schema + Ingestion layer.
> **Status:** implemented. **No LLM, no retrieval algorithm, no generation, no
> chat/API/UI, no embeddings, no vector DB** exist yet — this milestone only turns
> the supplied Markdown into a validated, structured, deterministic representation.
> **Source of truth:** the supplied files are read **read-only** and are never
> modified, rewritten, moved, or deleted.

This document supplements `docs/architecture.md` (M0). It does not replace or
rewrite it; it records the concrete schema and ingestion behaviour implemented in M1.

---

## 1. Layout and how to run

```
learnforge/
  __init__.py        # public exports
  schema.py          # Record dataclass, authority map, validation rules
  ingest.py          # parsing, normalization, validation gate, JSON writer, CLI
tests/
  test_ingest.py     # M1 ingestion tests (43)
  test_retrieval.py  # M2 retrieval tests (47)
docs/
  architecture.md    # M0
  data-schema.md     # this file (M1)
data/processed/
  kb_records.json    # generated, deterministic output (not a source)
```

Run ingestion (writes `data/processed/kb_records.json` and prints a report):

```
python -m learnforge.ingest
```

Options: `--base <dir>`, `--out <path>`, `--no-strict`, `--no-write`.

Run the tests (they run under **pytest** — already installed here — and also under
the standard library's `unittest`, so no new test dependency is introduced):

```
python -m pytest tests -q
python -m unittest discover -s tests -v   # equivalent, standard library only
```

---

## 2. Record schema

One record per supplied entry (**one record == one chunk**, per M0 §16). The
unmodified M0 fields are listed first; the additive M1 fields follow.

### 2.1 M0 fields (verbatim from `docs/architecture.md` §16)

| Field | Type | Meaning |
|---|---|---|
| `source_id` | string | Stable record ID, e.g. `FAQ-08`, `POLICY-09`, `TICKET-15`. |
| `source_type` | `"faq"` \| `"policy"` \| `"ticket"` | Source kind (lowercase). |
| `title` | string | The record heading text (FAQ question / policy title / ticket subject). |
| `chunk_text` | string | **Verbatim** source text of the record, including its heading line. |
| `metadata` | object | See §2.3. |
| `vector` | `null` (M1) | Embedding slot; populated in a later milestone only. |

### 2.2 Additive M1 fields

M1 explicitly asks the schema to also preserve a **citation key** and
**escalation/ambiguity indicators**. These are added without altering the M0
fields (see §8 "Relationship to M0"):

| Field | Type | Meaning |
|---|---|---|
| `citation_key` | string | Stable citation handle; equals `source_id` (M0 §FR-18 cites `source_id` + `source_type`). |

### 2.3 `metadata` object

| Field | Type | M0/M1 | Meaning |
|---|---|---|---|
| `freshness` | string | M0 | Canonical freshness line, e.g. `"Effective date: January 2026"`, or `"undated"`. |
| `is_stale` | boolean | M0 | `true` only when the record text explicitly says a prior version/document is outdated. |
| `stale_reason` | string \| `null` | M0 | **Verbatim** stale sentence(s) from the source; `null` when not stale. |
| `authority` | integer | M0 | `policy => 3`, `faq => 2`, `ticket => 1`. |
| `ticket_status` | string \| `null` | M0 | Ticket `STATUS:` value (tickets only; `null` otherwise). |
| `escalated` | boolean | M1 | Ticket status indicates escalation (tickets only; otherwise `false`). |
| `unresolved` | boolean | M1 | Ticket is not a plain `Resolved` outcome (tickets only). |
| `ambiguity_flags` | list[string] | M1 | Ambiguity/escalation indicators (see §5). |
| `contradiction_topics` | list[string] | M1 | Topic hints to support later contradiction detection/retrieval (see §6). |

### 2.4 Example record

```json
{
  "source_id": "POLICY-02",
  "source_type": "policy",
  "title": "Cancellation and Refund Policy",
  "chunk_text": "# POLICY-02 — Cancellation and Refund Policy\n\nYou can cancel ...\n\nEffective date: January 2026.",
  "citation_key": "POLICY-02",
  "metadata": {
    "freshness": "Effective date: January 2026",
    "is_stale": true,
    "stale_reason": "Older help-center documentation referenced a 7-day refund period for all digital products. That article remains accessible in some archived search results but should not be treated as the current standard policy.",
    "authority": 3,
    "ticket_status": null,
    "escalated": false,
    "unresolved": false,
    "ambiguity_flags": [],
    "contradiction_topics": ["refund_window"]
  },
  "vector": null
}
```

---

## 3. Parsing and normalization behaviour

1. Each file is opened **read-only**; its SHA-256 is recorded for integrity checks.
2. Line endings are normalized to `\n` so parsing is platform-stable.
3. Records are segmented **by heading** (`^# (FAQ|POLICY|TICKET)-\d\d — title`), not
   by `---` separators, so a missing or extra separator cannot silently merge or
   drop a record.
4. A trailing cross-file label (e.g. `SECTION 2 — POLICY / HELP-CENTER DOCUMENT
   EXCERPTS`) and trailing `---`/blank lines are trimmed from the last record of a
   file.
5. `chunk_text` is the verbatim record text (heading included). Because trimming
   only removes contiguous boundary text, `chunk_text` is always an exact substring
   of the normalized source file — this is asserted by a test.
6. `title` is the heading text after the record ID.
7. No record text is rewritten, summarized, or reworded.

---

## 4. Freshness rules

- A freshness value is extracted **only** from an explicit line beginning with
  `Last reviewed`, `Last updated`, `Effective date`, `Effective`, `Updated`, or
  `Reviewed` followed by `:` and a value.
- The value is canonicalized to `"<label>: <value>"` with the trailing period
  removed (e.g. `"Reviewed: April 2026"`).
- If no such line exists, `freshness` is `"undated"`.
- **No date is ever invented.** FAQs and tickets have no date lines in the corpus,
  so they are all `"undated"`; `POLICY-05`, `POLICY-08`, `POLICY-10` are also
  `"undated"`. A test asserts this.
- M1 does not decide whether a record is *current*; it only records the explicit
  date/version information where present (M0 §10).

Observed policy freshness values:

| Record | `freshness` |
|---|---|
| POLICY-01 | `Last reviewed: February 2026` |
| POLICY-02 | `Effective date: January 2026` |
| POLICY-03 | `Updated: March 2026` |
| POLICY-04 | `Last updated: May 2026` |
| POLICY-05 | `undated` |
| POLICY-06 | `Reviewed: April 2026` |
| POLICY-07 | `Effective: December 2025` |
| POLICY-08 | `undated` |
| POLICY-09 | `Last reviewed: June 2026` |
| POLICY-10 | `undated` |

---

## 5. Stale detection and ticket indicators

### 5.1 Stale detection (`is_stale`, `stale_reason`)

A record is flagged `is_stale` when its own text contains an **explicit statement
that a prior document/version was wrong or is no longer current** (M0 §16).

Detection requires a phrase of the form `(older|previous|archived)` **followed by a
document noun** (`version`, `documentation`, `article`, `guide`, `handbook`,
`help-center documentation`, `mobile help article`, `internal billing document`,
`instructor guide`, …), optionally with a year (`older 2024 article`).

A bare `older`/`previous` keyword is deliberately **not** sufficient, because it
false-positives on unrelated text present in this very corpus:

- FAQ-03: *"older progress information"*
- FAQ-06: *"Certificates from older courses"*
- FAQ-08: *"an older laptop"* / *"Older devices"*
- FAQ-12: *"previous purchases"*
- TICKET-04: *"older synchronized data"*

`stale_reason` stores the **verbatim** marker sentence, plus the immediately
following sentence when that sentence contains a corrective clause
(`outdated`, `obsolete`, `retired`, `no longer`, `removed`, `replaced`,
`should not be treated`, …).

Stale detection applies to **faq** and **policy** records only. Ticket transcripts
are historical evidence (M0 §9) and are never flagged as stale *sources*; e.g.
TICKET-15 discusses an outdated help article but is itself a ticket record.

Corpus result: exactly **10** stale records — `FAQ-07`, `POLICY-01`, `POLICY-02`,
`POLICY-03`, `POLICY-04`, `POLICY-05`, `POLICY-06`, `POLICY-08`, `POLICY-09`,
`POLICY-10`. `POLICY-07` is **not** stale (it contains no prior-version note).

### 5.2 Ticket status, escalation and ambiguity

- `ticket_status` is the verbatim `STATUS:` value (trailing period removed).
- `escalated` is `true` when the status contains "Escalat"
  (`TICKET-03`, `TICKET-08`, `TICKET-11`).
- `unresolved` is `true` unless the status is exactly `Resolved`
  (`TICKET-01`, `TICKET-04`, `TICKET-09` are the only resolved tickets).
- `ambiguity_flags` records ambiguity/escalation indicators. Values are derived
  from the status text plus a small, documented mapping for the specific tickets
  M0 §6.5 identified as ambiguous or escalation-worthy:

| Ticket | `ambiguity_flags` | Grounding |
|---|---|---|
| TICKET-03 | `promotional_terms_may_differ` | standard vs promotional refund terms |
| TICKET-07 | `ambiguous_intent_requires_clarification` | "Cancel my LearnForge" |
| TICKET-08 | `policy_wording_ambiguity` | cancel-page wording vs policy |
| TICKET-10 | `identity_verification_required` | work email vs Gmail account |
| TICKET-11 | `purchase_type_conflict` | $89 receipt ambiguous |
| TICKET-13 | `enrollment_transfer_requires_review` | certificate/transfer |
| TICKET-15 | `outdated_documentation_reliance` | relied on outdated article |

Flags are indicators only — M1 does **not** decide outcomes.

---

## 6. `contradiction_topics` (support for later evidence logic)

A deterministic keyword map tags topics so M2+ can group current-vs-stale pairs
(M0 §11) without re-deriving structure. Topics: `refund_window`, `annual_billing`,
`family_plan`, `browser_support`, `offline_downloads`, `progress_sync`,
`captions_accessibility`, `payment_data_collection`.

These are **tags, not resolutions**: M1 does not decide which of the contradictory
refund windows (7-day / 14-day / 30-day) is correct, does not choose the current
browser list, and does not pick between offline-download wordings. The conflicting
text remains verbatim in `chunk_text`, and the stale variant is marked via
`is_stale`/`stale_reason` only.

---

## 7. Validation rules

Every record is validated; malformed records are **never silently accepted**.

| Rule | Failure message includes |
|---|---|
| `source_id` matches `(FAQ|POLICY|TICKET)-\d\d` | `does not match the required pattern` |
| `source_id` prefix agrees with `source_type` | `implies source_type` |
| `title` non-empty | `title` |
| `chunk_text` non-empty | `chunk_text` |
| `citation_key` non-empty and equal to `source_id` | `citation_key` |
| all M0 metadata keys present | `metadata.<key> is required` |
| `freshness` is a non-empty string | `metadata.freshness` |
| `is_stale` is a boolean | `metadata.is_stale` |
| `authority` matches `source_type` | `metadata.authority` |
| `is_stale == true` ⇒ `stale_reason` present | `stale_reason` |
| `is_stale == false` ⇒ `stale_reason` is null | `stale_reason` |
| tickets have a `ticket_status`; non-tickets must not | `ticket_status` |
| `escalated` / `unresolved` are booleans and ticket-only | `escalated`/`unresolved` |
| `ambiguity_flags` / `contradiction_topics` are lists of strings | the key name |
| `vector` is `null` in M1 | `vector` |
| `source_id` values are unique across the corpus | `duplicate source_id` |
| file grouping matches the expected type (faqs → faq, etc.) | `does not match expected` |
| each file yields at least one record | `no records matched` |

Modes:

- `ingest_all(..., strict=True)` raises `IngestionError` listing every problem.
- `ingest_all(..., strict=False)` returns the errors in `IngestResult.errors`.

---

## 8. Output document

`data/processed/kb_records.json`:

```json
{
  "schema_version": "m1",
  "source_files": { "faqs.md": "<sha256>", "policies.md": "<sha256>", "tickets.md": "<sha256>" },
  "counts": { "faq": 15, "policy": 10, "ticket": 15, "total": 40 },
  "records": [ /* 40 records, in source order: faqs, then policies, then tickets */ ]
}
```

**Determinism:** no clocks, randomness, or network are used; key order is fixed;
records follow a fixed file order (faqs → policies → tickets) and source order
within each file. Two runs produce byte-identical output (asserted by a test).

**Integrity:** the SHA-256 of each source file is recorded in the output and
checked by tests, so any modification of the supplied data is detectable.

### Relationship to M0

M0 §16 proposed the record schema; M1 implements it **exactly** for its six fields.
M1 adds only what M1 itself asked for (`citation_key`, and the escalation /
ambiguity / contradiction-support metadata). Nothing in M0 is changed or removed,
and no architectural decision is altered: this is still a thin custom Python
pipeline with no framework, no embeddings, and no vector database.

**Test-framework gap (needs confirmation).** M0 §17 specified a testing *strategy*
but did not name a test framework, while M1 asked to "use the testing framework
specified in M0". Rather than invent a stack, the tests are written to run under
**both** `pytest` (already installed in this environment) and the standard library
`unittest` — so no new dependency is introduced either way.

---

## 9. Explicitly not done in M1

- No LLM calls, no embeddings, no vector database, no retrieval, no generation.
- No chat/API/UI.
- No modification of the supplied source files.
- No resolution of contradictions (7-/14-/30-day refunds, annual billing, offline
  downloads, browser support, captions, sync wording, payment-data collection) —
  only metadata to distinguish them later.
- No conversion of ticket transcripts into policy.