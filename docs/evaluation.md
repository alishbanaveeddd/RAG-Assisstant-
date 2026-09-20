# Evaluation (M7)

## Purpose

A small, deterministic benchmark that measures **behavioral correctness** of the
end-to-end pipeline: routing, evidence surfacing, citation safety, escalation,
multi-turn handling, and failure safety. It exercises the **real** M2 retriever
and **real** M3 assessor; the only stub is M4's `FakeProvider` (offline,
deterministic, no API key).

## What is measured — and what is deliberately not

Measured (deterministic, per case and rolled up by dimension):

| Dimension | Checks |
|---|---|
| `routing` | M3 state, routing action, security/escalation flags, conflict topics |
| `retrieval` | required evidence IDs surfaced where genuinely required; stale evidence visible where expected |
| `grounding` | citations ⊆ allowed citations; no invalid citation keys; no citations on decline cases |
| `conversation` | prepared query preserves prior-turn topic; history recorded; context never upgrades an insufficient answer or resolves an ambiguity |
| `failure_safety` | structured failure/error mode; correct failure type; no fabricated answer; conversation not updated |

**Not measured here.** Free-form answer *wording* quality and a calibrated
hallucination rate cannot be derived from deterministic checks against a fake
provider: they require human review or a separately validated LLM-judge
evaluation with real model responses. The benchmark instead reports
**grounding / unsupported-answer behavioral checks** (e.g., decline cases cite
nothing, all citations fall inside the allowed evidence). Citation *presence*
in answers is only observable with a real provider, because `FakeProvider`
emits no citation markers by design.

## Case matrix (28 cases)

| Category | # | Expected behavior |
|---|---|---|
| Answerable | 3 | grounded answer path, required evidence surfaced |
| Conflict | 4 | known contradiction family detected (refund 7/14/30, annual billing, captions); refunds escalate |
| Stale | 4 | stale evidence retrieved and visible; current-vs-stale conflict flagged (C4: stale present but *not* a conflict — negative control) |
| Insufficient | 2 | decline; no citations |
| Security | 3 | `security_escalation`; never requests CVV/password/auth code |
| Ambiguous | 2 | clarification required; no invented interpretation |
| Ticket evidence | 2 | historical tickets surface as evidence; never silently override policy |
| Multi-turn | 4 | topic preserved in prepared query; routing preserved (incl. insufficient stays insufficient) |
| Failure | 4 | retrieval/assessment/generation/provider failures are structured and safe |

Cases live in `data/eval/cases.json`; expected values are behavioral
(state/routing/flags/conflict topics/required IDs), never reference wording.
Exact record IDs are asserted only where genuinely required (current-vs-stale
and contradiction cases); otherwise relevant-evidence and routing behavior is
evaluated.

## How to run

```bash
python -m learnforge.evaluation                 # human-readable, offline (FakeProvider)
python -m learnforge.evaluation --json-out results.json
python -m pytest tests/test_evaluation.py -q    # benchmark harness tests
```

Exit code 0 iff every case passes and the run is deterministic (the CLI re-runs
the benchmark and compares per-case signatures).

## Results (actual output of the benchmark run)

```
category          passed  /total
answerable             3/3
conflict               4/4
stale                  4/4
insufficient           2/2
security               3/3
ambiguous              2/2
ticket_evidence        2/2
multi_turn             4/4
failure                4/4

dimension             passed  /total
routing                   66/66
retrieval                 17/17
grounding                 51/51
conversation               8/8
failure_safety            16/16

cases: 28  passed: 28  deterministic: True  ALL PASSED: True
```

All 158 behavioral checks pass and the benchmark is byte-for-byte deterministic
across repeats.

## Limitations

- 28 hand-maintained cases: small by design (maintainable, transparent), not a
  statistical quality measure.
- FakeProvider answers are canned; grounding checks therefore cover the
  *pipeline's* citation enforcement, not the LLM's prose.
- No automated wording-quality / faithfulness scoring (see above).
- Thresholds inside M3 (relevance bands, conflict matching) are heuristics; the
  benchmark verifies behavior, not threshold optimality.
