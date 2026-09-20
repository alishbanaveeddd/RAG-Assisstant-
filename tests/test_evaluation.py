"""M7 — tests for the deterministic evaluation benchmark.

The benchmark itself runs the real pipeline (real retriever + real M3) against
M4's FakeProvider, so these tests verify both the harness and — transitively —
the end-to-end behavior of the system, all offline.
"""
from __future__ import annotations

import json

import pytest

from learnforge.evaluation import (
    DEFAULT_CASES_PATH,
    EvalReport,
    build_assistant,
    format_report,
    load_cases,
    run_benchmark,
)


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture(scope="module")
def report(cases):
    return run_benchmark(cases_data=cases, repeats=1)


# --------------------------------------------------------------------------- #
# Harness correctness
# --------------------------------------------------------------------------- #
def test_cases_file_loads_and_validates(cases):
    ids = [c["id"] for c in cases["cases"] + cases["failure_cases"]]
    assert len(ids) == len(set(ids))
    assert len(cases["cases"]) >= 20
    assert len(cases["failure_cases"]) >= 3


def test_load_cases_rejects_malformed_input(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cases": [{"id": "Z1", "category": "nope"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(bad)

    dup = tmp_path / "dup.json"
    case = {"id": "Z1", "category": "conflict", "query": "q", "expected": {"state": "x"}}
    dup.write_text(json.dumps({"cases": [case, dict(case)]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(dup)


def test_benchmark_report_structure(report):
    assert isinstance(report, EvalReport)
    assert len(report.results) == 28
    assert report.all_passed is True
    assert report.deterministic is True
    payload = report.to_dict()
    # JSON-serialisable and free of secrets/raw prompts.
    text = json.dumps(payload)
    assert "api_key" not in text.lower()
    assert "GROQ" not in text
    assert "system:" not in text.lower().replace("system rules", "")


def test_report_is_deterministic_across_repeats(cases):
    repeat_report = run_benchmark(cases_data=cases, repeats=3)
    assert repeat_report.deterministic is True
    assert repeat_report.all_passed is True


def test_benchmark_can_fail_on_wrong_expectations(cases):
    """The harness must be able to detect a regression, not only pass."""
    mutated = json.loads(json.dumps(cases))
    mutated["cases"][0]["expected"]["state"] = "answerable"  # A1 is answerable;
    # make the expectation wrong:
    mutated["cases"][0]["expected"]["state"] = "clarification_required"
    bad_report = run_benchmark(cases_data=mutated, repeats=1)
    assert bad_report.all_passed is False
    failed_ids = [r.case_id for r in bad_report.results if not r.passed]
    assert failed_ids == ["A1"]
    assert any(c["name"] == "state" and not c["ok"] for c in bad_report.results[0].checks)


def test_harness_reports_exceptions_as_failed_cases(tmp_path, monkeypatch):
    from learnforge.evaluation import _FaultRetriever

    def broken_factory(**kwargs):
        return build_assistant(retriever=_FaultRetriever(RuntimeError("boom")))

    tiny = {"cases": [
        {"id": "T1", "category": "conflict", "query": "How long do I have to request a refund?",
         "expected": {"state": "conflicting_evidence"}},
    ]}
    report = run_benchmark(cases_data=tiny, factory=broken_factory, repeats=1)
    assert report.all_passed is False
    detail = report.results[0].checks[0]["detail"]

# --------------------------------------------------------------------------- #
# Behavioral assertions on the real benchmark run
# --------------------------------------------------------------------------- #
def _cat_results(report, category):
    return {r.case_id: r for r in report.results if r.category == category}


def test_answerable_cases_pass_and_are_grounded(report):
    a = _cat_results(report, "answerable")
    assert set(a) == {"A1", "A2", "A3"}
    for res in a.values():
        assert res.passed and res.state == "answerable" and res.mode == "llm"


def test_conflict_cases_detect_known_families_and_escalate_refunds(report):
    b = _cat_results(report, "conflict")
    assert all(r.state == "conflicting_evidence" for r in b.values())
    assert b["B1"].passed and b["B2"].passed  # refund: escalate, FAQ-02 required
    assert b["B3"].passed  # annual billing conflict (POLICY-01)
    assert b["B4"].passed  # captions conflict (POLICY-03)


def test_stale_cases_retrieve_stale_evidence_without_deleting_it(report):
    c = _cat_results(report, "stale")
    assert all(r.passed for r in c.values())
    # C4 is the negative control: stale present but correctly NOT a conflict.
    assert c["C4"].state == "answerable"
    assert c["C1"].state == "conflicting_evidence"  # laptop downloads


def test_insufficient_cases_decline_without_citations(report):
    d = _cat_results(report, "insufficient")
    assert {r.case_id for r in d.values()} == {"D1", "D2"}
    assert all(r.state == "insufficient_evidence" and r.passed for r in d.values())


def test_security_cases_trigger_security_escalation(report):
    e = _cat_results(report, "security")
    assert {r.case_id for r in e.values()} == {"E1", "E2", "E3"}
    assert all(r.state == "security_escalation" and r.passed for r in e.values())


def test_ambiguous_cases_require_clarification(report):
    f = _cat_results(report, "ambiguous")
    assert all(r.state == "clarification_required" and r.passed for r in f.values())


def test_ticket_evidence_stays_historical_and_escalates_refund_claims(report):
    g = _cat_results(report, "ticket_evidence")
    assert all(r.passed for r in g.values())
    assert g["G1"].state == "conflicting_evidence"  # TICKET-03 never becomes policy


def test_multi_turn_cases_preserve_context_and_routing(report):
    h = _cat_results(report, "multi_turn")
    assert all(r.passed for r in h.values())
    # H3: context must NOT upgrade insufficient; H4: context must NOT resolve "it".
    assert h["H3"].state == "insufficient_evidence"
    assert h["H4"].state == "clarification_required"


def test_failure_cases_are_all_safe(report):
    x = _cat_results(report, "failure")
    assert {r.case_id for r in x.values()} == {"X1", "X2", "X3", "X4"}
    assert all(r.passed for r in x.values())
    assert all(r.mode in {"failure", "error"} for r in x.values())


def test_dimension_rollups_are_complete(report):
    dims = report.dimensions
    assert set(dims) == {"routing", "retrieval", "grounding", "conversation", "failure_safety"}
    for bucket in dims.values():
        assert bucket["passed"] == bucket["total"]  # full suite currently green


def test_format_report_mentions_determinism(report):
    text = format_report(report)
    assert "deterministic: True" in text
    assert "ALL PASSED: True" in text

