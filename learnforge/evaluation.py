"""M7 — deterministic evaluation benchmark for the LearnForge assistant.

The benchmark is **behavioral**: it asserts what the pipeline does (routing
state, evidence surfaced, citation safety, escalation, conversation handling,
failure safety), never what the generated wording says. It runs the REAL M2
retriever and M3 assessor; the only stub is M4's ``FakeProvider`` (offline,
deterministic, no API key), so no metric here measures free-form answer
wording quality — that requires human review or a separately validated
LLM-judge evaluation (see docs/evaluation.md).

Usage::

    python -m learnforge.evaluation                 # fake provider, human-readable
    python -m learnforge.evaluation --json-out out.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from learnforge.assistant import Assistant, AssistantResult
from learnforge.conversation import new_conversation
from learnforge.generation import FakeProvider, ProviderTimeoutError
from learnforge.retrieval import Retriever

DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "cases.json"

_CATEGORIES = {
    "answerable", "conflict", "stale", "insufficient", "security",
    "ambiguous", "ticket_evidence", "multi_turn", "failure",
}
_INJECTIONS = {
    "retrieval_error", "assessment_error", "generation_error", "provider_timeout",
}

#: Fake-provider canned reply — deliberately contains no [ID] citation markers.
_FAKE_ANSWER = "(offline fake-provider answer)"


# --------------------------------------------------------------------------- #
# Fault injection (failure cases only; deterministic, no network)
# --------------------------------------------------------------------------- #
class _FaultRetriever:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def search(self, query: str, k: int = 5):
        raise self._error


class _FaultAssessor:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __call__(self, query, results):
        raise self._error


class _FaultGenerator:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __call__(self, query, assessment, provider=None, conversation_block=None):
        raise self._error


_shared_retriever: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    """Load the real M2 retriever once per process (embeddings are local)."""
    global _shared_retriever
    if _shared_retriever is None:
        _shared_retriever = Retriever.from_store()
    return _shared_retriever


def build_assistant(
    inject: Optional[str] = None,
    provider: Any = None,
    retriever: Any = None,
) -> Assistant:
    """Build a real-pipeline ``Assistant``; ``inject`` selects a failure mode."""
    if inject is None:
        return Assistant(
            retriever=retriever if retriever is not None else _get_retriever(),
            provider=provider if provider is not None else FakeProvider(default_response=_FAKE_ANSWER),
        )
    if inject not in _INJECTIONS:
        raise ValueError(f"unknown injection {inject!r}; expected one of {sorted(_INJECTIONS)}")
    if inject == "provider_timeout":
        return Assistant(
            retriever=retriever if retriever is not None else _get_retriever(),
            provider=FakeProvider(
                default_response=_FAKE_ANSWER,
                error=ProviderTimeoutError("injected timeout"),
            ),
        )
    if inject == "retrieval_error":
        return Assistant(
            retriever=_FaultRetriever(RuntimeError("injected retrieval failure")),
            provider=FakeProvider(default_response=_FAKE_ANSWER),
        )
    if inject == "assessment_error":
        return Assistant(
            retriever=retriever if retriever is not None else _get_retriever(),
            provider=FakeProvider(default_response=_FAKE_ANSWER),
            assessor=_FaultAssessor(ValueError("injected assessment failure")),
        )
    return Assistant(
        retriever=retriever if retriever is not None else _get_retriever(),
        provider=FakeProvider(default_response=_FAKE_ANSWER),
        generator=_FaultGenerator(RuntimeError("injected generation failure")),
    )


# --------------------------------------------------------------------------- #
# Case execution + checking
# --------------------------------------------------------------------------- #
def _run_pipeline_case(case: dict, factory: Callable[..., Assistant]) -> AssistantResult:
    bot = factory()
    for prior in case.get("conversation", []):
        prior_result = bot.handle_message(prior)
        if prior_result.mode == "error":
            raise RuntimeError(
                f"prior turn failed for {case['id']}: {prior_result.failure_detail}"
            )
    return bot.handle_message(case["query"])


def _check_case(case: dict, result: AssistantResult) -> list[dict]:
    """Return a list of ``{dimension, name, ok, detail}`` check records."""
    exp = case["expected"]
    checks: list[dict] = []

    def add(dimension: str, name: str, ok: bool, detail: str = "") -> None:
        checks.append({"dimension": dimension, "name": name, "ok": ok, "detail": detail})

    # --- routing (M3 state / routing / flags) ------------------------------ #
    if "state" in exp:
        add("routing", "state", result.state == exp["state"],
            f"expected {exp['state']}, got {result.state}")
    if "routing" in exp:
        add("routing", "routing", result.routing == exp["routing"],
            f"expected {exp['routing']}, got {result.routing}")
    if exp.get("security_triggered") is not None:
        add("routing", "security_flag",
            result.security_triggered == exp["security_triggered"],
            f"expected {exp['security_triggered']}, got {result.security_triggered}")
    if exp.get("escalation_required") is not None:
        add("routing", "escalation_flag",
            result.escalation_required == exp["escalation_required"],
            f"expected {exp['escalation_required']}, got {result.escalation_required}")
    if "conflict_topics" in exp:
        missing = set(exp["conflict_topics"]) - set(result.conflict_topics)
        add("routing", "conflict_topics", not missing,
            f"missing conflict topics: {sorted(missing)}" if missing else "")

    # --- retrieval (relevant / required evidence surfaced) ----------------- #
    if "required_evidence_ids" in exp:
        missing = set(exp["required_evidence_ids"]) - set(result.evidence_ids)
        add("retrieval", "required_ids", not missing,
            f"missing evidence ids: {sorted(missing)}" if missing else "")
    if exp.get("stale_evidence_expected") is not None:
        stale_visible = any(item.is_stale for item in result.evidence_items())
        add("retrieval", "stale_visible",
            stale_visible == exp["stale_evidence_expected"],
            f"expected stale evidence present={exp['stale_evidence_expected']}")

    # --- grounding / unsupported-answer behavioral checks ------------------ #
    add("grounding", "citations_within_allowed",
        set(result.citations_used) <= set(result.allowed_citations),
        f"citations {result.citations_used} vs allowed {result.allowed_citations}")
    add("grounding", "no_invalid_citations", not result.invalid_citations,
        f"invalid citations: {result.invalid_citations}")
    if exp.get("decline"):
        add("grounding", "no_citations_on_decline", not result.citations_used,
            f"decline case cited {result.citations_used}")
    if "answer_allowed" in exp:
        ok = (result.mode == "llm") if exp["answer_allowed"] else result.mode in {"failure", "error"}
        add("grounding", "answer_mode", ok, f"mode={result.mode}")

    # --- conversation (context preservation without corruption) ------------ #
    if "prepared_contains" in exp:
        pq = result.prepared_query.lower()
        missing = [s for s in exp["prepared_contains"] if s.lower() not in pq]
        add("conversation", "prepared_query_context", not missing,
            f"prepared query missing: {missing!r}" if missing else "")
    if "conversation" in case:
        add("conversation", "history_recorded",
            result.turn_count >= 2 * len(case["conversation"]) + 2,
            f"turn_count={result.turn_count}")

    return checks


def _check_failure_case(case: dict, result: AssistantResult) -> list[dict]:
    """Failure cases: safe, structured, no fabricated answer, history untouched."""
    exp = case["expected"]
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"dimension": "failure_safety", "name": name, "ok": ok, "detail": detail})

    add("mode_is_failure_or_error", result.mode in {"failure", "error"}, f"mode={result.mode}")
    if "failure_type" in exp:
        add("failure_type", result.failure_type == exp["failure_type"],
            f"expected {exp['failure_type']}, got {result.failure_type}")
    add("no_answer_content", result.answer == "" or result.mode == "failure",
        f"answer={result.answer!r}")
    add("conversation_not_updated", result.conversation_updated is False,
        f"conversation_updated={result.conversation_updated}")
    return checks


# --------------------------------------------------------------------------- #
# Benchmark runner + report
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    passed: bool
    checks: list[dict]
    state: Optional[str] = None
    mode: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.case_id,
            "category": self.category,
            "query": self.query,
            "passed": self.passed,
            "state": self.state,
            "mode": self.mode,
            "checks": self.checks,
        }


@dataclass
class EvalReport:
    results: list[CaseResult]
    dimensions: dict
    by_category: dict
    all_passed: bool
    deterministic: bool

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "deterministic": self.deterministic,
            "case_count": len(self.results),
            "passed_count": sum(1 for r in self.results if r.passed),
            "dimensions": self.dimensions,
            "by_category": self.by_category,
            "results": [r.to_dict() for r in self.results],
        }


def load_cases(path: Path = DEFAULT_CASES_PATH) -> dict:
    """Load and validate the benchmark cases (fail loudly on malformed input)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = list(data.get("cases", [])) + list(data.get("failure_cases", []))
    if not cases:
        raise ValueError(f"no cases found in {path}")
    seen: set[str] = set()
    for case in cases:
        for key in ("id", "category", "query", "expected"):
            if key not in case:
                raise ValueError(f"case missing {key!r}: {case!r}")
        if case["category"] not in _CATEGORIES:
            raise ValueError(f"case {case['id']}: unknown category {case['category']!r}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if "inject" in case and case["inject"] not in _INJECTIONS:
            raise ValueError(f"case {case['id']}: unknown inject {case['inject']!r}")
    return data


def _signature(result: CaseResult) -> tuple:
    """Comparable signature used for the determinism check."""
    return (
        result.case_id, result.passed, result.state, result.mode,
        tuple((c["name"], c["ok"]) for c in result.checks),
    )


def run_benchmark(
    cases_data: Optional[dict] = None,
    cases_path: Path = DEFAULT_CASES_PATH,
    factory: Callable[..., Assistant] = build_assistant,
    repeats: int = 2,
) -> EvalReport:
    """Run every case through the real pipeline and evaluate it.

    ``factory`` must return a fresh ``Assistant`` (each case gets its own
    conversation, so sessions stay isolated). ``repeats`` > 1 re-runs the whole
    benchmark and compares per-case signatures for determinism.
    """
    data = cases_data if cases_data is not None else load_cases(cases_path)
    all_cases = list(data.get("cases", [])) + list(data.get("failure_cases", []))

    results: list[CaseResult] = []
    for case in all_cases:
        is_failure = "inject" in case
        try:
            if is_failure:
                result = factory(inject=case["inject"]).handle_message(case["query"])
                checks = _check_failure_case(case, result)
            else:
                result = _run_pipeline_case(case, factory)
                checks = _check_case(case, result)
        except Exception as exc:  # report, never crash silently
            results.append(CaseResult(
                case_id=case["id"], category=case["category"], query=case["query"],
                passed=False,
                checks=[{
                    "dimension": "harness", "name": "no_exception", "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }],
            ))
            continue
        results.append(CaseResult(
            case_id=case["id"], category=case["category"], query=case["query"],
            passed=all(c["ok"] for c in checks), checks=checks,
            state=result.state, mode=result.mode,
        ))

    dimensions: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    for res in results:
        cat = by_category.setdefault(res.category, {"cases": 0, "passed": 0})
        cat["cases"] += 1
        cat["passed"] += int(res.passed)
        for chk in res.checks:
            bucket = dimensions.setdefault(chk["dimension"], {"passed": 0, "total": 0})
            bucket["total"] += 1
            bucket["passed"] += int(chk["ok"])

    deterministic = True
    if repeats > 1:
        first = [_signature(r) for r in results]
        for _ in range(repeats - 1):
            rerun = run_benchmark(data, cases_path, factory, repeats=1)
            if [_signature(r) for r in rerun.results] != first:
                deterministic = False
                break

    return EvalReport(
        results=results,
        dimensions=dimensions,
        by_category=by_category,
        all_passed=all(r.passed for r in results),
        deterministic=deterministic,
    )


def format_report(report: EvalReport) -> str:
    """Human-readable summary (per category + per dimension)."""
    lines = ["", "=== LearnForge deterministic benchmark (offline, FakeProvider) ==="]
    lines.append(f"{'category':<16}{'passed':>8}{'/total':>8}")
    for cat, bucket in report.by_category.items():
        lines.append(f"{cat:<16}{bucket['passed']:>8}/{bucket['cases']:<7}")
    lines.append("")
    lines.append(f"{'dimension':<20}{'passed':>8}{'/total':>8}")
    for dim, bucket in report.dimensions.items():
        lines.append(f"{dim:<20}{bucket['passed']:>8}/{bucket['total']:<7}")
    failed = [r for r in report.results if not r.passed]
    if failed:
        lines.append("")
        lines.append("FAILED cases:")
        for res in failed:
            lines.append(f"  {res.case_id} ({res.category}):")
            for chk in (c for c in res.checks if not c["ok"]):
                lines.append(f"    - {chk['dimension']}:{chk['name']} — {chk['detail']}")
    lines.append("")
    lines.append(
        f"cases: {len(report.results)}  "
        f"passed: {sum(1 for r in report.results if r.passed)}  "
        f"deterministic: {report.deterministic}  "
        f"ALL PASSED: {report.all_passed}"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m learnforge.evaluation",
        description="Run the deterministic LearnForge benchmark (offline by default).",
    )
    parser.add_argument("--cases", default=None, help="path to cases.json")
    parser.add_argument("--json-out", default=None, help="write the report as JSON")
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="benchmark repetitions for the determinism check (default 2)",
    )
    args = parser.parse_args(argv)

    path = Path(args.cases) if args.cases else DEFAULT_CASES_PATH
    report = run_benchmark(cases_path=path, repeats=max(1, args.repeats))
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"report written to {args.json_out}")
    return 0 if report.all_passed and report.deterministic else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
