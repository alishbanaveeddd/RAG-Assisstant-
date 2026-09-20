"""M6 — minimal CLI to exercise the **real** end-to-end pipeline.

Flow per turn (nothing is bypassed):

    conversation (M5) -> hybrid retrieval (M2) -> evidence assessment (M3)
    -> grounded generation (M4) -> conversation update -> printed answer

Provider choice is explicit:

* ``--provider real`` (default) uses M4's configured Groq provider and therefore
  requires ``GROQ_API_KEY``. If it is missing the CLI stops with a clear message;
  it never silently swaps in fake generation.
* ``--provider fake`` uses M4's :class:`~learnforge.generation.FakeProvider`
  (offline, deterministic, no LLM call) and says so in the banner.

Usage::

    python -m learnforge.cli --provider fake --once "What is the refund policy?"
    python -m learnforge.cli --provider real --show-details
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from learnforge.assistant import Assistant
from learnforge.conversation import EmptyMessageError
from learnforge.retrieval import DEFAULT_TOP_K

EXIT_OK = 0
EXIT_NO_API_KEY = 3

#: Fake-provider failure injection (offline demos of the M4 failure path).
_FAKE_ERRORS = {
    "timeout": "ProviderTimeoutError",
    "rate_limit": "ProviderRateLimitError",
    "connection": "ProviderConnectionError",
    "server": "ProviderServerError",
    "auth": "ProviderAuthError",
    "malformed": "MalformedResponseError",
    "missing_key": "MissingAPIKeyError",
}

_BANNER_COMMANDS = "commands : :reset   :details   :quit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m learnforge.cli",
        description="LearnForge support assistant (M6 end-to-end pipeline).",
    )
    parser.add_argument(
        "--provider",
        choices=("real", "fake"),
        default="real",
        help="real = Groq (needs GROQ_API_KEY); fake = offline deterministic test double",
    )
    parser.add_argument("--once", metavar="QUERY", help="run a single query and exit")
    parser.add_argument("--k", type=int, default=None, help="retrieval depth (default 5)")
    parser.add_argument("--records", default=None, help="normalized records JSON")
    parser.add_argument("--embeddings", default=None, help="local embedding store JSON")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="print retrieval/evidence/routing detail for each turn",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON per turn")
    parser.add_argument(
        "--fake-response",
        action="append",
        metavar="TEXT",
        help="canned fake-provider reply (repeatable, one per turn)",
    )
    parser.add_argument(
        "--fake-error",
        choices=tuple(_FAKE_ERRORS),
        help="make the fake provider fail with this M4 failure type (offline)",
    )
    return parser


def _resolve_error(name: str) -> Any:
    from learnforge import generation as gen

    return getattr(gen, _FAKE_ERRORS[name])


def build_provider(args: argparse.Namespace) -> tuple[Any, str, str]:
    """Return ``(provider, label, model)``. Never prints or stores the API key."""
    if args.provider == "fake":
        from learnforge.generation import FakeProvider

        error = None
        if args.fake_error:
            error = _resolve_error(args.fake_error)(f"injected {args.fake_error}")
        provider = FakeProvider(
            responses=list(args.fake_response or []),
            error=error,
            default_response="(offline fake-provider response: no LLM was called)",
        )
        label = "fake (offline deterministic test double; no LLM call)"
        if args.fake_error:
            label += f" [injecting {args.fake_error}]"
        return provider, label, provider.model

    from learnforge.generation import MissingAPIKeyError, default_provider

    try:
        provider = default_provider()
    except MissingAPIKeyError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Re-run with --provider fake for an offline demonstration, "
            "or set GROQ_API_KEY to use the real provider."
        ) from exc
    return provider, f"real ({provider.name})", provider.model


def _format_details(result: Any) -> str:
    """Human-readable retrieval/evidence/routing detail for a turn."""
    lines = [
        f"  prepared query : {result.prepared_query}",
        f"  state/routing  : {result.state} / {result.routing}"
        f" (confidence: {result.confidence})",
        f"  retrieved      : {', '.join(result.retrieved_ids) or '-'}",
        f"  evidence       : {', '.join(result.evidence_ids) or '-'}",
        f"  citations used : {', '.join(result.citations_used) or '-'}",
    ]
    if result.conflict_topics:
        lines.append(f"  conflicts      : {', '.join(result.conflict_topics)}")
    if result.security_triggered:
        lines.append("  security       : triggered")
    if result.escalation_required:
        lines.append("  escalation     : required")
    if result.failure_type:
        lines.append(f"  failure        : {result.failure_type} ({result.failure_detail})")
    return "\n".join(lines)


def run_once(args: argparse.Namespace) -> int:
    """Run a single query through the real pipeline and print the answer."""
    provider, label, model = build_provider(args)
    print(f"provider : {label}")
    bot = Assistant(
        provider=provider,
        top_k=args.k if args.k is not None else DEFAULT_TOP_K,
        records_path=args.records,
        embeddings_path=args.embeddings,
    )
    result = bot.handle_message(args.once)
    if args.show_details:
        print(_format_details(result))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"assistant: {result.answer}")
    return EXIT_OK if result.succeeded else EXIT_NO_API_KEY


def run_session(args: argparse.Namespace) -> int:
    """Interactive multi-turn session over one persistent Conversation."""
    provider, label, model = build_provider(args)
    print(f"LearnForge assistant — provider: {label}")
    print(_BANNER_COMMANDS)
    show_details = args.show_details
    bot = Assistant(
        provider=provider,
        top_k=args.k if args.k is not None else DEFAULT_TOP_K,
        records_path=args.records,
        embeddings_path=args.embeddings,
    )
    while True:
        try:
            message = input("\nyou     > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == ":quit":
            break
        if message == ":reset":
            bot.reset()
            print("(conversation reset)")
            continue
        if message == ":details":
            show_details = not show_details
            print(f"(details {'on' if show_details else 'off'})")
            continue
        try:
            result = bot.handle_message(message)
        except EmptyMessageError:
            print("assistant: please enter a message.")
            continue
        if show_details:
            print(_format_details(result))
        print(f"assistant> {result.answer}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Chooses exactly one provider and one run mode."""
    args = build_parser().parse_args(argv)
    if args.once:
        return run_once(args)
    return run_session(args)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

