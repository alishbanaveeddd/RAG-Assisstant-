"""Deterministic ingestion / normalization for the LearnForge knowledge base.

Reads the three supplied Markdown files (``faqs.md``, ``policies.md``,
``tickets.md``), splits them into per-record chunks (one record == one chunk,
per M0 section 16), extracts the M0 metadata fields plus the additive M1
metadata fields, validates every record, and produces a deterministic JSON
document for later retrieval.

The supplied source files are opened read-only and are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional

from learnforge.schema import (
    SCHEMA_VERSION,
    UNDATED,
    Record,
    build_metadata,
    validate_record,
)

# --------------------------------------------------------------------------- #
# Source configuration
# --------------------------------------------------------------------------- #

#: Ordered list of (filename, expected source type). Order fixes output order.
SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("faqs.md", "faq"),
    ("policies.md", "policy"),
    ("tickets.md", "ticket"),
)

DEFAULT_BASE_DIR = os.path.join(
    "learnforge-knowledge-base-data", "learnforge-knowledge-base"
)
DEFAULT_OUT_PATH = os.path.join("data", "processed", "kb_records.json")

# --------------------------------------------------------------------------- #
# Parsing patterns
# --------------------------------------------------------------------------- #

#: Record heading, e.g. ``# FAQ-01 — How do I access a course ...``
HEADING_RE = re.compile(
    r"^#\s*(?P<prefix>FAQ|POLICY|TICKET)-(?P<num>\d{2})\s*"
    r"[\u2014\u2013-]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

#: Trailing cross-file section labels appended to the supplied files, e.g.
#: ``SECTION 2 — POLICY / HELP-CENTER DOCUMENT EXCERPTS``.
SECTION_RE = re.compile(r"^SECTION\s+\d+\b", re.IGNORECASE)

SEPARATOR = "---"

#: Ticket resolution line, e.g. ``STATUS: Escalated due to policy ambiguity.``
STATUS_RE = re.compile(r"^STATUS:\s*(?P<status>.+?)\s*$", re.MULTILINE)

#: Explicit freshness / version lines found in policy excerpts.
#: Longest labels first so "Effective date" wins over "Effective".
FRESHNESS_RE = re.compile(
    r"^(?P<label>Last reviewed|Last updated|Effective date|Reviewed|Updated|Effective)"
    r"\s*:\s*(?P<value>[^\n]+?)\s*$",
    re.MULTILINE,
)

#: Phrases that reference a *prior document/version* of a source (the corpus
#: always phrases stale notes this way). A bare "older"/"previous" is NOT enough
#: — that would false-positive on e.g. "older progress information" (FAQ-03),
#: "older courses" (FAQ-06) or "an older laptop" (FAQ-08).
STALE_PHRASE_RE = re.compile(
    r"\b(?:older|previous|archived)\s+(?:\d{4}\s+)?"
    r"(?:versions?|documentation|documents?|docs?|articles?|guides?|handbook|"
    r"help(?:-center)?\s+(?:documentation|articles?)|"
    r"mobile help article|internal billing document|"
    r"instructor guide|instructor handbook)\b",
    re.IGNORECASE,
)

#: Corrective clause that usually accompanies a stale note.
CORRECTIVE_RE = re.compile(
    r"\b(outdated|obsolete|retired|no longer|has been removed|has been replaced|"
    r"should not be treated|may no longer apply|is no longer|removed)\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: Lightweight topic hints used later for contradiction detection (M0 section 11)
#: and retrieval (M0 section 16). These tag *topics*; they do not resolve
#: contradictions.
TOPIC_PATTERNS: dict[str, str] = {
    "refund_window": r"refund|money-back guarantee|refund period",
    "annual_billing": r"\bannual\b|billed monthly",
    "family_plan": r"family plan|five-user",
    "browser_support": r"\bbrowser|internet explorer",
    "offline_downloads": r"\boffline\b|download",
    "progress_sync": r"\bprogress\b|synchron",
    "captions_accessibility": r"caption|transcript|accessib",
    "payment_data_collection": (
        r"card number|first six|last four|\bCVV\b|\bPIN\b|banking password|"
        r"authentication code"
    ),
}
_TOPIC_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in TOPIC_PATTERNS.items()}

#: Ambiguity / escalation indicators for the specific tickets M0 section 6.5
#: identified as ambiguous or escalation-worthy. Grounded in the corpus text and
#: M0's analysis; recorded as metadata so M2+ evidence logic can act on it.
KNOWN_AMBIGUITY_FLAGS: dict[str, list[str]] = {
    "TICKET-03": ["promotional_terms_may_differ"],
    "TICKET-07": ["ambiguous_intent_requires_clarification"],
    "TICKET-08": ["policy_wording_ambiguity"],
    "TICKET-10": ["identity_verification_required"],
    "TICKET-11": ["purchase_type_conflict"],
    "TICKET-13": ["enrollment_transfer_requires_review"],
    "TICKET-15": ["outdated_documentation_reliance"],
}


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def normalize_newlines(text: str) -> str:
    """Normalize CRLF/CR to LF so parsing and comparisons are platform-stable."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def squash_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def clean_segment(segment: str) -> str:
    """Trim a raw record segment to its verbatim content.

    Drops a trailing cross-file ``SECTION n`` label and trailing ``---`` /
    blank lines. The result is still a contiguous substring of the source text.
    """
    lines = segment.split("\n")
    for index, line in enumerate(lines):
        if SECTION_RE.match(line.strip()):
            lines = lines[:index]
            break
    while lines and lines[-1].strip() in ("", SEPARATOR):
        lines.pop()
    return "\n".join(lines).strip()


def split_records(text: str) -> list[dict[str, str]]:
    """Split a normalized Markdown file into verbatim per-record dicts.

    Segmentation is heading-based (not separator-based), so a missing or extra
    ``---`` cannot silently merge or drop a record.
    """
    matches = list(HEADING_RE.finditer(text))
    parsed: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        prefix = match.group("prefix")
        parsed.append(
            {
                "source_id": f"{prefix}-{match.group('num')}",
                "source_type": {"FAQ": "faq", "POLICY": "policy", "TICKET": "ticket"}[
                    prefix
                ],
                "title": match.group("title").strip(),
                "chunk_text": clean_segment(text[start:end]),
            }
        )
    return parsed


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #

def extract_freshness(chunk_text: str) -> str:
    """Return a canonical freshness string, or ``"undated"`` when absent.

    No date is ever invented for a record that has no explicit date line.
    """
    match = FRESHNESS_RE.search(chunk_text)
    if not match:
        return UNDATED
    label = squash_whitespace(match.group("label"))
    value = squash_whitespace(match.group("value")).rstrip(".").strip()
    return f"{label}: {value}"


def detect_stale(chunk_text: str) -> tuple[bool, Optional[str]]:
    """Detect an explicit stale/outdated note about the record's own subject.

    Returns ``(is_stale, stale_reason)``. ``stale_reason`` is verbatim evidence
    from the source (the marker sentence plus a following corrective sentence
    when present) — never a paraphrase.
    """
    sentences = SENTENCE_SPLIT_RE.split(chunk_text)
    for index, sentence in enumerate(sentences):
        if STALE_PHRASE_RE.search(sentence):
            reason = squash_whitespace(sentence)
            if index + 1 < len(sentences):
                nxt = sentences[index + 1]
                if CORRECTIVE_RE.search(nxt):
                    reason = f"{reason} {squash_whitespace(nxt)}"
            return True, reason
    return False, None


def extract_ticket_status(chunk_text: str) -> Optional[str]:
    """Return the ticket's ``STATUS:`` value (without the trailing period)."""
    match = STATUS_RE.search(chunk_text)
    if not match:
        return None
    return squash_whitespace(match.group("status")).rstrip(".").strip()


def ticket_resolution_flags(status: str) -> tuple[bool, bool]:
    """Return ``(escalated, unresolved)`` derived from a ticket status string."""
    lowered = (status or "").lower()
    escalated = "escalat" in lowered
    unresolved = status.strip().rstrip(".").lower() != "resolved"
    return escalated, unresolved


def derive_ambiguity_flags(source_id: str, status: Optional[str]) -> list[str]:
    """Return ambiguity/escalation indicators for a ticket."""
    flags = list(KNOWN_AMBIGUITY_FLAGS.get(source_id, []))
    if status and "ambiguity" in status.lower() and "policy_wording_ambiguity" not in flags:
        flags.append("policy_wording_ambiguity")
    return sorted(set(flags))


def detect_topics(chunk_text: str) -> list[str]:
    """Return the sorted topic hints present in a record's text."""
    return sorted(name for name, regex in _TOPIC_RE.items() if regex.search(chunk_text))


def build_record(parsed: dict[str, str]) -> Record:
    """Normalize a parsed record dict into a :class:`Record`."""
    source_id = parsed["source_id"]
    source_type = parsed["source_type"]
    chunk_text = parsed["chunk_text"]

    ticket_status: Optional[str] = None
    escalated = False
    unresolved = False
    ambiguity_flags: list[str] = []

    if source_type == "ticket":
        ticket_status = extract_ticket_status(chunk_text)
        escalated, unresolved = ticket_resolution_flags(ticket_status or "")
        ambiguity_flags = derive_ambiguity_flags(source_id, ticket_status)
        # Ticket transcripts are historical evidence, never a stale *source* of
        # current policy (M0 section 9), so staleness is not inferred for them.
        freshness = UNDATED
        is_stale, stale_reason = False, None
    else:
        freshness = extract_freshness(chunk_text)
        is_stale, stale_reason = detect_stale(chunk_text)

    metadata = build_metadata(
        source_type=source_type,
        freshness=freshness,
        is_stale=is_stale,
        stale_reason=stale_reason,
        ticket_status=ticket_status,
        escalated=escalated,
        unresolved=unresolved,
        ambiguity_flags=ambiguity_flags,
        contradiction_topics=detect_topics(chunk_text),
    )

    return Record(
        source_id=source_id,
        source_type=source_type,
        title=parsed["title"],
        chunk_text=chunk_text,
        citation_key=source_id,
        metadata=metadata,
        vector=None,
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class IngestResult:
    """Outcome of a full ingestion run."""

    records: list[Record]
    errors: list[dict[str, Any]]
    source_hashes: dict[str, str]

    @property
    def counts(self) -> dict[str, int]:
        counts = {"faq": 0, "policy": 0, "ticket": 0}
        for record in self.records:
            counts[record.source_type] += 1
        counts["total"] = len(self.records)
        return counts


class IngestionError(Exception):
    """Raised in strict mode when one or more records fail validation."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        detail = "; ".join(
            f"{e['file']}:{e['source_id']} -> {', '.join(e['issues'])}" for e in errors
        )
        super().__init__(f"{len(errors)} record(s) failed validation: {detail}")


def read_source_file(path: str) -> tuple[str, str]:
    """Read a source file read-only, returning ``(normalized_text, sha256)``."""
    with open(path, "rb") as handle:
        raw = handle.read()
    sha = hashlib.sha256(raw).hexdigest()
    return normalize_newlines(raw.decode("utf-8")), sha


def ingest_all(
    base_dir: str = DEFAULT_BASE_DIR, *, strict: bool = True
) -> IngestResult:
    """Ingest every source file into validated, normalized records.

    ``strict=True`` raises :class:`IngestionError` if any record is malformed.
    ``strict=False`` returns the errors alongside the successfully built records.
    """
    records: list[Record] = []
    errors: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    for filename, expected_type in SOURCE_FILES:
        path = os.path.join(base_dir, filename)
        text, sha = read_source_file(path)
        source_hashes[filename] = sha

        parsed = split_records(text)
        if not parsed:
            errors.append(
                {
                    "file": filename,
                    "source_id": "-",
                    "issues": ["no records matched the heading pattern"],
                }
            )
        for item in parsed:
            if item["source_type"] != expected_type:
                errors.append(
                    {
                        "file": filename,
                        "source_id": item["source_id"],
                        "issues": [
                            f"source_type {item['source_type']!r} does not match "
                            f"expected {expected_type!r} for {filename}"
                        ],
                    }
                )
            record = build_record(item)
            for issue in validate_record(record):
                errors.append(
                    {"file": filename, "source_id": record.source_id, "issues": [issue]}
                )
            records.append(record)

    # Duplicate source_id detection (deterministic, across all files).
    seen: dict[str, int] = {}
    for record in records:
        seen[record.source_id] = seen.get(record.source_id, 0) + 1
    for source_id, occurrences in seen.items():
        if occurrences > 1:
            errors.append(
                {
                    "file": "*",
                    "source_id": source_id,
                    "issues": [f"duplicate source_id appears {occurrences} times"],
                }
            )

    if strict and errors:
        raise IngestionError(errors)
    return IngestResult(records=records, errors=errors, source_hashes=source_hashes)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def build_document(result: IngestResult) -> dict[str, Any]:
    """Build the deterministic output document written for later milestones."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_files": dict(result.source_hashes),
        "counts": result.counts,
        "records": [record.to_dict() for record in result.records],
    }


def write_json(document: dict[str, Any], out_path: str) -> str:
    """Write the output document to ``out_path`` (creating directories)."""
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return out_path


def format_report(result: IngestResult) -> str:
    """Human-readable manual-verification report."""
    counts = result.counts
    lines: list[str] = []
    lines.append("Ingestion summary")
    lines.append(f"  faq:    {counts['faq']}")
    lines.append(f"  policy: {counts['policy']}")
    lines.append(f"  ticket: {counts['ticket']}")
    lines.append(f"  total:  {counts['total']}")
    lines.append("")
    lines.append("Source file sha256")
    for name, sha in result.source_hashes.items():
        lines.append(f"  {name}: {sha}")

    by_id = {record.source_id: record for record in result.records}

    lines.append("")
    lines.append("Representative records")
    for source_id in ("FAQ-01", "POLICY-01", "TICKET-01"):
        record = by_id[source_id]
        meta = record.metadata
        lines.append(
            f"  {source_id} [{record.source_type}] {record.title!r} | "
            f"freshness={meta['freshness']!r} authority={meta['authority']} "
            f"is_stale={meta['is_stale']} ticket_status={meta['ticket_status']!r}"
        )

    lines.append("")
    lines.append("Stale records (explicit outdated notes preserved, not resolved)")
    for record in result.records:
        if record.metadata["is_stale"]:
            lines.append(f"  {record.source_id}: {record.metadata['stale_reason']}")

    lines.append("")
    lines.append("Unresolved / escalated tickets (historical evidence, not policy)")
    for record in result.records:
        meta = record.metadata
        if record.source_type == "ticket" and (meta["unresolved"] or meta["escalated"]):
            flags = ",".join(meta["ambiguity_flags"]) or "-"
            lines.append(
                f"  {record.source_id}: status={meta['ticket_status']!r} "
                f"escalated={meta['escalated']} ambiguity_flags=[{flags}]"
            )

    lines.append("")
    lines.append("Payment-security / PII related records")
    for record in result.records:
        if "payment_data_collection" in record.metadata["contradiction_topics"]:
            lines.append(f"  {record.source_id} [{record.source_type}] {record.title!r}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point: ``python -m learnforge.ingest``."""
    parser = argparse.ArgumentParser(
        prog="learnforge.ingest",
        description="Deterministically ingest the LearnForge KB into JSON records.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE_DIR, help="KB source directory")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="output JSON path")
    parser.add_argument(
        "--no-strict", action="store_true", help="report malformed records instead of failing"
    )
    parser.add_argument("--no-write", action="store_true", help="do not write the JSON file")
    args = parser.parse_args(argv)

    result = ingest_all(args.base, strict=not args.no_strict)

    if result.errors:
        print("Malformed records detected:", file=sys.stderr)
        for error in result.errors:
            print(
                f"  {error['file']}:{error['source_id']} -> {'; '.join(error['issues'])}",
                file=sys.stderr,
            )

    if not args.no_write:
        out_path = write_json(build_document(result), args.out)
        print(f"Wrote {out_path}")

    print(format_report(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())