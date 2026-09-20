"""LearnForge knowledge-base record schema.

This module implements the record schema proposed in ``docs/architecture.md``
(M0, section 16) and adds only the fields that milestone M1 explicitly asks the
schema to preserve (a stable citation key plus escalation / ambiguity /
contradiction-support metadata).

Design rule for M1: ingestion **normalizes** records but never *resolves* corpus
contradictions. Stale flags and topic hints are descriptive metadata that later
retrieval / evidence logic (M2+) uses to decide how to answer.

M0 schema (verbatim shape)::

    {
      "source_id": "POLICY-02",
      "source_type": "faq" | "policy" | "ticket",
      "title": "...",
      "chunk_text": "<verbatim source text>",
      "metadata": {
        "freshness": "Effective date: January 2026",   # or "undated"
        "is_stale": true,
        "stale_reason": "...",
        "authority": 2,                                 # policy=>3, faq=>2, ticket=>1
        "ticket_status": "..."                          # tickets only
      },
      "vector": [...]                                   # added in a later milestone
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "m1"

SOURCE_TYPES = ("faq", "policy", "ticket")

# M0 section 16: "policy=>3, faq=>2, ticket=>1".
AUTHORITY_BY_TYPE = {"policy": 3, "faq": 2, "ticket": 1}

# M0 section 16: freshness sentinel used when a record carries no date line.
UNDATED = "undated"

# source_id format and the source type implied by each id prefix.
_ID_RE = re.compile(r"^(FAQ|POLICY|TICKET)-(\d{2})$")
_PREFIX_TO_TYPE = {"FAQ": "faq", "POLICY": "policy", "TICKET": "ticket"}

#: Metadata keys defined by the M0 schema (section 16).
M0_METADATA_KEYS = (
    "freshness",
    "is_stale",
    "stale_reason",
    "authority",
    "ticket_status",
)

#: Additive metadata keys required by milestone M1 (documented in docs/data-schema.md).
M1_METADATA_KEYS = (
    "escalated",
    "unresolved",
    "ambiguity_flags",
    "contradiction_topics",
)


def source_type_for_id(source_id: str) -> Optional[str]:
    """Return the source type implied by a ``source_id`` prefix, or ``None``."""
    match = _ID_RE.match(source_id or "")
    if not match:
        return None
    return _PREFIX_TO_TYPE[match.group(1)]


@dataclass
class Record:
    """A single normalized knowledge-base record (one record == one chunk)."""

    source_id: str
    source_type: str
    title: str
    chunk_text: str
    citation_key: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Optional[list[Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` with a deterministic key order."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "chunk_text": self.chunk_text,
            "citation_key": self.citation_key,
            "metadata": dict(self.metadata),
            "vector": self.vector,
        }


def build_metadata(
    *,
    source_type: str,
    freshness: str,
    is_stale: bool,
    stale_reason: Optional[str],
    ticket_status: Optional[str] = None,
    escalated: bool = False,
    unresolved: bool = False,
    ambiguity_flags: Optional[list[str]] = None,
    contradiction_topics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Assemble a metadata dict in the documented key order."""
    return {
        "freshness": freshness,
        "is_stale": is_stale,
        "stale_reason": stale_reason,
        "authority": AUTHORITY_BY_TYPE[source_type],
        "ticket_status": ticket_status,
        "escalated": escalated,
        "unresolved": unresolved,
        "ambiguity_flags": sorted(ambiguity_flags or []),
        "contradiction_topics": sorted(contradiction_topics or []),
    }


def validate_record(record: Record) -> list[str]:
    """Return a list of human-readable validation problems (empty == valid).

    Validation is deterministic and never mutates the record.
    """
    issues: list[str] = []

    # source_id + implied source type.
    implied = source_type_for_id(record.source_id)
    if implied is None:
        issues.append(
            f"source_id {record.source_id!r} does not match the required "
            "pattern '(FAQ|POLICY|TICKET)-NN'"
        )
    elif implied != record.source_type:
        issues.append(
            f"source_id {record.source_id!r} implies source_type {implied!r} "
            f"but record declares {record.source_type!r}"
        )
    if record.source_type not in SOURCE_TYPES:
        issues.append(f"unknown source_type {record.source_type!r}")

    # Required text fields.
    if not (record.title or "").strip():
        issues.append("title is required and must not be empty")
    if not (record.chunk_text or "").strip():
        issues.append("chunk_text is required and must not be empty")

    # Citation key must be present and stable.
    if not (record.citation_key or "").strip():
        issues.append("citation_key is required and must not be empty")
    elif record.citation_key != record.source_id:
        issues.append(
            f"citation_key {record.citation_key!r} must equal source_id "
            f"{record.source_id!r}"
        )

    # Metadata presence + types.
    meta = record.metadata or {}
    for key in M0_METADATA_KEYS:
        if key not in meta:
            issues.append(f"metadata.{key} is required")

    freshness = meta.get("freshness")
    if not isinstance(freshness, str) or not freshness.strip():
        issues.append("metadata.freshness must be a non-empty string (use 'undated')")

    is_stale = meta.get("is_stale")
    if not isinstance(is_stale, bool):
        issues.append("metadata.is_stale must be a boolean")

    authority = meta.get("authority")
    expected_authority = AUTHORITY_BY_TYPE.get(record.source_type)
    if authority != expected_authority:
        issues.append(
            f"metadata.authority must be {expected_authority} for source_type "
            f"{record.source_type!r} (got {authority!r})"
        )

    stale_reason = meta.get("stale_reason")
    if is_stale is True and not (stale_reason or "").strip():
        issues.append("metadata.stale_reason is required when metadata.is_stale is true")
    if is_stale is False and stale_reason:
        issues.append("metadata.stale_reason must be null when metadata.is_stale is false")

    # Ticket-only fields.
    ticket_status = meta.get("ticket_status")
    if record.source_type == "ticket":
        if not (ticket_status or "").strip():
            issues.append("metadata.ticket_status is required for ticket records")
        if not isinstance(meta.get("escalated"), bool):
            issues.append("metadata.escalated must be a boolean")
        if not isinstance(meta.get("unresolved"), bool):
            issues.append("metadata.unresolved must be a boolean")
    else:
        if ticket_status is not None:
            issues.append("metadata.ticket_status is only valid for ticket records")
        if meta.get("escalated") or meta.get("unresolved"):
            issues.append(
                "metadata.escalated/unresolved are only valid for ticket records"
            )

    for list_key in ("ambiguity_flags", "contradiction_topics"):
        value = meta.get(list_key)
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            issues.append(f"metadata.{list_key} must be a list of strings")

    # The vector slot stays empty in M1 (embeddings arrive in a later milestone).
    if record.vector is not None:
        issues.append("vector must be null in milestone M1")

    return issues
