"""Automated tests for the M1 schema + ingestion layer.

Run locally with either runner (no new test dependency is required)::

    python -m pytest tests -q
    python -m unittest discover -s tests -v

The suite is written against the standard library ``unittest`` (dependency-light,
matching M0's "locally runnable" constraint) and is also collected and run by
``pytest``, which is already installed in this environment.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from learnforge import ingest
from learnforge.schema import (
    AUTHORITY_BY_TYPE,
    Record,
    build_metadata,
    validate_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = str(REPO_ROOT / "learnforge-knowledge-base-data" / "learnforge-knowledge-base")

#: sha256 of the untouched supplied source files (recorded at M1 implementation
#: time). If any of these change, the ingestion inputs were modified.
EXPECTED_SOURCE_SHA256 = {
    "faqs.md": "6543690010504fd4b56a7271eb59b15179ffd1e3d098045691b798ef2a374e8b",
    "policies.md": "cd735fd5ae7fd581be3b872ae7dc3bc42ee962ad0d1927e5e4718f445d682c2b",
    "tickets.md": "79bf0117062c2cad0b7ca00ec5bd070927be7ee438553a350c7eed46bf9af19e",
}

EXPECTED_FAQ_IDS = [f"FAQ-{n:02d}" for n in range(1, 16)]
EXPECTED_POLICY_IDS = [f"POLICY-{n:02d}" for n in range(1, 11)]
EXPECTED_TICKET_IDS = [f"TICKET-{n:02d}" for n in range(1, 16)]

EXPECTED_POLICY_FRESHNESS = {
    "POLICY-01": "Last reviewed: February 2026",
    "POLICY-02": "Effective date: January 2026",
    "POLICY-03": "Updated: March 2026",
    "POLICY-04": "Last updated: May 2026",
    "POLICY-05": "undated",
    "POLICY-06": "Reviewed: April 2026",
    "POLICY-07": "Effective: December 2025",
    "POLICY-08": "undated",
    "POLICY-09": "Last reviewed: June 2026",
    "POLICY-10": "undated",
}

EXPECTED_STALE_IDS = {
    "FAQ-07",
    "POLICY-01",
    "POLICY-02",
    "POLICY-03",
    "POLICY-04",
    "POLICY-05",
    "POLICY-06",
    "POLICY-08",
    "POLICY-09",
    "POLICY-10",
}

EXPECTED_TICKET_STATUS = {
    "TICKET-01": "Resolved",
    "TICKET-02": "Pending monitoring",
    "TICKET-03": "Escalated",
    "TICKET-04": "Resolved",
    "TICKET-05": "Cancellation completed; refund review requested",
    "TICKET-06": "Awaiting transaction reference",
    "TICKET-07": "Refund request opened",
    "TICKET-08": "Escalated due to policy ambiguity",
    "TICKET-09": "Resolved",
    "TICKET-10": "Awaiting verification",
    "TICKET-11": "Escalated",
    "TICKET-12": "Alternative resource provided; accessibility issue reported",
    "TICKET-13": "Transfer request pending",
    "TICKET-14": "Pending bank authorization",
    "TICKET-15": (
        "Refund review requested due to potentially misleading outdated documentation"
    ),
}

EXPECTED_ESCALATED = {"TICKET-03", "TICKET-08", "TICKET-11"}
EXPECTED_RESOLVED = {"TICKET-01", "TICKET-04", "TICKET-09"}


def make_record(**overrides) -> Record:
    """Construct a valid FAQ record, applying overrides for negative tests."""
    defaults = dict(
        source_id="FAQ-99",
        source_type="faq",
        title="Example question?",
        chunk_text="# FAQ-99 — Example question?\n\nQUESTION:\nQ?\n\nANSWER:\nA.",
        citation_key="FAQ-99",
        metadata=build_metadata(
            source_type="faq",
            freshness="undated",
            is_stale=False,
            stale_reason=None,
        ),
        vector=None,
    )
    defaults.update(overrides)
    return Record(**defaults)


class CorpusTestCase(unittest.TestCase):
    """Base class that ingests the real supplied corpus once per test class."""

    @classmethod
    def setUpClass(cls):
        cls.result = ingest.ingest_all(BASE_DIR)
        cls.by_id = {r.source_id: r for r in cls.result.records}


class IngestionRecordTests(CorpusTestCase):
    """Shared tests over the real supplied corpus."""

    # -- coverage / counts -------------------------------------------------- #

    def test_all_expected_records_ingested(self):
        self.assertEqual(15, self.result.counts["faq"])
        self.assertEqual(10, self.result.counts["policy"])
        self.assertEqual(15, self.result.counts["ticket"])
        self.assertEqual(40, self.result.counts["total"])
        self.assertEqual(40, len(self.result.records))
        self.assertEqual([], self.result.errors)

    def test_expected_source_types_present(self):
        types = {r.source_type for r in self.result.records}
        self.assertEqual({"faq", "policy", "ticket"}, types)

    def test_ids_preserved_and_unique(self):
        faqs = [r.source_id for r in self.result.records if r.source_type == "faq"]
        policies = [r.source_id for r in self.result.records if r.source_type == "policy"]
        tickets = [r.source_id for r in self.result.records if r.source_type == "ticket"]
        self.assertEqual(EXPECTED_FAQ_IDS, faqs)
        self.assertEqual(EXPECTED_POLICY_IDS, policies)
        self.assertEqual(EXPECTED_TICKET_IDS, tickets)
        self.assertEqual(40, len({r.source_id for r in self.result.records}))

    def test_source_authority_preserved(self):
        for record in self.result.records:
            self.assertEqual(
                AUTHORITY_BY_TYPE[record.source_type],
                record.metadata["authority"],
                msg=record.source_id,
            )

    def test_citation_key_and_vector_slot(self):
        for record in self.result.records:
            self.assertEqual(record.source_id, record.citation_key)
            self.assertIsNone(record.vector)

    def test_original_content_preserved(self):
        source_for_type = {"faq": "faqs.md", "policy": "policies.md", "ticket": "tickets.md"}
        texts = {
            filename: ingest.read_source_file(os.path.join(BASE_DIR, filename))[0]
            for filename in source_for_type.values()
        }
        for record in self.result.records:
            text = texts[source_for_type[record.source_type]]
            self.assertIn(record.chunk_text, text, msg=record.source_id)
            self.assertTrue(
                record.chunk_text.startswith(f"# {record.source_id} —"),
                msg=record.source_id,
            )

    def test_undated_records_have_no_fabricated_dates(self):
        year_re = re.compile(r"\b(19|20)\d{2}\b")
        for record in self.result.records:
            if record.source_type in ("faq", "ticket"):
                self.assertEqual("undated", record.metadata["freshness"], msg=record.source_id)
                self.assertIsNone(
                    year_re.search(record.metadata["freshness"]), msg=record.source_id
                )
        for policy_id in ("POLICY-05", "POLICY-08", "POLICY-10"):
            self.assertEqual("undated", self.by_id[policy_id].metadata["freshness"])

    def test_policy_freshness_metadata_preserved(self):
        for policy_id, expected in EXPECTED_POLICY_FRESHNESS.items():
            self.assertEqual(expected, self.by_id[policy_id].metadata["freshness"])

    def test_titles_preserved(self):
        self.assertEqual(
            "How do I access a course after purchasing it?", self.by_id["FAQ-01"].title
        )
        self.assertEqual("Subscription Plans and Billing", self.by_id["POLICY-01"].title)
        self.assertEqual("Login Loop", self.by_id["TICKET-01"].title)

    # -- integrity ---------------------------------------------------------- #

    def test_source_files_unmodified(self):
        for filename, expected in EXPECTED_SOURCE_SHA256.items():
            _, sha = ingest.read_source_file(os.path.join(BASE_DIR, filename))
            self.assertEqual(expected, sha, msg=f"{filename} was modified")
        self.assertEqual(EXPECTED_SOURCE_SHA256, self.result.source_hashes)

    # -- determinism -------------------------------------------------------- #

    def test_ingestion_is_deterministic(self):
        first = ingest.ingest_all(BASE_DIR)
        second = ingest.ingest_all(BASE_DIR)
        self.assertEqual(
            json.dumps(ingest.build_document(first), ensure_ascii=False),
            json.dumps(ingest.build_document(second), ensure_ascii=False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            a = ingest.write_json(ingest.build_document(first), os.path.join(tmp, "a.json"))
            b = ingest.write_json(ingest.build_document(second), os.path.join(tmp, "b.json"))
            self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())


class DatasetSpecificTests(CorpusTestCase):
    """Known corpus-specific cases must be represented correctly (not resolved)."""

    def test_stale_flag_set_exactly(self):
        flagged = {r.source_id for r in self.result.records if r.metadata["is_stale"]}
        self.assertEqual(EXPECTED_STALE_IDS, flagged)
        # POLICY-07 has no stale note and must not be flagged.
        self.assertFalse(self.by_id["POLICY-07"].metadata["is_stale"])
        self.assertIsNone(self.by_id["POLICY-07"].metadata["stale_reason"])

    def test_stale_reason_is_verbatim_source_evidence(self):
        for record in self.result.records:
            if record.metadata["is_stale"]:
                reason = record.metadata["stale_reason"]
                self.assertTrue(reason.strip(), msg=record.source_id)
                # reason must be drawn from the source text (whitespace-normalized)
                squashed = ingest.squash_whitespace(record.chunk_text)
                self.assertIn(reason, squashed, msg=record.source_id)

    def test_contradictory_refund_windows_preserved(self):
        # 7-day (stale, POLICY-02), 14-day (FAQ-02 + POLICY-02), 30-day (tickets).
        self.assertIn("7-day", self.by_id["POLICY-02"].chunk_text)
        self.assertIn("14 days", self.by_id["POLICY-02"].chunk_text)
        self.assertIn("14 days", self.by_id["FAQ-02"].chunk_text)
        self.assertIn("30-day", self.by_id["TICKET-03"].chunk_text)
        self.assertIn("14 day", self.by_id["TICKET-08"].chunk_text)
        self.assertTrue(self.by_id["POLICY-02"].metadata["is_stale"])
        self.assertIn("refund_window", self.by_id["POLICY-02"].metadata["contradiction_topics"])

    def test_annual_subscription_ambiguity_preserved(self):
        record = self.by_id["POLICY-01"]
        self.assertTrue(record.metadata["is_stale"])
        self.assertIn("billed monthly", record.metadata["stale_reason"])
        self.assertIn("annual_billing", record.metadata["contradiction_topics"])

    def test_offline_download_contradiction_preserved(self):
        self.assertTrue(self.by_id["FAQ-07"].metadata["is_stale"])
        self.assertIn("desktop download", self.by_id["FAQ-07"].metadata["stale_reason"])
        self.assertTrue(self.by_id["POLICY-04"].metadata["is_stale"])
        self.assertIn("offline_downloads", self.by_id["FAQ-07"].metadata["contradiction_topics"])

    def test_browser_support_staleness_preserved(self):
        record = self.by_id["POLICY-09"]
        self.assertTrue(record.metadata["is_stale"])
        self.assertIn("Internet Explorer", record.metadata["stale_reason"])
        self.assertIn("browser_support", record.metadata["contradiction_topics"])

    def test_caption_contradiction_preserved(self):
        record = self.by_id["POLICY-03"]
        self.assertTrue(record.metadata["is_stale"])
        self.assertIn("captions", record.metadata["stale_reason"])
        self.assertIn("captions_accessibility", record.metadata["contradiction_topics"])

    def test_progress_synchronization_wording_preserved(self):
        record = self.by_id["POLICY-06"]
        self.assertTrue(record.metadata["is_stale"])
        self.assertIn("instantly", record.metadata["stale_reason"])
        self.assertIn("progress_sync", record.metadata["contradiction_topics"])

    def test_payment_security_records_flagged(self):
        for source_id in ("FAQ-15", "POLICY-07", "POLICY-10", "TICKET-05"):
            self.assertIn(
                "payment_data_collection",
                self.by_id[source_id].metadata["contradiction_topics"],
                msg=source_id,
            )
        self.assertIn("full card number", self.by_id["FAQ-15"].chunk_text)
        self.assertIn("first six", self.by_id["POLICY-10"].chunk_text)

    def test_ambiguous_ticket_cancel_request(self):
        record = self.by_id["TICKET-07"]
        self.assertIn("Cancel my LearnForge", record.chunk_text)
        self.assertIn(
            "ambiguous_intent_requires_clarification",
            record.metadata["ambiguity_flags"],
        )

    def test_outdated_document_reliance_ticket(self):
        record = self.by_id["TICKET-15"]
        self.assertIn(
            "outdated_documentation_reliance", record.metadata["ambiguity_flags"]
        )
        self.assertIn("outdated", record.metadata["ticket_status"])


class TicketStatusTests(CorpusTestCase):
    """Tickets stay identifiable as historical evidence with their status intact."""

    def test_all_ticket_statuses_preserved(self):
        for ticket_id, expected in EXPECTED_TICKET_STATUS.items():
            self.assertEqual(expected, self.by_id[ticket_id].metadata["ticket_status"])

    def test_escalated_flags(self):
        escalated = {
            r.source_id for r in self.result.records if r.metadata["escalated"]
        }
        self.assertEqual(EXPECTED_ESCALATED, escalated)

    def test_unresolved_flags(self):
        for ticket_id in EXPECTED_TICKET_IDS:
            record = self.by_id[ticket_id]
            if ticket_id in EXPECTED_RESOLVED:
                self.assertFalse(record.metadata["unresolved"], msg=ticket_id)
            else:
                self.assertTrue(record.metadata["unresolved"], msg=ticket_id)

    def test_tickets_are_never_stale_sources(self):
        for ticket_id in EXPECTED_TICKET_IDS:
            record = self.by_id[ticket_id]
            self.assertFalse(record.metadata["is_stale"], msg=ticket_id)
            self.assertIsNone(record.metadata["stale_reason"], msg=ticket_id)
            self.assertEqual("undated", record.metadata["freshness"], msg=ticket_id)


VALID_FAQ = "# FAQ-01 — Q\n\nQUESTION:\nQ?\n\nANSWER:\nA.\n"
VALID_POLICY = "# POLICY-01 — P\n\nBody text.\n"
VALID_TICKET = "# TICKET-01 — T\n\nUSER:\nhi\n\nAGENT:\nhello\n\nSTATUS: Resolved.\n"


def write_corpus(directory, *, faqs=VALID_FAQ, policies=VALID_POLICY, tickets=VALID_TICKET):
    """Write a minimal three-file corpus for malformed-input tests."""
    os.makedirs(directory, exist_ok=True)
    for name, content in (
        ("faqs.md", faqs),
        ("policies.md", policies),
        ("tickets.md", tickets),
    ):
        Path(directory, name).write_text(content, encoding="utf-8")
    return directory


class ValidationTests(unittest.TestCase):
    """Direct tests of the required-field validation rules."""

    def test_valid_record_has_no_issues(self):
        self.assertEqual([], validate_record(make_record()))

    def test_empty_title_rejected(self):
        self.assertTrue(any("title" in i for i in validate_record(make_record(title="  "))))

    def test_empty_content_rejected(self):
        self.assertTrue(
            any("chunk_text" in i for i in validate_record(make_record(chunk_text="")))
        )

    def test_malformed_source_id_rejected(self):
        issues = validate_record(make_record(source_id="FAQ-1", citation_key="FAQ-1"))
        self.assertTrue(any("pattern" in i for i in issues))

    def test_id_type_mismatch_rejected(self):
        issues = validate_record(make_record(source_id="POLICY-99", citation_key="POLICY-99"))
        self.assertTrue(any("implies source_type" in i for i in issues))

    def test_citation_key_mismatch_rejected(self):
        issues = validate_record(make_record(citation_key="FAQ-99x"))
        self.assertTrue(any("citation_key" in i for i in issues))

    def test_stale_without_reason_rejected(self):
        record = make_record(
            metadata=build_metadata(
                source_type="faq", freshness="undated", is_stale=True, stale_reason=None
            )
        )
        self.assertTrue(any("stale_reason" in i for i in validate_record(record)))

    def test_reason_without_stale_rejected(self):
        record = make_record(
            metadata=build_metadata(
                source_type="faq",
                freshness="undated",
                is_stale=False,
                stale_reason="should not be here",
            )
        )
        self.assertTrue(any("stale_reason" in i for i in validate_record(record)))

    def test_wrong_authority_rejected(self):
        meta = build_metadata(
            source_type="faq", freshness="undated", is_stale=False, stale_reason=None
        )
        meta["authority"] = 3
        self.assertTrue(
            any("authority" in i for i in validate_record(make_record(metadata=meta)))
        )

    def test_non_ticket_with_status_rejected(self):
        meta = build_metadata(
            source_type="faq",
            freshness="undated",
            is_stale=False,
            stale_reason=None,
            ticket_status="Resolved",
        )
        self.assertTrue(any("ticket_status" in i for i in validate_record(make_record(metadata=meta))))

    def test_ticket_without_status_rejected(self):
        meta = build_metadata(
            source_type="ticket",
            freshness="undated",
            is_stale=False,
            stale_reason=None,
            ticket_status=None,
        )
        issues = validate_record(
            make_record(
                source_id="TICKET-99",
                source_type="ticket",
                citation_key="TICKET-99",
                metadata=meta,
            )
        )
        self.assertTrue(any("ticket_status" in i for i in issues))

    def test_non_null_vector_rejected(self):
        self.assertTrue(any("vector" in i for i in validate_record(make_record(vector=[0.1]))))


class MalformedInputTests(unittest.TestCase):
    """Malformed source files must be reported, not silently accepted."""

    def test_valid_minimal_corpus_ingests(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_corpus(tmp)
            result = ingest.ingest_all(tmp)
        self.assertEqual(3, result.counts["total"])
        self.assertEqual([], result.errors)

    def test_missing_ticket_status_reported_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_corpus(tmp, tickets="# TICKET-01 — T\n\nUSER:\nhi\n\nAGENT:\nhello\n")
            with self.assertRaises(ingest.IngestionError):
                ingest.ingest_all(tmp, strict=True)
            result = ingest.ingest_all(tmp, strict=False)
        self.assertTrue(
            any("ticket_status" in issue for e in result.errors for issue in e["issues"])
        )

    def test_empty_file_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_corpus(tmp, faqs="")
            result = ingest.ingest_all(tmp, strict=False)
        self.assertTrue(
            any("no records matched" in issue for e in result.errors for issue in e["issues"])
        )

    def test_duplicate_ids_reported(self):
        duplicate = VALID_FAQ + "\n---\n\n" + VALID_FAQ
        with tempfile.TemporaryDirectory() as tmp:
            write_corpus(tmp, faqs=duplicate)
            result = ingest.ingest_all(tmp, strict=False)
        self.assertTrue(
            any("duplicate source_id" in issue for e in result.errors for issue in e["issues"])
        )

    def test_wrong_source_type_in_file_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_corpus(tmp, faqs=VALID_TICKET)
            result = ingest.ingest_all(tmp, strict=False)
        self.assertTrue(
            any("does not match expected" in issue for e in result.errors for issue in e["issues"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
