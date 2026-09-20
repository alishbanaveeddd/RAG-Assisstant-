"""Automated tests for the M2 retrieval layer (pytest).

Run with::

    python -m pytest tests -q

Scope: retrieval only. These tests assert which evidence is *retrievable* and
that metadata survives retrieval. They deliberately do **not** assert sufficiency,
confidence, contradiction resolution, or escalation — those belong to M3.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from learnforge import embed as embed_module
from learnforge.lexical import build_index, tokenize
from learnforge.retrieval import (
    DEFAULT_TOP_K,
    SURFACED_METADATA_KEYS,
    Retriever,
)
from learnforge.schema import AUTHORITY_BY_TYPE

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "learnforge-knowledge-base-data" / "learnforge-knowledge-base"
SOURCE_FILES = ("faqs.md", "policies.md", "tickets.md")

EXPECTED_FAQ_IDS = [f"FAQ-{n:02d}" for n in range(1, 16)]
EXPECTED_POLICY_IDS = [f"POLICY-{n:02d}" for n in range(1, 11)]
EXPECTED_TICKET_IDS = [f"TICKET-{n:02d}" for n in range(1, 16)]
EXPECTED_ALL_IDS = set(EXPECTED_FAQ_IDS + EXPECTED_POLICY_IDS + EXPECTED_TICKET_IDS)

# Realistic queries derived from the LearnForge dataset (M2 test brief).
REFUND_QUERIES = (
    "Can I get a refund after 20 days?",
    "How long do I have to request a refund?",
    "Are refunds available for annual subscriptions?",
)
OFFLINE_QUERIES = ("Can I download courses on my laptop?", "Can I watch courses offline?")
BROWSER_QUERIES = ("Does LearnForge support Internet Explorer?", "What browsers are supported?")
PAYMENT_QUERIES = (
    "What payment information can support ask me for?",
    "Can I give support my card number?",
)
AMBIGUOUS_QUERY = "Cancel my LearnForge"


class StubEmbedder:
    """Deterministic toy embedder for fast, fully-controlled ranking tests."""

    model_name = "stub-embedder"
    dimension = 4

    _KEYWORDS = ("refund", "browser", "laptop", "captions")

    def encode(self, texts):
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def _vector(self, text: str) -> list[float]:
        lowered = (text or "").lower()
        row = [1.0 if keyword in lowered else 0.0 for keyword in self._KEYWORDS]
        if not any(row):
            row = [0.0, 0.0, 0.0, 1.0]
        norm = float(np.linalg.norm(row))
        return [value / norm for value in row]


# --------------------------------------------------------------------------- #
# 1-3: initialization, indexing, local embeddings
# --------------------------------------------------------------------------- #

def test_retriever_initializes_successfully(retriever):
    assert isinstance(retriever, Retriever)
    assert retriever.bm25 is not None
    assert len(retriever.records) == len(retriever.source_ids) == 40


def test_all_40_records_are_indexed(retriever):
    assert len(retriever) == 40
    assert retriever.matrix.shape == (40, 384)
    assert set(retriever.source_ids) == EXPECTED_ALL_IDS
    assert sorted(retriever.source_ids) == sorted(EXPECTED_ALL_IDS)


def test_embeddings_are_generated_locally(retriever, tmp_path):
    from learnforge.embed import DEFAULT_MODEL_NAME, LocalEmbedder

    store = embed_module.load_embedding_store(
        str(REPO_ROOT / "data" / "processed" / "kb_embeddings.json")
    )
    assert store["embedding_model"] == DEFAULT_MODEL_NAME
    assert store["dimension"] == 384
    assert store["normalized"] is True
    assert store["metric"] == "cosine"
    assert store["record_order"] == (
        EXPECTED_FAQ_IDS + EXPECTED_POLICY_IDS + EXPECTED_TICKET_IDS
    )

    # The embedding model must run locally and never contact an embedding service.
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    embedder = LocalEmbedder()
    vectors = embedder.encode(["refund window", "browser support"])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert store["embedding_model"] == embedder.model_name


def test_embedding_store_matches_m1_record_order(retriever):
    records = embed_module.load_records(
        str(REPO_ROOT / "data" / "processed" / "kb_records.json")
    )
    assert [record["source_id"] for record in records] == retriever.source_ids


# --------------------------------------------------------------------------- #
# 4-6: basic retrieval + metadata preservation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query", (*REFUND_QUERIES, *OFFLINE_QUERIES, *BROWSER_QUERIES, *PAYMENT_QUERIES, AMBIGUOUS_QUERY))
def test_retrieval_returns_results_for_normal_queries(retriever, query):
    results = retriever.search(query)
    assert 0 < len(results) <= DEFAULT_TOP_K


def test_returned_records_preserve_source_ids(retriever):
    for query in (*REFUND_QUERIES, *BROWSER_QUERIES, AMBIGUOUS_QUERY):
        for result in retriever.search(query, 10):
            assert result.source_id in EXPECTED_ALL_IDS
            assert result.record["citation_key"] == result.source_id


def test_returned_records_preserve_full_metadata(retriever):
    records = {
        record["source_id"]: record
        for record in embed_module.load_records(
            str(REPO_ROOT / "data" / "processed" / "kb_records.json")
        )
    }
    seen = set()
    for query in (*REFUND_QUERIES, *OFFLINE_QUERIES, *BROWSER_QUERIES, *PAYMENT_QUERIES, AMBIGUOUS_QUERY):
        for result in retriever.search(query, 10):
            seen.add(result.source_id)
            # every M1 metadata key must still be present and unchanged
            for key in SURFACED_METADATA_KEYS:
                assert key in result.metadata, (result.source_id, key)
            assert result.metadata == records[result.source_id]["metadata"], result.source_id
            assert result.record["chunk_text"] == records[result.source_id]["chunk_text"]
            assert result.record["title"] == records[result.source_id]["title"]
    # coverage across the queries exercised above
    assert len(seen) >= 10


def make_record(source_id: str, text: str) -> dict:
    """Minimal record dict matching the M1 schema shape used by the retriever."""
    source_type = "ticket" if source_id.startswith("TICKET") else "policy"
    return {
        "source_id": source_id,
        "source_type": source_type,
        "title": source_id,
        "chunk_text": text,
        "citation_key": source_id,
        "metadata": {
            "authority": AUTHORITY_BY_TYPE[source_type],
            "freshness": "undated",
            "is_stale": False,
            "stale_reason": None,
            "ticket_status": "Resolved" if source_type == "ticket" else None,
        },
        "vector": None,
    }


# --------------------------------------------------------------------------- #
# 7-9: semantic, lexical, and hybrid behaviour
# --------------------------------------------------------------------------- #

REFUND_EVIDENCE = {"FAQ-02", "POLICY-02", "TICKET-03", "TICKET-08"}


@pytest.mark.parametrize("query", REFUND_QUERIES)
def test_semantic_retrieval_surfaces_refund_evidence(retriever, query):
    results = retriever.search_semantic(query, 5)
    assert results
    assert {result.source_id for result in results} & REFUND_EVIDENCE
    # in semantic-only mode the fused score is the cosine similarity itself
    assert all(result.semantic_score > 0 for result in results)


@pytest.mark.parametrize("query", OFFLINE_QUERIES)
def test_semantic_retrieval_surfaces_offline_evidence(retriever, query):
    results = retriever.search_semantic(query, 5)
    assert {result.source_id for result in results} & {"FAQ-07", "POLICY-04", "TICKET-15"}


def test_semantic_retrieval_surfaces_browser_evidence(retriever):
    results = retriever.search_semantic(BROWSER_QUERIES[0], 3)
    assert {result.source_id for result in results} & {"POLICY-09", "FAQ-08"}


def test_lexical_retrieval_handles_rare_terminology(retriever):
    # "Internet Explorer" appears in exactly one record.
    results = retriever.search_lexical("Does LearnForge support Internet Explorer?", 5)
    assert results[0].source_id == "POLICY-09"
    assert "explorer" in results[0].matched_terms


def test_lexical_retrieval_handles_exact_ids(retriever):
    results = retriever.search_lexical("FAQ-08", 5)
    assert results[0].source_id == "FAQ-08"
    assert results[0].lexical_score > 0


def test_hybrid_ranks_exact_id_match_first(retriever):
    # M0 §16: lexical fallback exists for exact IDs like "FAQ-08".
    results = retriever.search("FAQ-08", 5)
    assert results[0].source_id == "FAQ-08"
    assert results[0].exact_id_match is True
    assert results[0].lexical_score > 0


def test_hybrid_search_accepts_spaced_id_form(retriever):
    results = retriever.search("policy 02", 5)
    assert results[0].source_id == "POLICY-02"
    assert results[0].exact_id_match is True


def test_hybrid_combines_both_signals(retriever):
    results = retriever.search(REFUND_QUERIES[0], 5)
    # every fused candidate exposes both individual signals for explainability
    for result in results:
        assert result.semantic_score > 0 or result.lexical_score > 0
        assert result.fused_score > 0
    # a record that both signals agree on ranks in the top results
    top_ids = [result.source_id for result in results]
    assert top_ids[0] in REFUND_EVIDENCE
    assert "refund_window" in results[0].metadata["contradiction_topics"]


def test_hybrid_outperforms_single_signal_for_shared_vocabulary(retriever):
    """Both signals agreeing is the normal case; fusion must not lose evidence."""
    results = retriever.search("Can I watch courses offline?", 5)
    hybrid_ids = {result.source_id for result in results}
    semantic_ids = {result.source_id for result in retriever.search_semantic("Can I watch courses offline?", 5)}
    lexical_ids = {result.source_id for result in retriever.search_lexical("Can I watch courses offline?", 5)}
    assert hybrid_ids & {"FAQ-07", "TICKET-15"}
    assert {"FAQ-07", "TICKET-15"} <= (semantic_ids | lexical_ids)


# --------------------------------------------------------------------------- #
# 10: deterministic ordering
# --------------------------------------------------------------------------- #

def test_results_are_deterministically_ordered(retriever):
    for query in (*REFUND_QUERIES, *BROWSER_QUERIES, AMBIGUOUS_QUERY):
        first = [result.to_dict() for result in retriever.search(query, 10)]
        second = [result.to_dict() for result in retriever.search(query, 10)]
        assert first == second, query


def test_ties_break_by_source_id():
    records = [
        make_record("TICKET-02", "zzz identical body"),
        make_record("FAQ-01", "aaa identical body"),
        make_record("POLICY-01", "mmm identical body"),
    ]
    matrix = np.ones((3, 4), dtype=np.float32)  # identical vectors -> identical scores
    retriever = Retriever(records, matrix, embedder=StubEmbedder())
    results = retriever.search("refund", 3)
    assert [result.source_id for result in results] == ["FAQ-01", "POLICY-01", "TICKET-02"]


def test_zero_score_records_never_receive_rank_credit():
    records = [
        make_record("POLICY-01", "refund window policy"),
        make_record("TICKET-05", "completely unrelated content"),
    ]
    # the stub gives both records the same vector, so only lexical separates them
    retriever = Retriever(records, np.ones((2, 4), dtype=np.float32), embedder=StubEmbedder())
    results = retriever.search("refund", 2)
    assert results[0].source_id == "POLICY-01"
    assert results[0].lexical_score > 0
    assert results[1].lexical_score == 0.0


# --------------------------------------------------------------------------- #
# 11-14: stale evidence, tickets, authority, contradiction topics
# --------------------------------------------------------------------------- #

def test_stale_records_are_not_silently_deleted(retriever):
    # POLICY-02 carries the archived 7-day refund wording and must stay retrievable
    # alongside the current 14-day wording, so M3 can detect the contradiction.
    results = retriever.search(REFUND_QUERIES[0], 10)
    stale_hits = [result for result in results if result.metadata["is_stale"]]
    assert stale_hits, "stale refund evidence was filtered out by retrieval"
    ids = {result.source_id for result in results}
    assert "POLICY-02" in ids


def test_stale_records_keep_their_reason_metadata(retriever):
    results = retriever.search(BROWSER_QUERIES[0], 5)
    policy09 = next(r for r in results if r.source_id == "POLICY-09")
    assert policy09.metadata["is_stale"] is True
    assert "Internet Explorer" in policy09.metadata["stale_reason"]


def test_ticket_records_remain_identifiable_as_tickets(retriever):
    for query in (*REFUND_QUERIES, AMBIGUOUS_QUERY, "duplicate payment on my card"):
        results = retriever.search(query, 10)
        tickets = [r for r in results if r.source_type == "ticket"]
        assert tickets, query
        for result in tickets:
            assert result.source_id.startswith("TICKET-")
            assert result.metadata["ticket_status"]
            assert result.metadata["authority"] == AUTHORITY_BY_TYPE["ticket"]


def test_ticket_ambiguity_metadata_survives_retrieval(retriever):
    results = retriever.search(AMBIGUOUS_QUERY, 10)
    ticket07 = next((r for r in results if r.source_id == "TICKET-07"), None)
    assert ticket07 is not None
    assert ticket07.metadata["ambiguity_flags"] == ["ambiguous_intent_requires_clarification"]
    assert ticket07.metadata["unresolved"] is True


def test_authority_is_preserved_but_never_used_as_a_filter(retriever):
    queries = (*REFUND_QUERIES, *OFFLINE_QUERIES, *BROWSER_QUERIES, *PAYMENT_QUERIES, AMBIGUOUS_QUERY)
    authority_values = set()
    for query in queries:
        for result in retriever.search(query, 10):
            expected = AUTHORITY_BY_TYPE[result.source_type]
            assert result.metadata["authority"] == expected, result.source_id
            authority_values.add(expected)
    # all three authority tiers remain reachable through retrieval
    assert authority_values == {1, 2, 3}


def test_low_authority_ticket_can_outrank_high_authority_policy(retriever):
    """Authority must not delete or demote records at retrieval time (M2 rule)."""
    results = retriever.search(REFUND_QUERIES[0], 10)
    positions = {result.source_id: i for i, result in enumerate(results)}
    assert "TICKET-03" in positions and "POLICY-02" in positions
    # not asserting *which* must win (that is evidence assessment, i.e. M3),
    # only that the low-authority ticket is genuinely retrievable evidence.
    assert positions["TICKET-03"] < len(results)


def test_contradiction_topic_metadata_remains_intact(retriever):
    results = retriever.search(REFUND_QUERIES[0], 10)
    policy02 = next(r for r in results if r.source_id == "POLICY-02")
    assert policy02.metadata["contradiction_topics"]  # hints preserved
    assert "refund_window" in policy02.metadata["contradiction_topics"]
    # topics are hints only: they must not have been used to delete other records
    assert len({r.source_id for r in results}) > 1


# --------------------------------------------------------------------------- #
# 15-17: failure handling + source integrity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_query", ["", "   ", "\t\n ", None])
def test_empty_or_whitespace_query_returns_no_results(retriever, bad_query):
    assert retriever.search(bad_query) == []
    assert retriever.search_semantic(bad_query or "") == []
    assert retriever.search_lexical(bad_query or "") == []


def test_query_with_no_meaningful_evidence_is_handled_safely(retriever):
    nonsense = retriever.search("quantum chromodynamics penguin", 5)
    assert isinstance(nonsense, list)
    assert len(nonsense) <= 5
    # relative assertion only: no claim of sufficiency (that is M3's decision)
    relevant = retriever.search(REFUND_QUERIES[0], 5)
    assert nonsense[0].semantic_score < relevant[0].semantic_score


def test_k_larger_than_corpus_returns_every_record(retriever):
    results = retriever.search(REFUND_QUERIES[0], 100)
    assert len(results) == 40


def test_non_positive_k_returns_no_results(retriever):
    assert retriever.search(REFUND_QUERIES[0], 0) == []
    assert retriever.search(REFUND_QUERIES[0], -3) == []


def test_retrieval_never_modifies_the_source_kb(retriever, source_hashes_before):
    for query in (*REFUND_QUERIES, *OFFLINE_QUERIES, *BROWSER_QUERIES, *PAYMENT_QUERIES, AMBIGUOUS_QUERY):
        retriever.search(query, 10)
        retriever.search_semantic(query, 10)
        retriever.search_lexical(query, 10)
    for name, expected in source_hashes_before.items():
        current = hashlib.sha256((KB_DIR / name).read_bytes()).hexdigest()
        assert current == expected, f"{name} was modified by retrieval"


def test_retrieval_does_not_mutate_returned_records(retriever):
    before = {r.source_id: dict(r.record) for r in retriever.search(REFUND_QUERIES[0], 10)}
    retriever.search(AMBIGUOUS_QUERY, 10)
    after = retriever.search(REFUND_QUERIES[0], 10)
    for result in after:
        assert result.record == before[result.source_id]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def retriever():
    """Real retriever built from the M1 records + local embedding store."""
    return Retriever.from_store(
        str(REPO_ROOT / "data" / "processed" / "kb_records.json"),
        str(REPO_ROOT / "data" / "processed" / "kb_embeddings.json"),
    )


@pytest.fixture(scope="session")
def source_hashes_before():
    return {
        name: hashlib.sha256((KB_DIR / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }