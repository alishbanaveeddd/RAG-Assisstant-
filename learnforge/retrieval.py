"""Hybrid retrieval layer (Milestone M2).

Takes a user query and returns the most relevant KB records **with their metadata
intact**, for M3's evidence / confidence / contradiction assessment to decide what
to do with them.

Architecture (per M0 §16 + M2 requirements):

* **Semantic** — dense cosine similarity over locally-computed embeddings
  (``learnforge.embed``).
* **Lexical** — Okapi BM25 over the same per-record chunks (``learnforge.lexical``).
* **Fusion** — Reciprocal Rank Fusion (RRF), a standard rank-fusion method. M0
  proposed an "optional light lexical fallback" but specified **no fusion method
  and no weights**; RRF is used instead of inventing arbitrary score weights, and
  is documented in ``docs/retrieval.md``.

Deliberate non-decisions (these belong to M3, not M2):

* **Authority is NOT a retrieval filter** — a low-authority ticket may be the most
  relevant evidence for an exception/conflict question (M2 §"authority ≠ relevance").
* **Stale records are NOT deleted** — they must stay retrievable so M3 can detect
  contradictions (M2 §"freshness").
* **No sufficiency/confidence threshold** is applied; scores are exposed instead.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from learnforge.embed import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_RECORDS_PATH,
    load_embedding_store,
    load_records,
    store_matrix,
)
from learnforge.lexical import BM25Index, build_index, tokenize

#: Reciprocal Rank Fusion constant (the standard default from the RRF paper).
RRF_K = 60

#: Default number of results returned to the caller.
DEFAULT_TOP_K = 5

#: Size of the per-signal candidate pool that feeds the fusion step.
DEFAULT_CANDIDATE_POOL = 10

#: Metadata keys surfaced on every retrieval result (M2 requires none are dropped).
SURFACED_METADATA_KEYS = (
    "authority",
    "freshness",
    "is_stale",
    "stale_reason",
    "ticket_status",
    "escalated",
    "unresolved",
    "ambiguity_flags",
    "contradiction_topics",
)


@dataclass
class RetrievalResult:
    """One retrieved record plus the individual retrieval signals that ranked it.

    The full original record (including untouched ``metadata``) is preserved in
    ``record``; :meth:`to_dict` flattens the fields M3 needs.
    """

    record: dict[str, Any]
    semantic_score: float
    lexical_score: float
    fused_score: float
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    matched_terms: list[str] = field(default_factory=list)
    exact_id_match: bool = False

    # -- convenience accessors ------------------------------------------- #

    @property
    def source_id(self) -> str:
        return self.record["source_id"]

    @property
    def source_type(self) -> str:
        return self.record["source_type"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self.record["metadata"]

    def to_dict(self) -> dict[str, Any]:
        """Flatten the record, its metadata, and the retrieval signals."""
        meta = self.record["metadata"]
        flat: dict[str, Any] = {
            "source_id": self.record["source_id"],
            "source_type": self.record["source_type"],
            "title": self.record["title"],
            "chunk_text": self.record["chunk_text"],
            "citation_key": self.record["citation_key"],
            "scores": {
                "semantic_score": round(self.semantic_score, 6),
                "lexical_score": round(self.lexical_score, 6),
                "fused_score": round(self.fused_score, 6),
                "semantic_rank": self.semantic_rank,
                "lexical_rank": self.lexical_rank,
                "matched_terms": list(self.matched_terms),
                "exact_id_match": self.exact_id_match,
            },
            "metadata": dict(meta),
        }
        for key in SURFACED_METADATA_KEYS:
            flat[key] = meta.get(key)
        return flat

    def summary(self) -> str:
        """One-line human-readable summary for manual verification."""
        meta = self.metadata
        return (
            f"{self.source_id:<11} [{self.source_type:<6}] "
            f"fused={self.fused_score:.5f} sem={self.semantic_score:.4f} "
            f"lex={self.lexical_score:.4f} stale={str(meta.get('is_stale')):<5} "
            f"auth={meta.get('authority')} status={meta.get('ticket_status')!r}"
            + (" id_match=True" if self.exact_id_match else "")
        )


#: Matches an explicit record ID inside a query, e.g. "FAQ-08", "policy 02",
#: "ticket-11". M0 §16 named exact-ID lookup as the reason for the lexical fallback.
_ID_QUERY_RE = re.compile(r"\b(FAQ|POLICY|TICKET)\s*[-_]?\s*(\d{2})\b", re.IGNORECASE)


class Retriever:
    """Hybrid (dense + BM25, RRF-fused) retriever over the normalized KB records."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        matrix: np.ndarray,
        *,
        embedder: Any,
        bm25: Optional[BM25Index] = None,
        rrf_k: int = RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> None:
        self.records: list[dict[str, Any]] = [dict(record) for record in records]
        self.source_ids: list[str] = [record["source_id"] for record in self.records]
        self.matrix = np.asarray(matrix, dtype=np.float32)
        if self.matrix.shape[0] != len(self.records):
            raise ValueError(
                f"embedding matrix has {self.matrix.shape[0]} rows but there are "
                f"{len(self.records)} records"
            )
        self.embedder = embedder
        self.bm25 = bm25 if bm25 is not None else build_index(self.records)
        self.rrf_k = int(rrf_k)
        self.candidate_pool = int(candidate_pool)
        self._index = {source_id: i for i, source_id in enumerate(self.source_ids)}

    def __len__(self) -> int:
        return len(self.records)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_store(
        cls,
        records_path: str = DEFAULT_RECORDS_PATH,
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        *,
        embedder: Any = None,
        rrf_k: int = RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> "Retriever":
        """Load the M1 records + the local embedding store and build a retriever."""
        records = load_records(records_path)
        store = load_embedding_store(embeddings_path)
        source_ids = [record["source_id"] for record in records]
        matrix = store_matrix(store, source_ids)

        if embedder is None:
            from learnforge.embed import DEFAULT_MODEL_NAME, get_embedder

            embedder = get_embedder(store.get("embedding_model") or DEFAULT_MODEL_NAME)

        stored_model = store.get("embedding_model")
        embedder_model = getattr(embedder, "model_name", None)
        if stored_model and embedder_model and stored_model != embedder_model:
            raise ValueError(
                f"embedding store was built with {stored_model!r} but the query "
                f"embedder is {embedder_model!r}; rebuild the store with "
                "'python -m learnforge.embed'"
            )

        return cls(
            records,
            matrix,
            embedder=embedder,
            rrf_k=rrf_k,
            candidate_pool=candidate_pool,
        )

    # -- signals ---------------------------------------------------------- #

    def semantic_scores(self, query: str) -> list[float]:
        """Cosine similarity of the query against every record (normalized dot)."""
        if not self.records:
            return []
        query_vector = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        if query_vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query embedding has dimension {query_vector.shape[0]} but the "
                f"index has {self.matrix.shape[1]}"
            )
        return [float(value) for value in (self.matrix @ query_vector)]

    def lexical_scores(self, query: str) -> list[float]:
        """BM25 score of the query against every record."""
        return self.bm25.score(query)

    # -- ranking helpers -------------------------------------------------- #

    def _rank_order(self, scores: Sequence[float]) -> list[int]:
        """Indices with a **positive** score, sorted by score desc.

        Ties break by ascending ``source_id`` so the ordering is fully
        deterministic (M2 test 10).

        Documents whose score is zero (or negative) are **excluded**: rank
        fusion must only receive documents that the signal actually matched.
        Without this, an arbitrary tie-break would hand rank credit to
        unrelated records — e.g. an exact-ID query like "FAQ-08" scores 0.0
        on BM25 for every other record, which would otherwise let unrelated
        records outrank the true match.
        """
        matched = [index for index, score in enumerate(scores) if score > 0.0]
        matched.sort(key=lambda index: (-scores[index], self.source_ids[index]))
        return matched

    def _exact_id_matches(self, query: str) -> set[str]:
        """Return record IDs explicitly referenced in the query, e.g. "FAQ-08"."""
        referenced = {
            f"{match.group(1).upper()}-{match.group(2)}"
            for match in _ID_QUERY_RE.finditer(query)
        }
        return referenced & set(self.source_ids)

    def _matched_terms(self, query: str, index: int) -> list[str]:
        doc_terms = set(tokenize(self.bm25.documents[index]))
        return sorted(term for term in self.bm25.matching_terms(query) if term in doc_terms)

    def _build_result(
        self,
        index: int,
        *,
        semantic_score: float,
        lexical_score: float,
        fused_score: float,
        semantic_rank: Optional[int],
        lexical_rank: Optional[int],
        query: str,
        exact_id_match: bool = False,
    ) -> RetrievalResult:
        return RetrievalResult(
            record=dict(self.records[index]),
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            fused_score=fused_score,
            semantic_rank=semantic_rank,
            lexical_rank=lexical_rank,
            matched_terms=self._matched_terms(query, index) if query.strip() else [],
            exact_id_match=exact_id_match,
        )

    # -- public search API ------------------------------------------------ #

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Hybrid search: dense + BM25 candidates fused with RRF.

        Returns up to ``k`` results ordered by descending fused score, with ties
        broken deterministically by ``source_id``. An empty/whitespace query
        returns ``[]`` (no exception).

        Only records that a signal actually matched (score > 0) receive rank
        credit, and an explicit record ID in the query (e.g. "FAQ-08") is placed
        first — see ``_rank_order`` and the comment inside this method.
        """
        if not (query or "").strip() or k <= 0 or not self.records:
            return []

        semantic = self.semantic_scores(query)
        lexical = self.lexical_scores(query)

        # Only documents each signal actually matched (score > 0) are ranked.
        semantic_order = self._rank_order(semantic)
        lexical_order = self._rank_order(lexical)
        semantic_rank = {index: rank + 1 for rank, index in enumerate(semantic_order)}
        lexical_rank = {index: rank + 1 for rank, index in enumerate(lexical_order)}

        pool = max(self.candidate_pool, k)
        candidates = set(semantic_order[:pool]) | set(lexical_order[:pool])

        fused: dict[int, float] = {}
        for index in candidates:
            # A document may be matched by one signal only, so ranks are optional.
            semantic_pos = semantic_rank.get(index)
            lexical_pos = lexical_rank.get(index)
            score = 0.0
            if semantic_pos is not None and semantic_pos <= pool:
                score += 1.0 / (self.rrf_k + semantic_pos)
            if lexical_pos is not None and lexical_pos <= pool:
                score += 1.0 / (self.rrf_k + lexical_pos)
            fused[index] = score

        # M0 §16: the lexical fallback exists so that an explicit record ID in the
        # query (e.g. "FAQ-08") resolves to that record. Dense similarity is
        # uninformative for bare IDs, so an exact-ID hit is placed first, before
        # the RRF ordering. This is a retrieval-ranking rule only — it says
        # nothing about authority, currency, or sufficiency (that is M3's job).
        exact_ids = self._exact_id_matches(query)
        ordered = sorted(
            candidates,
            key=lambda index: (
                0 if self.source_ids[index] in exact_ids else 1,
                -fused[index],
                self.source_ids[index],
            ),
        )
        return [
            self._build_result(
                index,
                semantic_score=semantic[index],
                lexical_score=lexical[index],
                fused_score=fused[index],
                semantic_rank=semantic_rank.get(index),
                lexical_rank=lexical_rank.get(index),
                query=query,
                exact_id_match=self.source_ids[index] in exact_ids,
            )
            for index in ordered[:k]
        ]

    def search_semantic(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Dense-only search (used to inspect the semantic signal in isolation)."""
        if not (query or "").strip() or k <= 0 or not self.records:
            return []
        scores = self.semantic_scores(query)
        order = self._rank_order(scores)
        return [
            self._build_result(
                index,
                semantic_score=scores[index],
                lexical_score=0.0,
                fused_score=scores[index],
                semantic_rank=rank + 1,
                lexical_rank=None,
                query=query,
            )
            for rank, index in enumerate(order[:k])
        ]

    def search_lexical(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Lexical-only search (used to inspect the BM25 signal in isolation)."""
        if not (query or "").strip() or k <= 0 or not self.records:
            return []
        scores = self.lexical_scores(query)
        order = self._rank_order(scores)
        return [
            self._build_result(
                index,
                semantic_score=0.0,
                lexical_score=scores[index],
                fused_score=scores[index],
                semantic_rank=None,
                lexical_rank=rank + 1,
                query=query,
            )
            for rank, index in enumerate(order[:k])
        ]


# --------------------------------------------------------------------------- #
# CLI (manual verification)
# --------------------------------------------------------------------------- #

def format_results(query: str, results: Sequence[RetrievalResult], mode: str) -> str:
    """Render retrieval results for manual inspection."""
    lines = [f'query={query!r} mode={mode} top_k={len(results)}']
    for position, result in enumerate(results, 1):
        lines.append(f"  {position}. {result.summary()}")
        lines.append(f"     title: {result.record['title']!r}")
        meta = result.metadata
        if meta.get("is_stale"):
            lines.append(f"     stale_reason: {meta.get('stale_reason')}")
        if meta.get("contradiction_topics"):
            lines.append(f"     contradiction_topics: {meta['contradiction_topics']}")
        if meta.get("ambiguity_flags"):
            lines.append(f"     ambiguity_flags: {meta['ambiguity_flags']}")
        if result.matched_terms:
            lines.append(f"     matched_terms: {result.matched_terms}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: run a query against the hybrid retriever."""
    import json

    parser = argparse.ArgumentParser(
        prog="learnforge.retrieval",
        description="Hybrid retrieval over the normalized LearnForge KB.",
    )
    parser.add_argument("--query", required=True, help="user query")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="number of results")
    parser.add_argument("--mode", choices=("hybrid", "semantic", "lexical"), default="hybrid")
    parser.add_argument("--records", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--json", action="store_true", help="emit JSON results")
    args = parser.parse_args(argv)

    retriever = Retriever.from_store(args.records, args.embeddings)
    if args.mode == "semantic":
        results = retriever.search_semantic(args.query, args.k)
    elif args.mode == "lexical":
        results = retriever.search_lexical(args.query, args.k)
    else:
        results = retriever.search(args.query, args.k)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        print(format_results(args.query, results, args.mode))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())