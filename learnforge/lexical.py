"""Lightweight lexical retrieval (BM25) — Milestone M2.

M0 §16 asked for a "light lexical fallback for exact IDs like 'FAQ-08' or the
cancel-page wording 'cancel within 14 days'". This module implements a small,
dependency-free Okapi BM25 index over the same per-record chunks, so lexical
signals can be combined with dense retrieval.

Standard BM25 parameters are used (``k1 = 1.5``, ``b = 0.75``) rather than tuned
values, and no external search library is required.

Each record is indexed as ``source_id + " " + title + " " + chunk_text`` so that
record IDs ("FAQ-08"), titles, product terms, and rare policy phrases are all
lexically searchable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

#: Token pattern: alphanumeric runs, keeping internal hyphens ("14-day", "faq-08").
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: Standard English closed-class stopwords. Removing them keeps the *lexical
#: signal* about content words (e.g. "refund", "browser", "offline") instead of
#: shared interrogatives ("can", "i", "do"), which would otherwise let almost
#: every FAQ record collect a small BM25 score and RRF rank credit.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "between", "both", "but", "by",
        "can", "cannot", "could", "did", "do", "does", "doing", "done", "down",
        "during", "each", "few", "for", "from", "further", "get", "gets", "getting",
        "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more",
        "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
        "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    }
)

#: Standard Okapi BM25 constants (not tuned for this corpus).
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split, and (by default) drop stopwords from ``text``."""
    tokens = TOKEN_RE.findall((text or "").lower())
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def index_text(record: Any) -> str:
    """Return the searchable text for a record (id + title + verbatim content)."""
    get = (lambda key: record[key]) if isinstance(record, dict) else (lambda key: getattr(record, key))
    return f"{get('source_id')} {get('title')} {get('chunk_text')}"


class BM25Index:
    """A small in-memory Okapi BM25 index over a fixed list of documents."""

    def __init__(self, source_ids: Sequence[str], documents: Sequence[str]) -> None:
        if len(source_ids) != len(documents):
            raise ValueError("source_ids and documents must have the same length")
        self.source_ids: list[str] = list(source_ids)
        self.documents: list[str] = list(documents)
        self.k1 = BM25_K1
        self.b = BM25_B

        self._term_freqs: list[Counter[str]] = [Counter(tokenize(doc)) for doc in documents]
        self._lengths: list[int] = [sum(counter.values()) for counter in self._term_freqs]
        self._avg_length: float = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

        # Document frequency per term, for the IDF component.
        self._doc_freq: Counter[str] = Counter()
        for counter in self._term_freqs:
            self._doc_freq.update(counter.keys())

    def __len__(self) -> int:
        return len(self.source_ids)

    def _idf(self, term: str) -> float:
        total = len(self.source_ids)
        freq = self._doc_freq.get(term, 0)
        if freq == 0:
            return 0.0
        # Standard BM25 IDF with +0.5 smoothing; always non-negative.
        return math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> list[float]:
        """Return one BM25 score per indexed document (same order as source_ids)."""
        query_terms = tokenize(query)
        scores = [0.0] * len(self.source_ids)
        if not query_terms:
            return scores

        for term in query_terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for index, counter in enumerate(self._term_freqs):
                freq = counter.get(term, 0)
                if not freq:
                    continue
                length = self._lengths[index] or 1
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / (self._avg_length or 1.0))
                )
                scores[index] += idf * (freq * (self.k1 + 1.0)) / denominator
        return scores

    def matching_terms(self, query: str) -> set[str]:
        """Return the query terms that occur in at least one document."""
        return {term for term in tokenize(query) if self._doc_freq.get(term, 0) > 0}


def build_index(records: Sequence[Any]) -> BM25Index:
    """Build a :class:`BM25Index` over M1 records (dicts or ``Record`` objects)."""
    return BM25Index(
        [r["source_id"] if isinstance(r, dict) else r.source_id for r in records],
        [index_text(r) for r in records],
    )