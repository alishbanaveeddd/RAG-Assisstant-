"""Local embedding model + on-disk embedding store (Milestone M2).

M0 §16 proposed an embedding model "from the same provider family" (i.e. an
external embeddings endpoint). **M2 explicitly requires a local model with no
external embedding API and no API key.** Following the M2 directive (and M0 §5
NFR-4, "prefer local / no external calls"), this module runs a small
sentence-transformers model locally and never contacts an embedding service.

The model cache lives outside the repository (``~/.cache/huggingface``); no model
or cache files are committed.

Embeddings are L2-normalized, so cosine similarity is a plain dot product and
retrieval is a deterministic matrix multiplication.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional, Sequence

import numpy as np

# Force local-only behaviour *before* sentence-transformers/transformers import,
# unless the user explicitly opts in to downloading a model.
if os.environ.get("LEARNFORGE_ALLOW_MODEL_DOWNLOAD") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

#: Locally-runnable embedding model (already present in this environment's cache).
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_STORE_VERSION = "m2"
DEFAULT_EMBEDDINGS_PATH = os.path.join("data", "processed", "kb_embeddings.json")
DEFAULT_RECORDS_PATH = os.path.join("data", "processed", "kb_records.json")

_MODEL_CACHE: dict[tuple[str, bool], "LocalEmbedder"] = {}


class LocalEmbedder:
    """Thin wrapper around a locally-run sentence-transformers model."""

    def __init__(
        self, model_name: str = DEFAULT_MODEL_NAME, *, local_files_only: bool = True
    ) -> None:
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Imported lazily so the rest of the package never pays the torch import
        # cost, and so tests can inject a lightweight stub instead.
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(
                model_name, device="cpu", local_files_only=local_files_only
            )
        except TypeError:  # signature without local_files_only
            self._model = SentenceTransformer(model_name, device="cpu")

        self.model_name = model_name
        self.local_files_only = local_files_only
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an L2-normalized ``float32`` matrix (n, dim)."""
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(
    model_name: str = DEFAULT_MODEL_NAME, *, local_files_only: bool = True
) -> LocalEmbedder:
    """Return a cached :class:`LocalEmbedder` (the model loads once per process)."""
    key = (model_name, local_files_only)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = LocalEmbedder(model_name, local_files_only=local_files_only)
    return _MODEL_CACHE[key]


def build_embedding_store(
    records: Sequence[Any], embedder: Optional[LocalEmbedder] = None
) -> dict[str, Any]:
    """Embed every record's verbatim ``chunk_text`` and build the store document.

    Embedding the whole record (heading included) matches M0 §16: semantic search
    runs over "per-record chunks" (one record == one chunk).
    """
    embedder = embedder or get_embedder()

    def _get(record: Any, key: str) -> Any:
        return record[key] if isinstance(record, dict) else getattr(record, key)

    source_ids = [_get(record, "source_id") for record in records]
    texts = [_get(record, "chunk_text") for record in records]
    if texts:
        matrix = embedder.encode(texts)
    else:
        matrix = np.zeros((0, embedder.dimension), dtype=np.float32)

    return {
        "embedding_store_version": EMBEDDING_STORE_VERSION,
        "embedding_model": embedder.model_name,
        "dimension": embedder.dimension,
        "normalized": True,
        "metric": "cosine",
        "record_order": source_ids,
        "vectors": {
            source_id: [float(value) for value in matrix[index]]
            for index, source_id in enumerate(source_ids)
        },
    }


def save_embedding_store(document: dict[str, Any], path: str = DEFAULT_EMBEDDINGS_PATH) -> str:
    """Write the embedding store to ``path`` (creating directories)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return path


def load_embedding_store(path: str = DEFAULT_EMBEDDINGS_PATH) -> dict[str, Any]:
    """Read the embedding store from ``path``."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: str = DEFAULT_RECORDS_PATH) -> list[dict[str, Any]]:
    """Read the M1 record document and return its ``records`` list."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["records"]


def store_matrix(store: dict[str, Any], source_ids: Sequence[str]) -> np.ndarray:
    """Return the ``(len(source_ids), dimension)`` matrix in the given id order."""
    vectors = store.get("vectors", {})
    missing = [source_id for source_id in source_ids if source_id not in vectors]
    if missing:
        raise KeyError(f"embedding store is missing vectors for: {missing}")
    return np.asarray([vectors[source_id] for source_id in source_ids], dtype=np.float32)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: build (or rebuild) the local embedding store."""
    parser = argparse.ArgumentParser(
        prog="learnforge.embed",
        description="Embed the normalized KB records with a local model.",
    )
    parser.add_argument("--records", default=DEFAULT_RECORDS_PATH, help="M1 records JSON")
    parser.add_argument("--out", default=DEFAULT_EMBEDDINGS_PATH, help="embedding store path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="local model name")
    args = parser.parse_args(argv)

    records = load_records(args.records)
    embedder = get_embedder(args.model)
    store = build_embedding_store(records, embedder)
    out_path = save_embedding_store(store, args.out)

    print(f"Embedded {len(records)} records with {embedder.model_name} (dim {embedder.dimension})")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())