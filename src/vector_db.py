"""Custom NumPy-backed vector database for retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    MMR_FETCH_K_MULTIPLIER,
    MMR_LAMBDA,
    VECTOR_CHUNKS_PATH,
    VECTOR_EMBEDDINGS_PATH,
    VECTOR_MANIFEST_PATH,
    VECTOR_STORE_DIR,
)
from src.embeddings import EmbeddingModel, normalize_vectors
from src.utils import ensure_directory, read_jsonl, write_json, write_jsonl


class VectorDB:
    """Persisted vector store with cosine similarity search."""

    def __init__(self, store_dir: str | Path = VECTOR_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.embeddings_path = self.store_dir / VECTOR_EMBEDDINGS_PATH.name
        self.chunks_path = self.store_dir / VECTOR_CHUNKS_PATH.name
        self.manifest_path = self.store_dir / VECTOR_MANIFEST_PATH.name
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.chunks: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        """Return whether the store has chunk metadata and embeddings loaded."""
        return bool(self.chunks) and self.embeddings.size > 0

    def build(self, chunks: list[dict[str, Any]], embedding_model: EmbeddingModel) -> None:
        """Build embeddings and metadata from retrieval chunks."""
        texts = [str(chunk.get("text") or "") for chunk in chunks]
        embeddings = embedding_model.embed_batch(texts, is_query=False)
        self.embeddings = normalize_vectors(embeddings).astype(np.float32)
        self.chunks = [dict(chunk) for chunk in chunks]
        self.manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": embedding_model.model_name,
            "vector_dimension": embedding_model.vector_dimension,
            "chunk_count": len(self.chunks),
            "embedding_shape": list(self.embeddings.shape),
            "embeddings_file": self.embeddings_path.name,
            "chunks_file": self.chunks_path.name,
            "normalized": True,
        }

    def save(self) -> None:
        """Persist embeddings, chunk metadata, and manifest."""
        ensure_directory(self.store_dir)
        np.save(self.embeddings_path, self.embeddings)
        write_jsonl(self.chunks_path, self.chunks)
        write_json(self.manifest_path, self.manifest)

    def load(self) -> None:
        """Load an existing vector store without recomputing embeddings."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {self.embeddings_path}")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks file: {self.chunks_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest file: {self.manifest_path}")

        self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        self.chunks = read_jsonl(self.chunks_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _matching_indices(self, filters: dict[str, Any] | None = None) -> np.ndarray:
        """Return candidate indices after applying optional metadata filters."""
        if not self.chunks:
            return np.array([], dtype=np.int64)
        if not filters:
            return np.arange(len(self.chunks), dtype=np.int64)

        matching: list[int] = []
        for index, chunk in enumerate(self.chunks):
            include = True
            for key, expected_value in filters.items():
                if expected_value in {None, ""}:
                    continue
                actual_value = chunk.get(key)
                if str(actual_value).casefold() != str(expected_value).casefold():
                    include = False
                    break
            if include:
                matching.append(index)
        return np.array(matching, dtype=np.int64)

    def _mmr_select(
        self,
        candidate_indices: np.ndarray,
        scores: np.ndarray,
        result_count: int,
        mmr_lambda: float,
        max_per_doc: int | None,
    ) -> list[int]:
        """Greedy MMR selection over a pre-scored candidate pool.

        Iteratively picks the candidate that maximises:
            mmr_lambda × sim(chunk, query) − (1 − mmr_lambda) × max_sim(chunk, selected)

        max_per_doc (title-keyed) is enforced inside the loop so rejected
        candidates are skipped rather than blocking the search entirely.

        Returns a list of positions (into the candidate arrays, not chunk indices).
        """
        selected: list[int] = []
        selected_vecs: list[np.ndarray] = []
        doc_chunk_counts: dict[str, int] = {}
        remaining = list(range(len(scores)))

        while len(selected) < result_count and remaining:
            if not selected_vecs:
                best_pos = max(remaining, key=lambda i: scores[i])
            else:
                sel_mat = np.stack(selected_vecs)
                best_mmr, best_pos = -np.inf, -1
                for i in remaining:
                    relevance = float(scores[i])
                    redundancy = float(
                        (self.embeddings[int(candidate_indices[i])] @ sel_mat.T).max()
                    )
                    mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                    if mmr_score > best_mmr:
                        best_mmr, best_pos = mmr_score, i

            if max_per_doc is not None:
                chunk = self.chunks[int(candidate_indices[best_pos])]
                title = (chunk.get("title") or "").strip().lower()
                key = title if title else str(chunk.get("doc_id") or "")
                if doc_chunk_counts.get(key, 0) >= max_per_doc:
                    remaining.remove(best_pos)
                    continue
                doc_chunk_counts[key] = doc_chunk_counts.get(key, 0) + 1

            selected.append(best_pos)
            selected_vecs.append(self.embeddings[int(candidate_indices[best_pos])].copy())
            remaining.remove(best_pos)

        return selected

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        filters: dict[str, Any] | None = None,
        max_per_doc: int | None = None,
        use_mmr: bool = True,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks by cosine similarity, with optional MMR reranking.

        When use_mmr=True (default), a candidate pool of top_k × MMR_FETCH_K_MULTIPLIER
        chunks is first selected by raw similarity, then MMR greedily picks the final
        top_k results balancing relevance and semantic diversity.

        max_per_doc caps chunks per source document (title-keyed) and is enforced
        inside both the MMR loop and the plain fallback path.
        """
        if top_k <= 0 or not self.chunks or self.embeddings.size == 0:
            return []

        candidate_indices = self._matching_indices(filters)
        if candidate_indices.size == 0:
            return []

        query = normalize_vectors(np.asarray(query_vector, dtype=np.float32))
        if query.ndim != 1 or np.linalg.norm(query) == 0:
            return []

        candidate_embeddings = normalize_vectors(self.embeddings[candidate_indices])
        scores = candidate_embeddings @ query
        result_count = min(top_k, len(scores))

        if use_mmr:
            fetch_k = min(top_k * MMR_FETCH_K_MULTIPLIER, len(scores))
            pool_positions = np.argsort(scores)[::-1][:fetch_k]
            pool_indices = candidate_indices[pool_positions]
            pool_scores = scores[pool_positions]

            selected_positions = self._mmr_select(
                candidate_indices=pool_indices,
                scores=pool_scores,
                result_count=result_count,
                mmr_lambda=mmr_lambda,
                max_per_doc=max_per_doc,
            )

            results: list[dict[str, Any]] = []
            for pos in selected_positions:
                chunk_index = int(pool_indices[pos])
                result = dict(self.chunks[chunk_index])
                result["similarity"] = float(pool_scores[pos])
                results.append(result)
            return results

        # Plain fallback: sorted top-k with max_per_doc cap
        sorted_positions = np.argsort(scores)[::-1]
        results = []
        doc_chunk_counts: dict[str, int] = {}

        for position in sorted_positions:
            if len(results) >= result_count:
                break
            chunk_index = int(candidate_indices[position])
            chunk = self.chunks[chunk_index]

            if max_per_doc is not None:
                title = (chunk.get("title") or "").strip().lower()
                key = title if title else str(chunk.get("doc_id") or "")
                if doc_chunk_counts.get(key, 0) >= max_per_doc:
                    continue
                doc_chunk_counts[key] = doc_chunk_counts.get(key, 0) + 1

            result = dict(chunk)
            result["similarity"] = float(scores[position])
            results.append(result)

        return results
