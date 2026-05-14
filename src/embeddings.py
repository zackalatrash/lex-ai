"""Embedding model wrapper for semantic retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# This project uses Sentence Transformers through PyTorch. If TensorFlow/Keras is
# installed in the environment, transformers may auto-import it and fail on
# incompatible Keras versions.
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_CONFIG_KWARGS, EMBEDDING_MODEL_NAME, EMBEDDING_TRUST_REMOTE_CODE


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QWEN3_QUERY_PROMPT_NAME = "query"


@dataclass(frozen=True)
class EmbeddingCheckResult:
    """Result from a lightweight embedding sanity check."""

    vector_dimension: int
    self_similarity: float
    other_similarity: float
    empty_vector_norm: float
    passed: bool


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors row-wise, keeping zero vectors safe."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm == 0:
            return array
        return array / norm

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return array / safe_norms


class EmbeddingModel:
    """Wrapper around a pre-trained Sentence Transformers embedding model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(
            model_name,
            device="mps",
            trust_remote_code=EMBEDDING_TRUST_REMOTE_CODE or None,
            config_kwargs=EMBEDDING_CONFIG_KWARGS or None,
        )
        if hasattr(self.model, "get_embedding_dimension"):
            self.vector_dimension = int(self.model.get_embedding_dimension())
        else:
            self.vector_dimension = int(self.model.get_sentence_embedding_dimension())
        self.uses_bge_query_prefix = "bge-" in model_name.lower() or "arctic-embed" in model_name.lower()
        self.uses_qwen3_query_prompt = "qwen3-embedding" in model_name.lower()

    def prepare_text(self, text: str, is_query: bool = False) -> str:
        """Prepare text before embedding.

        BGE v1.5 models prepend an instruction prefix for retrieval queries.
        Qwen3-Embedding models handle the query prompt via prompt_name at encode time.
        Document chunks never receive a prefix.
        """
        cleaned = text.strip()
        if is_query and self.uses_bge_query_prefix:
            return f"{BGE_QUERY_PREFIX}{cleaned}"
        return cleaned

    def embed_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """Embed a single text string as a NumPy vector."""
        if not text or not text.strip():
            return np.zeros(self.vector_dimension, dtype=np.float32)
        prepared_text = self.prepare_text(text, is_query=is_query)
        encode_kwargs: dict = {"convert_to_numpy": True, "normalize_embeddings": False, "batch_size": EMBEDDING_BATCH_SIZE}
        if is_query and self.uses_qwen3_query_prompt:
            encode_kwargs["prompt_name"] = QWEN3_QUERY_PROMPT_NAME
        vector = self.model.encode(prepared_text, **encode_kwargs)
        vector = np.asarray(vector, dtype=np.float32)
        return normalize_vectors(vector) if self.normalize else vector

    def embed_batch(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Embed a batch of texts as a two-dimensional NumPy array."""
        if not texts:
            return np.empty((0, self.vector_dimension), dtype=np.float32)

        cleaned_texts = [text.strip() if text and text.strip() else "" for text in texts]
        non_empty_indices = [index for index, text in enumerate(cleaned_texts) if text]
        embeddings = np.zeros((len(cleaned_texts), self.vector_dimension), dtype=np.float32)

        if non_empty_indices:
            non_empty_texts = [
                self.prepare_text(cleaned_texts[index], is_query=is_query) for index in non_empty_indices
            ]
            encode_kwargs: dict = {"convert_to_numpy": True, "normalize_embeddings": False, "batch_size": EMBEDDING_BATCH_SIZE}
            if is_query and self.uses_qwen3_query_prompt:
                encode_kwargs["prompt_name"] = QWEN3_QUERY_PROMPT_NAME
            encoded = self.model.encode(non_empty_texts, **encode_kwargs)
            encoded = np.asarray(encoded, dtype=np.float32)
            if self.normalize:
                encoded = normalize_vectors(encoded)
            for row_index, vector in zip(non_empty_indices, encoded):
                embeddings[row_index] = vector

        return embeddings

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine similarity for normalized or unnormalized vectors."""
        left_vector = normalize_vectors(np.asarray(left, dtype=np.float32))
        right_vector = normalize_vectors(np.asarray(right, dtype=np.float32))
        if np.linalg.norm(left_vector) == 0 or np.linalg.norm(right_vector) == 0:
            return 0.0
        return float(np.dot(left_vector, right_vector))


def run_embedding_sanity_check(embedding_model: EmbeddingModel) -> EmbeddingCheckResult:
    """Check basic embedding behavior without depending on the vector database."""
    query = "AI systems should be transparent and accountable."
    same = "AI systems should be transparent and accountable."
    different = "The weather forecast predicts rain tomorrow."

    query_vector = embedding_model.embed_text(query, is_query=True)
    batch_vectors = embedding_model.embed_batch([same, different, ""])
    self_similarity = embedding_model.similarity(query_vector, batch_vectors[0])
    other_similarity = embedding_model.similarity(query_vector, batch_vectors[1])
    empty_vector_norm = float(np.linalg.norm(batch_vectors[2]))

    return EmbeddingCheckResult(
        vector_dimension=embedding_model.vector_dimension,
        self_similarity=round(self_similarity, 6),
        other_similarity=round(other_similarity, 6),
        empty_vector_norm=round(empty_vector_norm, 6),
        passed=self_similarity > other_similarity and empty_vector_norm == 0.0,
    )
