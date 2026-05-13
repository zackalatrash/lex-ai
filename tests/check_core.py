"""Lightweight automated checks for core non-LLM components (Phase 10, Task 10.3).

These checks do not require Ollama to be running. They verify the retrieval
pipeline: chunking, vector search, save/load round-trip, and metadata parsing.

Run with:
    python tests/check_core.py

Exit code is 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.embeddings import EmbeddingModel, normalize_vectors
from src.preprocessing import (
    chunk_words,
    create_chunks_for_document,
    split_words,
)
from src.downloader import (
    clean_optional_string,
    infer_initial_source_type,
    normalize_document_record,
    stable_doc_id,
)
from src.vector_db import VectorDB


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _report(name: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    suffix = f"  — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


# ---------------------------------------------------------------------------
# Check group 1: Chunking (Task 10.3)
# ---------------------------------------------------------------------------

def check_chunking() -> list[bool]:
    print("\n[Chunking]")
    results: list[bool] = []

    # 1a: basic chunk count
    words = list(range(100))  # 100 words
    chunks = chunk_words([str(w) for w in words], chunk_size=30, overlap=10)
    # Expect: ceil((100-10)/(30-10)) = ceil(90/20) = 5 chunks (roughly)
    results.append(_report("chunk count >0", len(chunks) > 0, f"got {len(chunks)} chunks"))

    # 1b: overlap – consecutive chunks must share words at the boundary
    if len(chunks) >= 2:
        end_of_first = set(chunks[0].split()[-10:])
        start_of_second = set(chunks[1].split()[:10])
        shared = end_of_first & start_of_second
        results.append(
            _report(
                "consecutive chunks share overlap words",
                len(shared) > 0,
                f"shared {len(shared)} words",
            )
        )
    else:
        results.append(_report("consecutive chunks share overlap words", False, "need ≥2 chunks"))

    # 1c: last chunk covers the final word
    last_word = str(words[-1])
    results.append(
        _report("last chunk contains last word", last_word in chunks[-1], f"last word: {last_word!r}")
    )

    # 1d: no empty chunks
    results.append(
        _report("no empty chunks", all(c.strip() for c in chunks), "all chunks non-empty")
    )

    # 1e: chunk_size=1, overlap=0 produces N chunks for N words
    single_chunks = chunk_words(["a", "b", "c"], chunk_size=1, overlap=0)
    results.append(
        _report("chunk_size=1 yields one chunk per word", len(single_chunks) == 3, f"got {len(single_chunks)}")
    )

    # 1f: document with extraction_status != "extracted" produces zero chunks
    bad_doc = {"doc_id": "TEST", "text": "hello world", "extraction_status": "too_short"}
    chunks_bad = create_chunks_for_document(bad_doc)
    results.append(
        _report("non-extracted doc yields 0 chunks", len(chunks_bad) == 0, f"got {len(chunks_bad)}")
    )

    # 1g: valid document produces at least one chunk
    long_text = " ".join(["policy"] * 500)
    good_doc = {
        "doc_id": "TEST-GOOD",
        "title": "Test",
        "theme": "test",
        "url": "http://example.com",
        "publisher": "",
        "date": "",
        "citation": "",
        "source_type": "html",
        "local_path": None,
        "text": long_text,
        "extraction_status": "extracted",
    }
    good_chunks = create_chunks_for_document(good_doc)
    results.append(
        _report("valid doc yields ≥1 chunk", len(good_chunks) >= 1, f"got {len(good_chunks)}")
    )

    return results


# ---------------------------------------------------------------------------
# Check group 2: Metadata parser (Task 10.3)
# ---------------------------------------------------------------------------

def check_metadata_parser() -> list[bool]:
    print("\n[Metadata Parser]")
    results: list[bool] = []

    import pandas as pd

    # 2a: missing URL → status=missing_url
    row_missing_url = pd.Series({
        "id": "TEST-01", "title": "Test", "url": float("nan"),
        "publisher": "", "date": "", "theme": "", "keywords": "", "bronvermelding": "",
    })
    rec = normalize_document_record(row_missing_url, 0)
    results.append(_report("missing URL → status=missing_url", rec["status"] == "missing_url"))

    # 2b: NaN values become empty strings, not "nan"
    results.append(
        _report(
            "NaN fields become empty string",
            rec["url"] == "" and rec["title"] == "Test",
            f"url={rec['url']!r}",
        )
    )

    # 2c: clean_optional_string handles NaN
    results.append(
        _report(
            "clean_optional_string(nan) returns ''",
            clean_optional_string(float("nan")) == "",
        )
    )

    # 2d: stable_doc_id uses dataset id when present
    row_with_id = pd.Series({"id": "R99-01"})
    results.append(
        _report(
            "stable_doc_id uses dataset id",
            stable_doc_id(row_with_id, 0) == "R99-01",
        )
    )

    # 2e: stable_doc_id falls back to DOC-XXXX when id is missing
    row_no_id = pd.Series({"id": float("nan")})
    fallback = stable_doc_id(row_no_id, 4)
    results.append(
        _report(
            "stable_doc_id falls back to DOC-XXXX",
            fallback == "DOC-0005",
            f"got {fallback!r}",
        )
    )

    # 2f: PDF URL classified correctly
    results.append(
        _report(
            "PDF URL classified as pdf",
            infer_initial_source_type("https://example.com/doc.pdf") == "pdf",
        )
    )

    # 2g: HTTP URL classified as web
    results.append(
        _report(
            "HTTP URL classified as web",
            infer_initial_source_type("https://example.com/page") == "web",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Check group 3: Embeddings (Task 10.3)
# ---------------------------------------------------------------------------

def check_embeddings() -> list[bool]:
    print("\n[Embeddings]")
    results: list[bool] = []

    model = EmbeddingModel()

    # 3a: empty string returns zero vector
    vec = model.embed_text("", is_query=False)
    results.append(
        _report(
            "empty string → zero vector",
            float(np.linalg.norm(vec)) == 0.0,
            f"norm={np.linalg.norm(vec):.6f}",
        )
    )

    # 3b: self-similarity == 1.0 (normalized)
    v = model.embed_text("AI systems should be transparent.", is_query=False)
    self_sim = float(np.dot(v, v))
    results.append(
        _report(
            "normalized vector: self dot-product ≈ 1.0",
            abs(self_sim - 1.0) < 1e-5,
            f"dot={self_sim:.6f}",
        )
    )

    # 3c: same text embedded twice gives identical vectors
    v1 = model.embed_text("fairness in AI hiring", is_query=False)
    v2 = model.embed_text("fairness in AI hiring", is_query=False)
    results.append(
        _report(
            "same text → identical vectors",
            np.allclose(v1, v2),
        )
    )

    # 3d: query similarity: same text ≫ unrelated text
    query_vec = model.embed_text("transparency obligations for AI", is_query=True)
    related_vec = model.embed_text("AI systems must be transparent and explainable", is_query=False)
    unrelated_vec = model.embed_text("chocolate chip cookie recipe", is_query=False)
    sim_related = float(np.dot(query_vec, related_vec))
    sim_unrelated = float(np.dot(query_vec, unrelated_vec))
    results.append(
        _report(
            "related text has higher similarity than unrelated",
            sim_related > sim_unrelated,
            f"related={sim_related:.4f}, unrelated={sim_unrelated:.4f}",
        )
    )

    # 3e: batch embedding shape matches single embedding
    batch = model.embed_batch(["hello", "world"], is_query=False)
    results.append(
        _report(
            "batch embedding shape is (2, dim)",
            batch.shape == (2, model.vector_dimension),
            f"shape={batch.shape}",
        )
    )

    # 3f: normalize_vectors handles zero vector safely
    zero = np.zeros(4, dtype=np.float32)
    normed = normalize_vectors(zero)
    results.append(
        _report(
            "normalize_vectors(zero) does not divide by zero",
            np.all(normed == 0),
        )
    )

    return results


# ---------------------------------------------------------------------------
# Check group 4: VectorDB save/load round-trip (Task 10.3)
# ---------------------------------------------------------------------------

def check_vector_db_round_trip() -> list[bool]:
    print("\n[VectorDB save/load]")
    results: list[bool] = []

    model = EmbeddingModel()

    sample_chunks = [
        {
            "chunk_id": f"TEST::chunk-{i:04d}",
            "doc_id": "TEST",
            "title": f"Test Document {i}",
            "theme": "test theme",
            "url": f"https://example.com/{i}",
            "publisher": "EU",
            "date": "2024-01-01",
            "citation": "",
            "source_type": "html",
            "local_path": None,
            "chunk_index": i,
            "chunk_word_count": 5,
            "text": f"AI policy document chunk number {i} about transparency and accountability",
        }
        for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db = VectorDB(store_dir=tmp_path)

        # 4a: build does not raise
        try:
            db.build(sample_chunks, model)
            results.append(_report("build() completes without error", True))
        except Exception as exc:
            results.append(_report("build() completes without error", False, str(exc)))
            return results

        # 4b: chunk count matches
        results.append(
            _report(
                "chunk count after build matches input",
                len(db.chunks) == len(sample_chunks),
                f"{len(db.chunks)} == {len(sample_chunks)}",
            )
        )

        # 4c: embedding shape matches
        results.append(
            _report(
                "embedding shape is (N, dim)",
                db.embeddings.shape == (len(sample_chunks), model.vector_dimension),
                f"shape={db.embeddings.shape}",
            )
        )

        # 4d: save/load round-trip preserves chunk count
        db.save()
        db2 = VectorDB(store_dir=tmp_path)
        db2.load()
        results.append(
            _report(
                "save/load preserves chunk count",
                len(db2.chunks) == len(sample_chunks),
                f"saved={len(sample_chunks)}, loaded={len(db2.chunks)}",
            )
        )

        # 4e: save/load preserves embedding shape
        results.append(
            _report(
                "save/load preserves embedding shape",
                db2.embeddings.shape == db.embeddings.shape,
                f"before={db.embeddings.shape}, after={db2.embeddings.shape}",
            )
        )

        # 4f: manifest records correct metadata
        results.append(
            _report(
                "manifest records model name",
                db.manifest.get("embedding_model") == model.model_name,
                f"{db.manifest.get('embedding_model')!r}",
            )
        )

        # 4g: identical query returns the same top chunk both times
        query_vec = model.embed_text("transparency and accountability", is_query=True)
        r1 = db2.search(query_vec, top_k=1)
        r2 = db2.search(query_vec, top_k=1)
        results.append(
            _report(
                "identical query returns the same top chunk",
                r1 and r2 and r1[0]["chunk_id"] == r2[0]["chunk_id"],
                f"chunk_id={r1[0]['chunk_id'] if r1 else 'none'}",
            )
        )

        # 4h: empty DB search returns empty list
        empty_db = VectorDB(store_dir=Path(tmpdir) / "empty")
        empty_result = empty_db.search(query_vec, top_k=5)
        results.append(
            _report("search on empty DB returns []", empty_result == [], f"got {empty_result}")
        )

        # 4i: theme filter returns only matching chunks
        theme_query = model.embed_text("AI policy", is_query=True)
        filtered = db2.search(theme_query, top_k=10, filters={"theme": "nonexistent_theme"})
        results.append(
            _report(
                "theme filter with no matches returns []",
                filtered == [],
                f"got {len(filtered)} results",
            )
        )

        # 4j: top_k=0 returns nothing
        results.append(
            _report(
                "top_k=0 returns []",
                db2.search(query_vec, top_k=0) == [],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("NLP Policy Chatbot — Core Automated Checks (Phase 10.3)")
    print("=" * 60)

    all_results: list[bool] = []
    all_results.extend(check_metadata_parser())
    all_results.extend(check_chunking())
    all_results.extend(check_embeddings())
    all_results.extend(check_vector_db_round_trip())

    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed.")
    if failed == 0:
        print("All checks passed.")
    else:
        print("Some checks failed — see details above.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
