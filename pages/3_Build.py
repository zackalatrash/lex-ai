"""Build page — download documents and (re)build the vector store from scratch."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import (
    DEFAULT_BUILD_LIMIT,
    DOWNLOAD_REPORT_PATH,
    PROJECT_ROOT,
    VECTOR_MANIFEST_PATH,
)
from src.ui.components import (
    page_header,
    sidebar_vector_store_status,
)
from src.ui.styles import inject_styles

st.set_page_config(
    page_title="Build · EU AI Policy",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🇪🇺 EU AI Policy\n**Build Pipeline**")
st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
sidebar_vector_store_status()

# ── Main ──────────────────────────────────────────────────────────────────────

page_header(
    icon="🏗️",
    title="Build Knowledge Base",
    subtitle=(
        "Download EU policy documents, extract text, create retrieval chunks, "
        "and build the vector store. Run once — results are cached to disk."
    ),
)

# ── Current KB status ─────────────────────────────────────────────────────────

st.markdown("### Current Status")

if VECTOR_MANIFEST_PATH.exists():
    try:
        manifest = json.loads(VECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", manifest.get("chunk_count", "?"))
        c2.metric("Vector dim.", manifest.get("vector_dimension", "?"))
        c3.metric("Model", manifest.get("embedding_model", "?").split("/")[-1])
        c4.metric("Built on", (manifest.get("created_at") or "")[:10] or "?")
    except Exception:
        st.warning("Manifest file found but could not be parsed.")
else:
    st.info("No knowledge base found. Use the controls below to build one.")

# Download report summary
if DOWNLOAD_REPORT_PATH.exists():
    try:
        dl_report = json.loads(DOWNLOAD_REPORT_PATH.read_text(encoding="utf-8"))
        with st.expander("📋 Previous download report"):
            st.json(
                {
                    k: v
                    for k, v in dl_report.items()
                    if k != "failed_urls"
                },
                expanded=False,
            )
            if dl_report.get("failed_urls"):
                st.markdown("**Failed / unsupported URLs:**")
                for item in dl_report["failed_urls"]:
                    st.markdown(
                        f"- `{item.get('doc_id')}` — {item.get('status')} — {item.get('error', '')}"
                    )
    except Exception:
        pass

st.divider()

# ── Build controls ────────────────────────────────────────────────────────────

st.markdown("### Build Controls")

col_a, col_b = st.columns([1, 1])

with col_a:
    limit_mode = st.radio(
        "Document limit",
        options=["All documents (87)", f"Test mode (first {DEFAULT_BUILD_LIMIT})", "Custom limit"],
        index=0,
        horizontal=False,
    )

    limit: int | None = None
    if "Test mode" in limit_mode:
        limit = DEFAULT_BUILD_LIMIT
    elif "Custom" in limit_mode:
        limit = st.number_input(
            "Number of documents",
            min_value=1,
            max_value=87,
            value=10,
            step=1,
        )

with col_b:
    force_rebuild = st.checkbox(
        "Force re-download (ignore cached files)",
        value=False,
        help="Re-downloads all documents even if local copies exist.",
    )
    st.caption(
        "Leave unchecked to skip documents already in `data/raw/` — "
        "much faster for incremental updates."
    )

st.markdown(
    f"**Ready to build:** {'all 87' if limit is None else str(limit)} document(s). "
    f"Force re-download: {'yes' if force_rebuild else 'no'}."
)

build_btn = st.button("🚀 Start Build", type="primary")

# ── Build execution ───────────────────────────────────────────────────────────

if build_btn:
    import os
    os.environ.setdefault("USE_TF", "0")

    from src.downloader import download_documents, parse_dataset
    from src.embeddings import EmbeddingModel
    from src.preprocessing import chunk_documents, extract_and_clean_documents
    from src.vector_db import VectorDB

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    with st.status("Building knowledge base…", expanded=True) as build_status:

        # Step 1: Parse dataset
        st.write("📋 **Step 1/5** — Parsing dataset…")
        try:
            records, summary = parse_dataset(limit=limit)
            st.write(
                f"✅ {summary.rows} rows · {summary.urls} URLs · "
                f"{summary.duplicate_urls} duplicates"
            )
            log(f"Dataset parsed: {summary.rows} rows, {summary.urls} URLs")
        except Exception as exc:
            st.error(f"Dataset parsing failed: {exc}")
            build_status.update(label="Build failed.", state="error")
            st.stop()

        # Step 2: Download
        st.write("⬇️ **Step 2/5** — Downloading source documents…")
        st.caption("This can take several minutes for a full build (rate limits enforced).")
        try:
            records, dl_summary = download_documents(records, force=force_rebuild)
            st.write(
                f"✅ Downloaded: {dl_summary.downloaded} · "
                f"Skipped: {dl_summary.skipped} · "
                f"Failed: {dl_summary.failed} · "
                f"Unsupported: {dl_summary.unsupported}"
            )
            log(
                f"Downloads: {dl_summary.downloaded} ok, "
                f"{dl_summary.failed} failed, {dl_summary.skipped} skipped"
            )
            if dl_summary.failed_urls:
                with st.expander(f"⚠️ {dl_summary.failed} failed URLs"):
                    for item in dl_summary.failed_urls:
                        st.markdown(f"- `{item.get('doc_id')}` — {item.get('error', '')}")
        except Exception as exc:
            st.error(f"Download step failed: {exc}")
            build_status.update(label="Build failed.", state="error")
            st.stop()

        # Step 3: Extract text
        st.write("📝 **Step 3/5** — Extracting and cleaning text…")
        try:
            records, ex_summary = extract_and_clean_documents(records)
            st.write(
                f"✅ Extracted: {ex_summary.extracted} · "
                f"Too short: {ex_summary.too_short} · "
                f"Empty: {ex_summary.empty} · "
                f"Skipped: {ex_summary.skipped}"
            )
            log(f"Extraction: {ex_summary.extracted} ok, {ex_summary.failed} failed")
        except Exception as exc:
            st.error(f"Text extraction failed: {exc}")
            build_status.update(label="Build failed.", state="error")
            st.stop()

        # Step 4: Chunk
        st.write("✂️ **Step 4/5** — Chunking documents…")
        try:
            chunks, ch_summary = chunk_documents(records)
            st.write(
                f"✅ {ch_summary.chunk_count} chunks · "
                f"avg {ch_summary.average_words} words · "
                f"range [{ch_summary.min_words}–{ch_summary.max_words}]"
            )
            log(f"Chunks: {ch_summary.chunk_count} total")
        except Exception as exc:
            st.error(f"Chunking failed: {exc}")
            build_status.update(label="Build failed.", state="error")
            st.stop()

        # Step 5: Embed & build vector store
        st.write("🔢 **Step 5/5** — Building vector store (embedding ~may take ~1 min)…")
        try:
            embedding_model = EmbeddingModel()
            vector_db = VectorDB()
            vector_db.build(chunks, embedding_model)
            vector_db.save()
            st.write(
                f"✅ Vector store saved · {len(chunks)} chunks · "
                f"dim {embedding_model.vector_dimension} · "
                f"model {embedding_model.model_name.split('/')[-1]}"
            )
            log(f"Vector store: {len(chunks)} chunks, dim {embedding_model.vector_dimension}")
        except Exception as exc:
            st.error(f"Embedding / vector store step failed: {exc}")
            build_status.update(label="Build failed.", state="error")
            st.stop()

        build_status.update(label="Build complete! ✅", state="complete")

    # Clear cached resources so Chat and Search reload the new store
    st.cache_resource.clear()

    st.success(
        f"Knowledge base built with **{ch_summary.chunk_count} chunks** from "
        f"**{ex_summary.extracted}** documents. "
        "Navigate to **💬 Chat** or **🔍 Search** to start querying."
    )

    # Show log
    with st.expander("📋 Build log"):
        st.code("\n".join(log_lines), language=None)
