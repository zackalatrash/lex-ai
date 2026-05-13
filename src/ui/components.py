"""Shared UI components for the NLP Policy Chatbot Streamlit interface.

All HTML-rendering helpers and resource-loading wrappers live here so every
page can import them without repeating code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    DEFAULT_RETRIEVAL_TOP_K,
    VECTOR_MANIFEST_PATH,
    VECTOR_STORE_DIR,
    PROCESSED_DOCUMENTS_PATH,
)
from src.ui.styles import inject_styles


# ── Re-export so pages only need one import ──────────────────────────────────
__all__ = [
    "inject_styles",
    "page_header",
    "sidebar_settings",
    "sidebar_vector_store_status",
    "source_card_html",
    "render_source_cards",
    "no_vector_store_warning",
    "get_all_themes",
    "get_embedding_model",
    "get_vector_db",
    "sim_class",
]


# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model():
    """Load and cache the sentence-transformer embedding model."""
    import os
    os.environ.setdefault("USE_TF", "0")
    from src.embeddings import EmbeddingModel
    return EmbeddingModel()


@st.cache_resource(show_spinner="Loading knowledge base…")
def get_vector_db():
    """Load and cache the vector store from disk."""
    from src.vector_db import VectorDB
    db = VectorDB()
    db.load()
    return db


# ── Page header ───────────────────────────────────────────────────────────────

def page_header(icon: str, title: str, subtitle: str) -> None:
    """Render the gold-strip page header banner."""
    inject_styles()
    st.markdown(
        f"""
        <div class="policy-header">
          <span class="policy-header-icon">{icon}</span>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar helpers ───────────────────────────────────────────────────────────

def sidebar_vector_store_status() -> bool:
    """Show knowledge-base status in the sidebar. Returns True when loaded."""
    manifest_path = VECTOR_MANIFEST_PATH
    if not manifest_path.exists():
        st.sidebar.markdown(
            '<span class="status-dot dot-red"></span> Knowledge base not built',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Run the **Build** page to create the vector store.")
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_count = manifest.get("chunk_count", "?")
        model = manifest.get("embedding_model", "unknown")
        created = manifest.get("created_at", "")[:10]
        st.sidebar.markdown(
            f'<span class="status-dot dot-green"></span> Knowledge base ready',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            f"{chunk_count} chunks · {model.split('/')[-1]} · {created}"
        )
        return True
    except Exception:
        st.sidebar.markdown(
            '<span class="status-dot dot-amber"></span> Manifest unreadable',
            unsafe_allow_html=True,
        )
        return False


def sidebar_settings(
    show_theme_filter: bool = True,
    default_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> tuple[int, str | None]:
    """Render retrieval settings in the sidebar. Returns (top_k, theme_filter)."""
    st.sidebar.markdown('<div class="sidebar-section-title">Retrieval Settings</div>', unsafe_allow_html=True)

    top_k = st.sidebar.slider(
        "Top-K results",
        min_value=1,
        max_value=15,
        value=default_top_k,
        help="Number of source chunks retrieved per query.",
    )

    theme_filter: str | None = None
    if show_theme_filter:
        themes = ["(all themes)"] + get_all_themes()
        selected = st.sidebar.selectbox("Filter by theme", options=themes)
        if selected != "(all themes)":
            theme_filter = selected

    return top_k, theme_filter


# ── Similarity helpers ────────────────────────────────────────────────────────

def sim_class(score: float) -> str:
    """Return CSS class name for a similarity score."""
    if score >= 0.70:
        return "sim-high"
    if score >= 0.50:
        return "sim-mid"
    return "sim-low"


# ── Source / chunk card HTML ──────────────────────────────────────────────────

def source_card_html(
    chunk: dict[str, Any],
    index: int,
    show_excerpt: bool = True,
    max_excerpt: int = 320,
) -> str:
    """Return an HTML string for a single source/result card."""
    title = chunk.get("title") or "Untitled"
    doc_id = chunk.get("doc_id") or ""
    theme = chunk.get("theme") or ""
    url = chunk.get("url") or ""
    sim = float(chunk.get("similarity", 0.0))
    text = " ".join(str(chunk.get("text") or "").split())
    excerpt = (text[:max_excerpt] + "…") if len(text) > max_excerpt else text

    sim_html = (
        f'<span class="sim-badge {sim_class(sim)}">{sim:.3f}</span>'
        if sim > 0
        else ""
    )
    excerpt_html = (
        f'<div class="source-card-excerpt">{excerpt}</div>'
        if show_excerpt and excerpt
        else ""
    )
    url_html = (
        f'<div class="source-card-url"><a href="{url}" target="_blank">{url}</a></div>'
        if url
        else ""
    )

    return f"""
    <div class="source-card">
      <div class="source-card-header">
        <span class="source-card-index">[{index}]</span>
        <span class="source-card-title">{title}{sim_html}</span>
      </div>
      <div class="source-card-meta">
        <strong>{doc_id}</strong>
        {f'· <span class="theme-chip">{theme}</span>' if theme else ''}
      </div>
      {url_html}
      {excerpt_html}
    </div>
    """


def render_source_cards(
    chunks: list[dict[str, Any]],
    header: str = "Sources",
    show_excerpt: bool = True,
) -> None:
    """Render a list of chunks as source cards inside a Streamlit expander."""
    if not chunks:
        st.caption("No sources retrieved.")
        return
    with st.expander(f"📚 {header} ({len(chunks)})", expanded=False):
        html = "".join(
            source_card_html(c, i, show_excerpt=show_excerpt)
            for i, c in enumerate(chunks, start=1)
        )
        st.markdown(html, unsafe_allow_html=True)


# ── No-vector-store warning ───────────────────────────────────────────────────

def no_vector_store_warning() -> None:
    """Show a friendly message when the vector store has not been built."""
    st.warning(
        "**Knowledge base not found.**  \n"
        "Go to the **🏗️ Build** page to download documents and create the "
        "vector store before using chat or search.",
        icon="⚠️",
    )


# ── Theme list helper ─────────────────────────────────────────────────────────

def get_all_themes() -> list[str]:
    """Return sorted unique themes from the vector store chunks or documents."""
    # Try vector store chunks first (most up-to-date)
    chunks_path = VECTOR_STORE_DIR / "chunks.jsonl"
    if chunks_path.exists():
        themes: set[str] = set()
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        t = str(obj.get("theme") or "").strip()
                        if t:
                            themes.add(t)
                    except json.JSONDecodeError:
                        pass
        return sorted(themes)

    # Fall back to processed documents
    if PROCESSED_DOCUMENTS_PATH.exists():
        themes = set()
        with PROCESSED_DOCUMENTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        t = str(obj.get("theme") or "").strip()
                        if t:
                            themes.add(t)
                    except json.JSONDecodeError:
                        pass
        return sorted(themes)

    return []
