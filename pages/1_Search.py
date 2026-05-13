"""Vector Search page — search the knowledge base directly, without generation."""

from __future__ import annotations

import streamlit as st

from src.config import DEFAULT_MAX_CHUNKS_PER_DOC
from src.ui.components import (
    get_embedding_model,
    get_vector_db,
    no_vector_store_warning,
    page_header,
    sidebar_settings,
    sidebar_vector_store_status,
    sim_class,
    source_card_html,
)

st.set_page_config(
    page_title="Search · EU AI Policy",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🇪🇺 EU AI Policy\n**Vector Search**")
st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
kb_ok = sidebar_vector_store_status()

st.sidebar.divider()
top_k, theme = sidebar_settings(show_theme_filter=True, default_top_k=8)

st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Display</div>', unsafe_allow_html=True)
show_excerpts = st.sidebar.checkbox("Show text excerpts", value=True)
min_sim = st.sidebar.slider(
    "Min. similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Hide results below this cosine similarity score.",
)

# ── Main ──────────────────────────────────────────────────────────────────────

page_header(
    icon="🔍",
    title="Vector Search",
    subtitle=(
        "Search the EU AI policy knowledge base directly. "
        "Results are ranked by semantic similarity — no language model required."
    ),
)

if not kb_ok:
    no_vector_store_warning()
    st.stop()

# Query input
query = st.text_input(
    "Search query",
    placeholder="e.g. transparency obligations for high-risk AI systems",
    key="search_query",
)

col1, col2 = st.columns([1, 5])
run_search = col1.button("🔍 Search", type="primary", use_container_width=True)
col2.empty()

if run_search and not query.strip():
    st.warning("Please enter a search query.")
    st.stop()

if run_search and query.strip():
    with st.spinner("Embedding query and searching…"):
        try:
            embedding_model = get_embedding_model()
            vector_db = get_vector_db()
        except FileNotFoundError:
            no_vector_store_warning()
            st.stop()

        query_vector = embedding_model.embed_text(query.strip(), is_query=True)
        filters = {"theme": theme} if theme else None
        results = vector_db.search(query_vector, top_k=top_k, filters=filters, max_per_doc=DEFAULT_MAX_CHUNKS_PER_DOC)

    # Filter by minimum similarity
    results = [r for r in results if float(r.get("similarity", 0)) >= min_sim]

    if not results:
        st.info("No results found. Try a different query or lower the similarity threshold.")
        st.stop()

    # ── Summary row ──────────────────────────────────────────────────────────
    top_sim = float(results[0].get("similarity", 0))
    avg_sim = sum(float(r.get("similarity", 0)) for r in results) / len(results)
    unique_docs = len({r.get("doc_id") for r in results})

    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card highlight">
            <span class="metric-value">{len(results)}</span>
            <span class="metric-label">Results</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{top_sim:.3f}</span>
            <span class="metric-label">Top Similarity</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{avg_sim:.3f}</span>
            <span class="metric-label">Avg Similarity</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{unique_docs}</span>
            <span class="metric-label">Unique Docs</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="eu-divider"></div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    for i, chunk in enumerate(results, start=1):
        st.markdown(
            source_card_html(chunk, i, show_excerpt=show_excerpts),
            unsafe_allow_html=True,
        )
