"""Sources browser — explore all documents in the knowledge base."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import streamlit as st

from src.config import PROCESSED_DOCUMENTS_PATH, VECTOR_STORE_DIR
from src.ui.components import (
    no_vector_store_warning,
    page_header,
    sidebar_vector_store_status,
)
from src.ui.styles import inject_styles

st.set_page_config(
    page_title="Sources · EU AI Policy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🇪🇺 EU AI Policy\n**Sources**")
st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
kb_ok = sidebar_vector_store_status()

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading document index…")
def load_documents() -> list[dict]:
    """Load processed document metadata from JSONL."""
    path = PROCESSED_DOCUMENTS_PATH
    if not path.exists():
        return []
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return docs


@st.cache_data(show_spinner="Counting chunks per document…")
def chunk_counts_by_doc() -> dict[str, int]:
    """Return a mapping of doc_id → chunk count from the vector store."""
    chunks_path = VECTOR_STORE_DIR / "chunks.jsonl"
    counts: dict[str, int] = defaultdict(int)
    if not chunks_path.exists():
        return counts
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    doc_id = obj.get("doc_id")
                    if doc_id:
                        counts[doc_id] += 1
                except json.JSONDecodeError:
                    pass
    return dict(counts)


# ── Main ──────────────────────────────────────────────────────────────────────

page_header(
    icon="📚",
    title="Knowledge Base Browser",
    subtitle=(
        "Browse all EU policy documents in the knowledge base. "
        "Filter by theme or search by title to explore the sources."
    ),
)

if not kb_ok:
    no_vector_store_warning()
    st.stop()

docs = load_documents()
chunk_counts = chunk_counts_by_doc()

if not docs:
    st.info("No processed documents found. Run the **Build** page first.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────

all_themes = sorted({str(d.get("theme") or "").strip() for d in docs if d.get("theme")})
all_statuses = sorted({str(d.get("extraction_status") or "") for d in docs})

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    search_text = st.text_input("🔎 Search by title or keyword", placeholder="e.g. healthcare")
with col_f2:
    theme_filter = st.selectbox("Filter by theme", options=["(all themes)"] + all_themes)
with col_f3:
    status_filter = st.selectbox(
        "Filter by status",
        options=["(all statuses)"] + all_statuses,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────

filtered = docs

if search_text.strip():
    q = search_text.strip().lower()
    filtered = [
        d for d in filtered
        if q in str(d.get("title") or "").lower()
        or q in str(d.get("keywords") or "").lower()
        or q in str(d.get("publisher") or "").lower()
    ]

if theme_filter != "(all themes)":
    filtered = [d for d in filtered if str(d.get("theme") or "") == theme_filter]

if status_filter != "(all statuses)":
    filtered = [
        d for d in filtered
        if str(d.get("extraction_status") or "") == status_filter
    ]

# ── Summary metrics ───────────────────────────────────────────────────────────

total_chunks = sum(chunk_counts.get(d.get("doc_id", ""), 0) for d in filtered)
extracted_count = sum(1 for d in filtered if d.get("extraction_status") == "extracted")

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-card highlight">
        <span class="metric-value">{len(filtered)}</span>
        <span class="metric-label">Documents shown</span>
      </div>
      <div class="metric-card">
        <span class="metric-value">{len(docs)}</span>
        <span class="metric-label">Total in dataset</span>
      </div>
      <div class="metric-card success">
        <span class="metric-value">{extracted_count}</span>
        <span class="metric-label">Fully extracted</span>
      </div>
      <div class="metric-card">
        <span class="metric-value">{total_chunks}</span>
        <span class="metric-label">Chunks (filtered)</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not filtered:
    st.info("No documents match the current filters.")
    st.stop()

st.markdown('<div class="eu-divider"></div>', unsafe_allow_html=True)

# ── Pagination ────────────────────────────────────────────────────────────────

PAGE_SIZE = 15
total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
page_num = st.number_input(
    f"Page (1–{total_pages})", min_value=1, max_value=total_pages, value=1, step=1
)
page_docs = filtered[(page_num - 1) * PAGE_SIZE : page_num * PAGE_SIZE]

st.caption(
    f"Showing {(page_num-1)*PAGE_SIZE + 1}–"
    f"{min(page_num*PAGE_SIZE, len(filtered))} of {len(filtered)} documents"
)

# ── Document cards ────────────────────────────────────────────────────────────

_STATUS_DOT = {
    "extracted": '<span class="status-dot dot-green"></span>',
    "too_short": '<span class="status-dot dot-amber"></span>',
    "empty_text": '<span class="status-dot dot-red"></span>',
    "extraction_failed": '<span class="status-dot dot-red"></span>',
    "skipped": '<span class="status-dot dot-amber"></span>',
}

for doc in page_docs:
    doc_id = str(doc.get("doc_id") or "")
    title = str(doc.get("title") or "Untitled")
    theme = str(doc.get("theme") or "")
    publisher = str(doc.get("publisher") or "")
    date = str(doc.get("date") or "")
    url = str(doc.get("url") or "")
    status = str(doc.get("extraction_status") or "")
    chunks = chunk_counts.get(doc_id, 0)
    char_count = int(doc.get("text_char_count") or 0)

    dot_html = _STATUS_DOT.get(status, "")
    theme_html = f'<span class="theme-chip">{theme}</span>' if theme else ""
    chunks_html = (
        f'<span class="badge badge-info">{chunks} chunks</span>'
        if chunks > 0
        else '<span class="badge badge-warn">0 chunks</span>'
    )
    status_html = f'<span class="badge badge-{"pass" if status == "extracted" else "warn" if status in ("too_short","skipped") else "fail"}">{status}</span>'
    url_html = (
        f'<a href="{url}" target="_blank" style="color:#4b7fd8;font-size:0.75rem;font-family:\'JetBrains Mono\',monospace;word-break:break-all">{url[:90]}{"…" if len(url) > 90 else ""}</a>'
        if url
        else ""
    )

    st.markdown(
        f"""
        <div class="doc-card">
          <div class="doc-card-id">{dot_html}{doc_id}</div>
          <div class="doc-card-title">{title}</div>
          <div class="doc-card-meta">
            {f'Publisher: <strong>{publisher}</strong>' if publisher else ''}
            {f' · {date}' if date else ''}
            {f' · {char_count:,} chars' if char_count else ''}
          </div>
          {url_html}
          <div class="doc-card-footer">
            {theme_html}
            {status_html}
            {chunks_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
