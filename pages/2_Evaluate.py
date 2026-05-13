"""Evaluation page — run the Phase 10 question bank against the knowledge base."""

from __future__ import annotations

import json

import streamlit as st

from src.evaluate import (
    EVALUATION_QUESTIONS,
    check_edge_cases,
    run_evaluation,
    save_evaluation_report,
)
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
    page_title="Evaluate · EU AI Policy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🇪🇺 EU AI Policy\n**Evaluation**")
st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
kb_ok = sidebar_vector_store_status()

st.sidebar.divider()
top_k, _ = sidebar_settings(show_theme_filter=False, default_top_k=5)

st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-title">Options</div>', unsafe_allow_html=True)
run_edge_cases = st.sidebar.checkbox("Run edge-case checks", value=True)
save_report = st.sidebar.checkbox("Save report to disk", value=True)

# ── Main ──────────────────────────────────────────────────────────────────────

page_header(
    icon="📊",
    title="Retrieval Evaluation",
    subtitle=(
        f"Runs {len(EVALUATION_QUESTIONS)} representative questions through the retrieval pipeline "
        "and scores relevance. No language model required."
    ),
)

if not kb_ok:
    no_vector_store_warning()
    st.stop()

# Question preview table
with st.expander("📋 Question bank", expanded=False):
    for q in EVALUATION_QUESTIONS:
        src_badge = (
            '<span class="badge badge-info">scenario</span>'
            if "scenarios" in q["source"] or "script" in q["source"]
            else '<span class="badge badge-info">plan</span>'
        )
        st.markdown(
            f'<div style="margin:6px 0"><strong>{q["id"]}</strong> &nbsp;{src_badge}&nbsp; '
            f'{q["question"]}<br/>'
            f'<span style="color:#7a8ea8;font-size:0.78rem">'
            f'Source: {q["source"]} · Theme: {q["theme_hint"]}</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=False)

if run_btn:
    with st.spinner("Loading models and running evaluation…"):
        try:
            embedding_model = get_embedding_model()
            vector_db = get_vector_db()
        except FileNotFoundError:
            no_vector_store_warning()
            st.stop()

        results = run_evaluation(vector_db, embedding_model, top_k=top_k)

        edge_results = []
        if run_edge_cases:
            edge_results = check_edge_cases(vector_db, embedding_model)

        if save_report:
            save_evaluation_report(results)

    # ── Summary metrics ───────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed_relevance_check)
    failed = len(results) - passed
    avg_sim = sum(r.top_similarity for r in results) / len(results)
    top_sim = max(r.top_similarity for r in results)

    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card success">
            <span class="metric-value">{passed}</span>
            <span class="metric-label">Passed</span>
          </div>
          <div class="metric-card {'danger' if failed else ''}">
            <span class="metric-value">{failed}</span>
            <span class="metric-label">Needs Review</span>
          </div>
          <div class="metric-card highlight">
            <span class="metric-value">{avg_sim:.3f}</span>
            <span class="metric-label">Avg Similarity</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{top_sim:.3f}</span>
            <span class="metric-label">Best Similarity</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eu-divider"></div>', unsafe_allow_html=True)

    # ── Per-question results ───────────────────────────────────────────────────
    st.markdown("### Question Results")
    for result in results:
        status_html = (
            '<span class="badge badge-pass">PASS</span>'
            if result.passed_relevance_check
            else '<span class="badge badge-warn">REVIEW</span>'
        )

        header_html = (
            f'{status_html} &nbsp;<strong>{result.question_id}</strong> — {result.question}'
            f'<span class="sim-badge {sim_class(result.top_similarity)}">'
            f'{result.top_similarity:.3f}</span>'
        )

        with st.expander(
            f"{'✅' if result.passed_relevance_check else '⚠️'} "
            f"{result.question_id} — {result.question[:80]}{'…' if len(result.question) > 80 else ''}",
            expanded=False,
        ):
            st.markdown(header_html, unsafe_allow_html=True)
            st.caption(
                f"Source: {result.question_source} · Theme hint: {result.theme_hint}"
            )
            if result.notes:
                st.info(result.notes)

            # Source cards
            cards_html = "".join(
                source_card_html(
                    {
                        "title": s.title,
                        "doc_id": s.doc_id,
                        "theme": s.theme,
                        "url": s.url,
                        "similarity": s.similarity,
                        "text": s.excerpt,
                    },
                    i,
                    show_excerpt=True,
                )
                for i, s in enumerate(result.retrieved_sources, start=1)
            )
            st.markdown(cards_html, unsafe_allow_html=True)

    # ── Edge-case results ──────────────────────────────────────────────────────
    if edge_results:
        st.markdown('<div class="eu-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Edge-Case Checks")
        for check in edge_results:
            icon = "✅" if check["passed"] else "❌"
            badge = (
                '<span class="badge badge-pass">PASS</span>'
                if check["passed"]
                else '<span class="badge badge-fail">FAIL</span>'
            )
            st.markdown(
                f'{icon} {badge} &nbsp;<code>{check["check"]}</code> — {check["detail"]}',
                unsafe_allow_html=True,
            )

    # ── Download button ────────────────────────────────────────────────────────
    st.markdown('<div class="eu-divider"></div>', unsafe_allow_html=True)
    report_data = {
        "question_count": len(results),
        "passed": passed,
        "failed": failed,
        "avg_similarity": round(avg_sim, 4),
        "results": [
            {
                "id": r.question_id,
                "question": r.question,
                "passed": r.passed_relevance_check,
                "top_similarity": r.top_similarity,
                "notes": r.notes,
                "sources": [
                    {"title": s.title, "doc_id": s.doc_id, "sim": s.similarity}
                    for s in r.retrieved_sources
                ],
            }
            for r in results
        ],
    }
    st.download_button(
        label="⬇️ Download JSON report",
        data=json.dumps(report_data, indent=2, ensure_ascii=False),
        file_name="evaluation_report.json",
        mime="application/json",
    )
