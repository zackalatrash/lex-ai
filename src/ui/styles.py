"""Centralised CSS injection for the EU AI Policy Chatbot Streamlit UI.

All custom styling lives here so every page shares a consistent look.
Fonts: Playfair Display (headings) · Source Sans 3 (body) · JetBrains Mono (code/IDs).
Palette: deep navy background, EU-blue primary, gold accent (#d4a843).
"""

from __future__ import annotations

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Source+Sans+3:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global typography ───────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}
h1, h2, h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    letter-spacing: -0.02em;
}

/* ── Page header banner ──────────────────────────────────── */
.policy-header {
    background: linear-gradient(135deg, #0d1829 0%, #162444 100%);
    border: 1px solid #2a3a55;
    border-radius: 12px;
    padding: 22px 28px 18px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.policy-header::after {
    content: '★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★';
    position: absolute;
    bottom: 6px;
    right: 20px;
    font-size: 9px;
    color: #d4a843;
    opacity: 0.25;
    letter-spacing: 5px;
}
.policy-header-icon {
    font-size: 1.8rem;
    margin-bottom: 4px;
    display: block;
}
.policy-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.65rem !important;
    color: #e8edf5 !important;
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    line-height: 1.25 !important;
}
.policy-header p {
    color: #7a8ea8;
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.4;
}

/* ── Source / result card ────────────────────────────────── */
.source-card {
    background: #1a2540;
    border: 1px solid #2a3a55;
    border-left: 3px solid #d4a843;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'Source Sans 3', sans-serif;
}
.source-card-header {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 4px;
}
.source-card-index {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #d4a843;
    background: #1a3058;
    border: 1px solid #2a4a7a;
    border-radius: 4px;
    padding: 1px 6px;
    flex-shrink: 0;
    margin-top: 2px;
}
.source-card-title {
    font-weight: 600;
    color: #e8edf5;
    font-size: 0.88rem;
    line-height: 1.35;
}
.source-card-meta {
    color: #7a8ea8;
    font-size: 0.78rem;
    margin: 4px 0 6px 0;
    line-height: 1.4;
}
.source-card-excerpt {
    color: #a0b0c4;
    font-size: 0.81rem;
    line-height: 1.55;
    border-top: 1px solid #2a3a55;
    padding-top: 8px;
    margin-top: 6px;
    font-style: italic;
}
.source-card-url {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4b7fd8;
    word-break: break-all;
    margin-top: 4px;
}

/* ── Similarity badge ────────────────────────────────────── */
.sim-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    border-radius: 4px;
    padding: 1px 7px;
    margin-left: 6px;
    vertical-align: middle;
}
.sim-high { color: #4ade80; background: #0b2d1a; border: 1px solid #22c55e40; }
.sim-mid  { color: #fbbf24; background: #2d1e0a; border: 1px solid #f59e0b40; }
.sim-low  { color: #94a3b8; background: #1a2030; border: 1px solid #33445560; }

/* ── Status badges ───────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
.badge-pass { color: #4ade80; background: #0b2d1a; border: 1px solid #22c55e50; }
.badge-fail { color: #f87171; background: #2d0b0b; border: 1px solid #ef444450; }
.badge-warn { color: #fbbf24; background: #2d1e0a; border: 1px solid #f59e0b50; }
.badge-info { color: #60a5fa; background: #0b1a2d; border: 1px solid #3b82f650; }

/* ── Document cards (Sources page) ──────────────────────── */
.doc-card {
    background: #141d2e;
    border: 1px solid #2a3a55;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease;
}
.doc-card:hover { border-color: #4b7fd8; }
.doc-card-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7a8ea8;
    margin-bottom: 4px;
}
.doc-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.97rem;
    font-weight: 600;
    color: #e8edf5;
    margin-bottom: 6px;
    line-height: 1.3;
}
.doc-card-meta { color: #7a8ea8; font-size: 0.8rem; line-height: 1.6; }
.doc-card-footer {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #2a3a55;
    flex-wrap: wrap;
}

/* ── Theme chip ──────────────────────────────────────────── */
.theme-chip {
    display: inline-block;
    background: #0f1d36;
    border: 1px solid #2a4a7a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #60a5fa;
}

/* ── Metric card ─────────────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #1a2540;
    border: 1px solid #2a3a55;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8edf5;
    line-height: 1;
    display: block;
}
.metric-label {
    font-size: 0.72rem;
    color: #7a8ea8;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: block;
}
.metric-card.highlight .metric-value { color: #d4a843; }
.metric-card.success  .metric-value  { color: #4ade80; }
.metric-card.danger   .metric-value  { color: #f87171; }

/* ── Build log ───────────────────────────────────────────── */
.build-log {
    background: #080d16;
    border: 1px solid #2a3a55;
    border-radius: 8px;
    padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7a8ea8;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* ── Chat bubbles ────────────────────────────────────────── */
[data-testid="stChatMessage"][data-message-author-role="user"] > div {
    background: #162444 !important;
    border: 1px solid #2a4a7a !important;
    border-radius: 10px !important;
}
[data-testid="stChatMessage"][data-message-author-role="assistant"] > div {
    background: #141d2e !important;
    border: 1px solid #2a3a55 !important;
    border-left: 3px solid #4b7fd8 !important;
    border-radius: 10px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0f1c !important;
    border-right: 1px solid #1e2d44 !important;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5a72;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2d44;
}

/* ── Status indicator dot ────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e70; }
.dot-red   { background: #ef4444; box-shadow: 0 0 6px #ef444470; }
.dot-amber { background: #f59e0b; box-shadow: 0 0 6px #f59e0b70; }

/* ── Divider ─────────────────────────────────────────────── */
.eu-divider { border: none; border-top: 1px solid #2a3a55; margin: 14px 0; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0e1420; }
::-webkit-scrollbar-thumb { background: #2a3a55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4b7fd8; }

/* ── Misc ────────────────────────────────────────────────── */
.block-container { padding-top: 1.5rem !important; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
"""


def inject_styles() -> None:
    """Inject Google Fonts and all custom CSS into the current Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
