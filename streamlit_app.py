"""EU AI Policy Chatbot — Streamlit entry point (Chat page).

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from src.chat import ChatResponse, PolicyChatbot, format_sources
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.ui.components import (
    get_embedding_model,
    get_vector_db,
    no_vector_store_warning,
    page_header,
    render_source_cards,
    sidebar_settings,
    sidebar_vector_store_status,
)

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EU AI Policy Chatbot",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_session() -> None:
    """Initialise session-state keys used by the chat page."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "kb_available" not in st.session_state:
        st.session_state.kb_available = False


def _get_chatbot(top_k: int) -> PolicyChatbot | None:
    """Return a session-scoped PolicyChatbot, loading models on first call."""
    if not st.session_state.kb_available:
        return None

    if st.session_state.chatbot is None:
        try:
            embedding_model = get_embedding_model()
            vector_db = get_vector_db()
            st.session_state.chatbot = PolicyChatbot(
                vector_db=vector_db,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        except FileNotFoundError:
            st.session_state.kb_available = False
            return None

    return st.session_state.chatbot


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[int, str | None]:
    st.sidebar.markdown(
        "## 🇪🇺 EU AI Policy\n**Chatbot**",
    )
    st.sidebar.divider()

    st.sidebar.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
    kb_ok = sidebar_vector_store_status()
    st.session_state.kb_available = kb_ok

    st.sidebar.divider()

    top_k, theme = sidebar_settings(show_theme_filter=True)

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)

    if st.sidebar.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.reset()
        st.rerun()

    msg_count = len(st.session_state.messages)
    if msg_count:
        turns = msg_count // 2
        st.sidebar.caption(f"{turns} turn{'s' if turns != 1 else ''} in this session.")

    # Recent sources (from last assistant turn)
    last_sources = []
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and msg.get("sources"):
            last_sources = msg["sources"]
            break

    if last_sources:
        st.sidebar.divider()
        st.sidebar.markdown('<div class="sidebar-section-title">Last Sources</div>', unsafe_allow_html=True)
        for src in last_sources[:4]:
            title = src.get("title") or "Untitled"
            sim = src.get("best_similarity", 0.0)
            st.sidebar.caption(f"• {title[:48]}{'…' if len(title) > 48 else ''} `{sim:.3f}`")

    return top_k, theme


# ── Chat rendering ────────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single chat message with optional source cards."""
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("chunks"):
            render_source_cards(msg["chunks"], header="Sources used")
        if role == "assistant" and msg.get("error"):
            st.caption(f"⚠️ Model error: {msg['error']}")


def _handle_query(
    user_input: str,
    chatbot: PolicyChatbot,
    top_k: int,
    theme: str | None,
) -> None:
    """Process a user message, call the RAG pipeline, update session state."""
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating response…"):
            response: ChatResponse = chatbot.answer(user_input, top_k=top_k, theme=theme)

        st.markdown(response.answer)
        if response.retrieved_chunks:
            render_source_cards(response.retrieved_chunks, header="Sources used")
        if response.error:
            st.caption(f"⚠️ Model error: {response.error}")

    # Persist assistant turn — store raw chunks for re-rendering
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "chunks": response.retrieved_chunks,
        "sources": [
            {
                "title": s.title,
                "doc_id": s.doc_id,
                "url": s.url,
                "theme": s.theme,
                "best_similarity": s.best_similarity,
            }
            for s in response.sources
        ],
        "error": response.error,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    top_k, theme = _render_sidebar()

    page_header(
        icon="💬",
        title="EU AI Policy Chatbot",
        subtitle=(
            "Ask questions about AI ethics, transparency, fairness, and EU regulation. "
            "Every answer is grounded in official EU policy documents."
        ),
    )

    # Warn if KB not available
    if not st.session_state.kb_available:
        no_vector_store_warning()
        return

    chatbot = _get_chatbot(top_k=top_k)
    if chatbot is None:
        no_vector_store_warning()
        return

    # Replay conversation history
    for msg in st.session_state.messages:
        _render_message(msg)

    # Handle special commands + new user input
    user_input = st.chat_input(
        "Ask about EU AI policy… (or type /help, /reset, /sources)",
        key="chat_input",
    )

    if user_input:
        stripped = user_input.strip()
        if not stripped:
            return

        # Command handling
        cmd_response = chatbot.handle_command(stripped)
        if cmd_response is not None:
            if stripped.casefold() == "/reset":
                st.session_state.messages = []
                st.rerun()
            st.session_state.messages.append({"role": "assistant", "content": cmd_response})
            with st.chat_message("assistant"):
                st.markdown(cmd_response)
            return

        _handle_query(stripped, chatbot, top_k=top_k, theme=theme)

    # Example questions when conversation is empty
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("**Try asking:**")
        examples = [
            "Who is responsible when an AI system causes harm?",
            "What transparency obligations exist for AI systems?",
            "How does algorithmic bias affect fairness in hiring?",
            "What safeguards exist for AI use in healthcare?",
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(example, use_container_width=True, key=f"ex_{i}"):
                _handle_query(example, chatbot, top_k=top_k, theme=theme)
                st.rerun()


if __name__ == "__main__":
    main()
