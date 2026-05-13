"""Chatbot and Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai import OpenAIError

from src.config import (
    DEFAULT_MAX_CHUNKS_PER_DOC,
    DEFAULT_RETRIEVAL_TOP_K,
    MAX_CONTEXT_CHARS_PER_CHUNK,
    MAX_HISTORY_TURNS,
    MIN_EVIDENCE_SIMILARITY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    OPENAI_API_KEY_FOR_OLLAMA,
    RETRIEVAL_SIMILARITY_FLOOR,
)
from src.embeddings import EmbeddingModel
from src.vector_db import VectorDB


@dataclass(frozen=True)
class SourceAttribution:
    """De-duplicated source information for one response."""

    doc_id: str
    title: str
    url: str
    theme: str
    citation: str
    best_similarity: float


@dataclass(frozen=True)
class ChatResponse:
    """Structured chatbot response."""

    answer: str
    sources: list[SourceAttribution]
    retrieved_chunks: list[dict[str, Any]]
    prompt: str
    error: str | None = None


class PolicyChatbot:
    """RAG chatbot for EU AI policy documents."""

    def __init__(
        self,
        vector_db: VectorDB | None = None,
        embedding_model: EmbeddingModel | None = None,
        ollama_model: str = OLLAMA_MODEL_NAME,
        ollama_base_url: str = OLLAMA_BASE_URL,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        history_limit: int = MAX_HISTORY_TURNS,
    ):
        self.vector_db = vector_db or VectorDB()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.top_k = top_k
        self.history_limit = history_limit
        self.history: list[dict[str, str]] = []
        self.last_sources: list[SourceAttribution] = []
        self._client = OpenAI(
            base_url=ollama_base_url,
            api_key=OPENAI_API_KEY_FOR_OLLAMA,
            timeout=30.0,
        )

    def load_vector_store(self) -> None:
        """Load the persisted vector store."""
        self.vector_db.load()

    def reset(self) -> None:
        """Clear conversation history and recent sources."""
        self.history.clear()
        self.last_sources.clear()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        theme: str | None = None,
        max_per_doc: int | None = DEFAULT_MAX_CHUNKS_PER_DOC,
    ) -> list[dict[str, Any]]:
        """Embed a user query and retrieve relevant chunks."""
        if not query or not query.strip():
            return []
        query_vector = self.embedding_model.embed_text(query, is_query=True)
        filters = {"theme": theme} if theme else None
        return self.vector_db.search(
            query_vector,
            top_k=top_k or self.top_k,
            filters=filters,
            max_per_doc=max_per_doc,
        )

    def build_messages(
        self, question: str, retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Build the messages list for the Ollama chat completions API.

        History turns are sent as proper user/assistant role messages so the
        model receives a real multi-turn conversation rather than a text blob.
        The system prompt appears exactly once.
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an EU AI policy assistant for an NLP assignment. "
                    "Answer only from the retrieved source excerpts provided in the user's message. "
                    "Ground every claim in the sources and cite them inline as [Source 1], [Source 2], etc. "
                    "If the sources are insufficient, say so explicitly. "
                    "Do not invent policy details, legal obligations, dates, or citations."
                ),
            }
        ]
        for turn in self.history[-self.history_limit:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        context_text = self.format_context(retrieved_chunks)
        messages.append({
            "role": "user",
            "content": f"Retrieved source excerpts:\n{context_text}\n\nQuestion: {question}",
        })
        return messages

    def generate_answer(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str | None]:
        """Call Ollama through the OpenAI-compatible API."""
        try:
            completion = self._client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                temperature=0.2,
            )
            answer = completion.choices[0].message.content or ""
            return answer.strip(), None
        except OpenAIError as error:
            return (
                "I could retrieve relevant source material, but I could not contact the local Ollama model. "
                "Make sure Ollama is running and that the configured model is installed.",
                str(error),
            )
        except Exception as error:
            return (
                "I could retrieve relevant source material, but the local model request failed unexpectedly.",
                str(error),
            )

    def answer(self, question: str, top_k: int | None = None, theme: str | None = None) -> ChatResponse:
        """Retrieve context, call the model, update history, and return a response."""
        if not question or not question.strip():
            return ChatResponse(
                answer="Please enter a question about AI policy or ethics.",
                sources=[],
                retrieved_chunks=[],
                prompt="",
                error="empty_question",
            )

        retrieved_chunks = self.retrieve(question, top_k=top_k, theme=theme, max_per_doc=DEFAULT_MAX_CHUNKS_PER_DOC)
        filtered_chunks = [
            c for c in retrieved_chunks
            if float(c.get("similarity", 0.0)) >= RETRIEVAL_SIMILARITY_FLOOR
        ]

        top_sim = float(filtered_chunks[0].get("similarity", 0.0)) if filtered_chunks else 0.0

        if not filtered_chunks or top_sim < MIN_EVIDENCE_SIMILARITY:
            answer = (
                "The available EU policy documents do not contain sufficient information to "
                "answer this question. This topic may be outside the scope of the knowledge "
                "base, or the question may need to be rephrased using policy terminology."
            )
            self.add_turn(question, answer)
            return ChatResponse(
                answer=answer,
                sources=[],
                retrieved_chunks=filtered_chunks,
                prompt="",
            )

        sources = deduplicate_sources(filtered_chunks)
        self.last_sources = sources

        messages = self.build_messages(question, filtered_chunks)
        answer, error = self.generate_answer(messages)
        self.add_turn(question, answer)
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=filtered_chunks,
            prompt=messages[-1]["content"],
            error=error,
        )

    def add_turn(self, question: str, answer: str) -> None:
        """Add one conversational turn and keep the rolling history bounded."""
        self.history.append({"user": question, "assistant": answer})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]

    def format_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks with source labels and metadata."""
        if not retrieved_chunks:
            return "No relevant source excerpts were retrieved."

        sections: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            text = " ".join(str(chunk.get("text") or "").split())
            if len(text) > MAX_CONTEXT_CHARS_PER_CHUNK:
                text = f"{text[:MAX_CONTEXT_CHARS_PER_CHUNK].rstrip()}..."
            sections.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Title: {chunk.get('title')}",
                        f"Document ID: {chunk.get('doc_id')}",
                        f"Theme: {chunk.get('theme')}",
                        f"URL: {chunk.get('url')}",
                        f"Similarity: {float(chunk.get('similarity', 0.0)):.4f}",
                        f"Excerpt: {text}",
                    ]
                )
            )
        return "\n\n".join(sections)

    def handle_command(self, command: str) -> str | None:
        """Handle chat commands. Returns text for command responses."""
        normalized = command.strip().casefold()
        if normalized == "/reset":
            self.reset()
            return "Conversation history and recent sources were cleared."
        if normalized == "/sources":
            return format_sources(self.last_sources)
        if normalized == "/help":
            return (
                "Commands: /help, /reset, /sources, /exit, /quit\n"
                "Ask a question about AI ethics or EU AI policy to retrieve source-grounded context."
            )
        return None


def deduplicate_sources(chunks: list[dict[str, Any]]) -> list[SourceAttribution]:
    """Merge duplicate sources by document id or URL."""
    by_key: dict[str, SourceAttribution] = {}
    for chunk in chunks:
        key = str(chunk.get("doc_id") or chunk.get("url") or chunk.get("title"))
        similarity = float(chunk.get("similarity", 0.0))
        current = by_key.get(key)
        if current is not None and current.best_similarity >= similarity:
            continue
        by_key[key] = SourceAttribution(
            doc_id=str(chunk.get("doc_id") or ""),
            title=str(chunk.get("title") or ""),
            url=str(chunk.get("url") or ""),
            theme=str(chunk.get("theme") or ""),
            citation=str(chunk.get("citation") or ""),
            best_similarity=similarity,
        )
    return list(by_key.values())


def format_sources(sources: list[SourceAttribution]) -> str:
    """Format source attribution for CLI output."""
    if not sources:
        return "No sources were used for the most recent response."

    lines = ["Sources:"]
    for index, source in enumerate(sources, start=1):
        title = source.title or "Untitled source"
        doc_id = f" ({source.doc_id})" if source.doc_id else ""
        lines.append(f"[{index}] {title}{doc_id}")
        if source.theme:
            lines.append(f"    Theme: {source.theme}")
        if source.url:
            lines.append(f"    URL: {source.url}")
        if source.citation:
            lines.append(f"    Citation: {source.citation}")
        lines.append(f"    Best similarity: {source.best_similarity:.4f}")
    return "\n".join(lines)
