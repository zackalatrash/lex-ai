"""Text extraction and cleaning for downloaded policy documents.

Chunking is added in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import unicodedata

import urllib.parse

from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests

from src.config import CHUNKS_PATH, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, PROCESSED_DOCUMENTS_PATH, PROJECT_ROOT, REQUEST_TIMEOUT_SECONDS, USER_AGENT
from src.utils import project_relative, write_jsonl


MIN_USEFUL_TEXT_CHARS = 500


@dataclass(frozen=True)
class ExtractionSummary:
    """Summary of a text extraction run."""

    total_records: int
    extracted: int
    too_short: int
    empty: int
    failed: int
    skipped: int
    output_path: str
    problem_documents: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "total_records": self.total_records,
            "extracted": self.extracted,
            "too_short": self.too_short,
            "empty": self.empty,
            "failed": self.failed,
            "skipped": self.skipped,
            "output_path": self.output_path,
            "problem_documents": self.problem_documents,
        }


@dataclass(frozen=True)
class ChunkingSummary:
    """Summary of a chunking run."""

    document_count: int
    chunk_count: int
    min_words: int
    max_words: int
    average_words: float
    zero_chunk_documents: list[dict[str, Any]]
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "average_words": self.average_words,
            "zero_chunk_documents": self.zero_chunk_documents,
            "output_path": self.output_path,
        }


def resolve_local_path(local_path: str | None) -> Path | None:
    """Resolve a metadata local path relative to the project root."""
    if not local_path:
        return None
    path = Path(local_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_text(text: str) -> str:
    """Normalize extracted text into readable plain text."""
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\x00", " ")
    cleaned = cleaned.replace("\ufeff", " ")
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"([A-Za-z])-\n\s*([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
    return cleaned.strip()


def extract_pdf_text(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    raw_text = "\n\n".join(page_texts)
    metadata = {
        "extraction_method": "pypdf",
        "page_count": len(reader.pages),
        "raw_text_char_count": len(raw_text),
    }
    return raw_text, metadata


def find_pdf_link(html: str, base_url: str) -> str | None:
    """Return the first absolute PDF URL found in an HTML page, or None."""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if ".pdf" in href.lower():
            return urllib.parse.urljoin(base_url, href)
    return None


def download_pdf_to_path(url: str, dest: Path) -> None:
    """Download a PDF from url and write it to dest."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    response = session.get(url, allow_redirects=True, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    dest.write_bytes(response.content)


def extract_html_text(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract main visible text from an HTML document."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "button",
        ]
    ):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw_text = main.get_text(separator="\n")
    metadata = {
        "extraction_method": "beautifulsoup4",
        "page_count": None,
        "raw_text_char_count": len(raw_text),
    }
    return raw_text, metadata


def mark_extraction_result(record: dict[str, Any], text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Attach cleaned text and extraction status to one document record."""
    updated = dict(record)
    cleaned = clean_text(text)

    updated.update(metadata)
    updated["text"] = cleaned
    updated["text_char_count"] = len(cleaned)

    if not cleaned:
        updated["extraction_status"] = "empty_text"
        updated["extraction_error"] = "No extractable text found; document may be scanned, image-only, or empty."
    elif len(cleaned) < MIN_USEFUL_TEXT_CHARS:
        updated["extraction_status"] = "too_short"
        updated["extraction_error"] = f"Extracted text is shorter than {MIN_USEFUL_TEXT_CHARS} characters."
    else:
        updated["extraction_status"] = "extracted"
        updated["extraction_error"] = None

    return updated


def extract_text_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract and clean text for one downloaded document record."""
    updated = dict(record)
    source_type = str(updated.get("source_type") or "").lower()
    local_path = resolve_local_path(updated.get("local_path"))

    if updated.get("status") not in {"downloaded", "skipped_existing"}:
        updated["text"] = ""
        updated["text_char_count"] = 0
        updated["extraction_status"] = "skipped"
        updated["extraction_error"] = f"Download status is {updated.get('status')}; no local source available."
        return updated

    if local_path is None:
        updated["text"] = ""
        updated["text_char_count"] = 0
        updated["extraction_status"] = "skipped"
        updated["extraction_error"] = "Missing local_path."
        return updated

    if not local_path.exists():
        updated["text"] = ""
        updated["text_char_count"] = 0
        updated["extraction_status"] = "extraction_failed"
        updated["extraction_error"] = f"Local file not found: {project_relative(local_path)}"
        return updated

    try:
        if source_type == "pdf":
            raw_text, metadata = extract_pdf_text(local_path)
        elif source_type == "html":
            raw_text, metadata = extract_html_text(local_path)
            cleaned_preview = clean_text(raw_text)
            if len(cleaned_preview) < MIN_USEFUL_TEXT_CHARS:
                page_url = str(updated.get("final_url") or updated.get("url") or "")
                html_content = local_path.read_text(encoding="utf-8", errors="replace")
                pdf_url = find_pdf_link(html_content, page_url)
                if pdf_url:
                    pdf_path = local_path.with_suffix(".pdf")
                    try:
                        download_pdf_to_path(pdf_url, pdf_path)
                        raw_text, metadata = extract_pdf_text(pdf_path)
                        updated["local_path"] = project_relative(pdf_path)
                        updated["source_type"] = "pdf"
                        metadata["pdf_fallback_url"] = pdf_url
                    except Exception:
                        pass
        else:
            updated["text"] = ""
            updated["text_char_count"] = 0
            updated["extraction_status"] = "skipped"
            updated["extraction_error"] = f"Unsupported source_type for extraction: {source_type}"
            return updated
        return mark_extraction_result(updated, raw_text, metadata)
    except Exception as error:
        updated["text"] = ""
        updated["text_char_count"] = 0
        updated["extraction_status"] = "extraction_failed"
        updated["extraction_error"] = str(error)
        return updated


def build_extraction_summary(
    records: list[dict[str, Any]],
    output_path: Path = PROCESSED_DOCUMENTS_PATH,
) -> ExtractionSummary:
    """Create a compact extraction summary for CLI output."""
    problem_statuses = {"too_short", "empty_text", "extraction_failed", "skipped"}
    problem_documents = [
        {
            "doc_id": record.get("doc_id"),
            "title": record.get("title"),
            "source_type": record.get("source_type"),
            "local_path": record.get("local_path"),
            "extraction_status": record.get("extraction_status"),
            "extraction_error": record.get("extraction_error"),
            "text_char_count": record.get("text_char_count"),
        }
        for record in records
        if record.get("extraction_status") in problem_statuses
    ]

    return ExtractionSummary(
        total_records=len(records),
        extracted=sum(1 for record in records if record.get("extraction_status") == "extracted"),
        too_short=sum(1 for record in records if record.get("extraction_status") == "too_short"),
        empty=sum(1 for record in records if record.get("extraction_status") == "empty_text"),
        failed=sum(1 for record in records if record.get("extraction_status") == "extraction_failed"),
        skipped=sum(1 for record in records if record.get("extraction_status") == "skipped"),
        output_path=project_relative(output_path),
        problem_documents=problem_documents,
    )


def extract_and_clean_documents(
    records: list[dict[str, Any]],
    output_path: Path = PROCESSED_DOCUMENTS_PATH,
) -> tuple[list[dict[str, Any]], ExtractionSummary]:
    """Extract text for all document records and save updated JSONL."""
    processed_records = [extract_text_for_record(record) for record in records]
    write_jsonl(output_path, processed_records)
    summary = build_extraction_summary(processed_records, output_path=output_path)
    return processed_records, summary


def split_words(text: str) -> list[str]:
    """Split text into words for chunking."""
    return re.findall(r"\S+", text or "")


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    return [s for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def chunk_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Create overlapping chunks that respect sentence boundaries.

    Sentences are accumulated until adding the next would exceed chunk_size
    words.  The overlap is approximated by rewinding enough sentences from the
    end of the previous chunk to cover overlap words before starting the next.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not sentences:
        return []

    sent_words = [split_words(s) for s in sentences]
    chunks: list[str] = []
    i = 0

    while i < len(sent_words):
        chunk: list[str] = []
        j = i

        while j < len(sent_words):
            candidate = chunk + sent_words[j]
            if len(candidate) > chunk_size and chunk:
                break
            chunk = candidate
            j += 1

        if not chunk:
            # Single sentence longer than chunk_size — emit it whole
            chunk = sent_words[i]
            j = i + 1

        chunks.append(" ".join(chunk))

        # Rewind from j to find how many sentences cover the overlap
        overlap_acc = 0
        next_i = j
        for k in range(j - 1, i, -1):
            overlap_acc += len(sent_words[k])
            if overlap_acc >= overlap:
                next_i = k
                break

        i = max(i + 1, next_i)

    return chunks


def chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Create overlapping word chunks (kept for backwards compatibility)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Create a stable chunk id from document id and zero-based chunk index."""
    return f"{doc_id}::chunk-{chunk_index:04d}"


def create_chunks_for_document(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Create retrieval chunks for one processed document."""
    if document.get("extraction_status") != "extracted":
        return []

    sentences = split_sentences(str(document.get("text") or ""))
    text_chunks = chunk_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
    chunks: list[dict[str, Any]] = []
    for chunk_index, text in enumerate(text_chunks):
        chunk_words_list = split_words(text)
        if not chunk_words_list:
            continue
        chunks.append(
            {
                "chunk_id": make_chunk_id(str(document.get("doc_id")), chunk_index),
                "doc_id": document.get("doc_id"),
                "title": document.get("title"),
                "theme": document.get("theme"),
                "url": document.get("url"),
                "publisher": document.get("publisher"),
                "date": document.get("date"),
                "citation": document.get("citation"),
                "source_type": document.get("source_type"),
                "local_path": document.get("local_path"),
                "chunk_index": chunk_index,
                "chunk_word_count": len(chunk_words_list),
                "text": text,
            }
        )
    return chunks


def build_chunking_summary(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    output_path: Path = CHUNKS_PATH,
) -> ChunkingSummary:
    """Create quality stats for generated chunks."""
    word_counts = [int(chunk["chunk_word_count"]) for chunk in chunks]
    chunked_doc_ids = {chunk["doc_id"] for chunk in chunks}
    zero_chunk_documents = [
        {
            "doc_id": document.get("doc_id"),
            "title": document.get("title"),
            "extraction_status": document.get("extraction_status"),
            "text_char_count": document.get("text_char_count"),
        }
        for document in documents
        if document.get("doc_id") not in chunked_doc_ids
    ]

    return ChunkingSummary(
        document_count=len(documents),
        chunk_count=len(chunks),
        min_words=min(word_counts) if word_counts else 0,
        max_words=max(word_counts) if word_counts else 0,
        average_words=round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0,
        zero_chunk_documents=zero_chunk_documents,
        output_path=project_relative(output_path),
    )


def chunk_documents(
    documents: list[dict[str, Any]],
    output_path: Path = CHUNKS_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[dict[str, Any]], ChunkingSummary]:
    """Create and persist overlapping chunks for all extracted documents."""
    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks.extend(create_chunks_for_document(document, chunk_size=chunk_size, overlap=overlap))

    write_jsonl(output_path, chunks)
    summary = build_chunking_summary(documents, chunks, output_path=output_path)
    return chunks, summary
