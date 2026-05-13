"""Command-line entry point for the NLP Policy Chatbot."""

from __future__ import annotations

import argparse

from src.config import (
    APP_NAME,
    CHUNKS_PATH,
    DEFAULT_BUILD_LIMIT,
    DEFAULT_MAX_CHUNKS_PER_DOC,
    DEFAULT_RETRIEVAL_TOP_K,
    DOWNLOAD_REPORT_PATH,
    PROCESSED_DOCUMENTS_PATH,
    VECTOR_CHUNKS_PATH,
    VECTOR_EMBEDDINGS_PATH,
    VECTOR_MANIFEST_PATH,
)
from src.chat import PolicyChatbot, format_sources
from src.downloader import download_documents, parse_dataset
from src.embeddings import EmbeddingModel, run_embedding_sanity_check
from src.evaluate import (
    EVALUATION_QUESTIONS,
    check_edge_cases,
    format_edge_case_report,
    format_evaluation_report,
    run_evaluation,
    save_evaluation_report,
)
from src.preprocessing import chunk_documents, extract_and_clean_documents
from src.utils import project_relative
from src.vector_db import VectorDB


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description="NLP Policy Chatbot command-line interface.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the processed documents and vector store.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild outputs even when cached files exist.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the chatbot interface.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Ask one source-grounded question using the RAG pipeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Limit document processing for test builds. Default test limit: {DEFAULT_BUILD_LIMIT}.",
    )
    parser.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Load the embedding model and run a lightweight sanity check.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search the saved vector store for a query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of search results to return. Default: {DEFAULT_RETRIEVAL_TOP_K}.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Optional exact theme filter for vector search.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Phase 10 evaluation: retrieval quality check + edge-case tests.",
    )
    return parser


def main() -> None:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args()

    if args.check_embeddings:
        print("Loading embedding model...")
        embedding_model = EmbeddingModel()
        result = run_embedding_sanity_check(embedding_model)
        print(f"Model: {embedding_model.model_name}")
        print(f"Vector dimension: {result.vector_dimension}")
        print(f"Self similarity: {result.self_similarity}")
        print(f"Different-text similarity: {result.other_similarity}")
        print(f"Empty vector norm: {result.empty_vector_norm}")
        print(f"Sanity check passed: {result.passed}")
        return

    if args.search is not None:
        if not args.search.strip():
            print("Search query cannot be empty.")
            return
        vector_db = VectorDB()
        try:
            vector_db.load()
        except FileNotFoundError as error:
            print(f"Vector store is not ready: {error}")
            print_build_instruction()
            return
        print("Loading embedding model...")
        embedding_model = EmbeddingModel()
        query_vector = embedding_model.embed_text(args.search, is_query=True)
        filters = {"theme": args.theme} if args.theme else None
        results = vector_db.search(query_vector, top_k=args.top_k, filters=filters, max_per_doc=DEFAULT_MAX_CHUNKS_PER_DOC)
        print(f"Results: {len(results)}")
        for index, result in enumerate(results, start=1):
            print(f"\n[{index}] {result.get('title')} ({result.get('doc_id')})")
            print(f"Theme: {result.get('theme')}")
            print(f"Similarity: {result.get('similarity'):.4f}")
            print(f"URL: {result.get('url')}")
            preview = " ".join(str(result.get("text") or "").split())[:500]
            print(f"Text: {preview}")
        return

    if args.ask is not None:
        if not args.ask.strip():
            print("Please provide a non-empty question.")
            return
        try:
            chatbot = PolicyChatbot(top_k=args.top_k)
            chatbot.load_vector_store()
        except FileNotFoundError as error:
            print(f"Vector store is not ready: {error}")
            print_build_instruction()
            return

        response = chatbot.answer(args.ask, top_k=args.top_k, theme=args.theme)
        print(response.answer)
        if response.error:
            print(f"\nModel error detail: {response.error}")
        print()
        print(format_sources(response.sources))
        return

    if args.evaluate:
        vector_db = VectorDB()
        try:
            vector_db.load()
        except FileNotFoundError as error:
            print(f"Vector store is not ready: {error}")
            print_build_instruction()
            return
        print("Loading embedding model...")
        embedding_model = EmbeddingModel()
        print(f"Running evaluation on {len(EVALUATION_QUESTIONS)} questions (top-k={args.top_k})...")
        results = run_evaluation(vector_db, embedding_model, top_k=args.top_k)
        print(format_evaluation_report(results))
        print("Running edge-case checks...")
        edge_checks = check_edge_cases(vector_db, embedding_model)
        print(format_edge_case_report(edge_checks))
        json_path, txt_path = save_evaluation_report(results)
        print(f"\nReports saved:")
        print(f"  JSON: {project_relative(json_path)}")
        print(f"  Text: {project_relative(txt_path)}")
        return

    if args.build or args.rebuild:
        records, summary = parse_dataset(limit=args.limit)
        print(f"{APP_NAME}: dataset parsed.")
        print(f"Dataset: {summary.dataset_path}")
        print(f"Sheet: {summary.sheet_name}")
        print(f"Rows read: {summary.rows}")
        print(f"URLs found: {summary.urls}")
        print(f"Missing URLs: {summary.missing_urls}")
        print(f"Duplicate URLs: {summary.duplicate_urls}")
        print(f"Columns: {', '.join(summary.columns)}")
        print(f"Metadata records saved: {len(records)}")
        print(f"Output: {project_relative(PROCESSED_DOCUMENTS_PATH)}")
        print("Downloading source documents...")
        updated_records, download_summary = download_documents(records, force=args.rebuild)
        print(f"Download records processed: {download_summary.total_records}")
        print(f"Downloaded: {download_summary.downloaded}")
        print(f"Skipped existing: {download_summary.skipped}")
        print(f"Failed: {download_summary.failed}")
        print(f"Unsupported: {download_summary.unsupported}")
        print(f"Duplicates: {download_summary.duplicates}")
        print(f"Missing URLs: {download_summary.missing_urls}")
        print(f"Updated metadata: {project_relative(PROCESSED_DOCUMENTS_PATH)}")
        print(f"Download report: {project_relative(DOWNLOAD_REPORT_PATH)}")
        if download_summary.failed_urls:
            print("Failed or unsupported URLs:")
            for failed in download_summary.failed_urls[:10]:
                print(f"- {failed['doc_id']}: {failed['status']} - {failed['error']}")
            remaining = len(download_summary.failed_urls) - 10
            if remaining > 0:
                print(f"...and {remaining} more. See the download report for details.")
        print("Extracting and cleaning text...")
        processed_records, extraction_summary = extract_and_clean_documents(updated_records)
        print(f"Extraction records processed: {extraction_summary.total_records}")
        print(f"Extracted: {extraction_summary.extracted}")
        print(f"Too short: {extraction_summary.too_short}")
        print(f"Empty: {extraction_summary.empty}")
        print(f"Failed: {extraction_summary.failed}")
        print(f"Skipped: {extraction_summary.skipped}")
        print(f"Cleaned metadata and text: {project_relative(PROCESSED_DOCUMENTS_PATH)}")
        if extraction_summary.problem_documents:
            print("Extraction issues:")
            for problem in extraction_summary.problem_documents[:10]:
                print(
                    f"- {problem['doc_id']}: {problem['extraction_status']} - "
                    f"{problem['extraction_error']}"
                )
            remaining = len(extraction_summary.problem_documents) - 10
            if remaining > 0:
                print(f"...and {remaining} more. See documents.jsonl for details.")
        print(f"Processed records ready for later phases: {len(processed_records)}")
        print("Creating retrieval chunks...")
        chunks, chunking_summary = chunk_documents(processed_records)
        print(f"Documents processed for chunking: {chunking_summary.document_count}")
        print(f"Chunks created: {chunking_summary.chunk_count}")
        print(f"Chunk word count min: {chunking_summary.min_words}")
        print(f"Chunk word count max: {chunking_summary.max_words}")
        print(f"Chunk word count average: {chunking_summary.average_words}")
        print(f"Chunks output: {project_relative(CHUNKS_PATH)}")
        if chunking_summary.zero_chunk_documents:
            print("Documents with zero chunks:")
            for document in chunking_summary.zero_chunk_documents[:10]:
                print(
                    f"- {document['doc_id']}: {document['extraction_status']} "
                    f"({document['text_char_count']} chars)"
                )
            remaining = len(chunking_summary.zero_chunk_documents) - 10
            if remaining > 0:
                print(f"...and {remaining} more.")
        print(f"Chunks ready for vectorization: {len(chunks)}")
        print("Building vector store...")
        embedding_model = EmbeddingModel()
        vector_db = VectorDB()
        vector_db.build(chunks, embedding_model)
        vector_db.save()
        print(f"Embedding model: {embedding_model.model_name}")
        print(f"Vector dimension: {embedding_model.vector_dimension}")
        print(f"Vector chunks saved: {project_relative(VECTOR_CHUNKS_PATH)}")
        print(f"Vector embeddings saved: {project_relative(VECTOR_EMBEDDINGS_PATH)}")
        print(f"Vector manifest saved: {project_relative(VECTOR_MANIFEST_PATH)}")
        print("Build complete. Run `python3 app.py` to start the chatbot.")
        return

    if args.chat:
        run_chat_loop(top_k=args.top_k, theme=args.theme)
        return

    run_chat_loop(top_k=args.top_k, theme=args.theme)


def print_build_instruction() -> None:
    """Print consistent instructions for creating the vector store."""
    print("Build a small test store with `python3 app.py --build --limit 3`.")
    print("Build the full store with `python3 app.py --build`.")


def run_chat_loop(top_k: int = DEFAULT_RETRIEVAL_TOP_K, theme: str | None = None) -> None:
    """Run a simple command-line chat loop."""
    try:
        chatbot = PolicyChatbot(top_k=top_k)
        chatbot.load_vector_store()
    except FileNotFoundError as error:
        print(f"Vector store is not ready: {error}")
        print_build_instruction()
        return

    print(f"{APP_NAME} chat")
    print(f"Vector store loaded with {len(chatbot.vector_db.chunks)} chunks. Retrieval top-k: {top_k}.")
    if theme:
        print(f"Theme filter: {theme}")
    print("Type /help for commands. Type /exit or /quit to stop.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_input:
            print("Please enter a question or command.")
            continue
        if user_input.casefold() in {"/exit", "/quit"}:
            print("Goodbye.")
            return

        command_response = chatbot.handle_command(user_input)
        if command_response is not None:
            print(command_response)
            continue

        response = chatbot.answer(user_input, top_k=top_k, theme=theme)
        print(f"\nAssistant: {response.answer}")
        if response.error:
            print(f"\nModel error detail: {response.error}")
        print()
        print(format_sources(response.sources))


if __name__ == "__main__":
    main()
