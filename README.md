# NLP Policy Chatbot

Submission Date: %Y-%m-%d
Team Members:
- Name (Student Number)

## Project Overview

This project is a command-line Retrieval-Augmented Generation chatbot about AI ethics and EU policy. The final chatbot will answer questions about trust, fairness, responsibility, transparency, bias, privacy, public sector AI, healthcare AI, law enforcement, migration, education, and related themes.

The knowledge base is built from official EU policy documents listed in `dataset/20260331_dataset2.xlsx`. The chatbot is designed to ground its answers in retrieved source text and provide source attribution instead of inventing unsupported claims.

## Development Status

Implemented so far:

- Phase 1: project structure and central configuration
- Phase 2: Excel dataset parsing and metadata normalization
- Phase 3: document acquisition for PDFs and web pages
- Phase 4: text extraction and cleaning for downloaded PDFs and HTML pages
- Phase 5: overlapping retrieval chunk generation
- Phase 6: pre-trained embedding model wrapper and vector normalization
- Phase 7: custom vector database with persisted embeddings and similarity search
- Phase 8: RAG pipeline, prompt builder, source attribution, and Ollama integration
- Phase 9: command-line chatbot application and build commands

Not implemented yet:

- Streamlit interface

## Project Structure

```text
.
├── README.md
├── app.py
├── requirements.txt
├── dataset/
│   └── 20260331_dataset2.xlsx
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   │   ├── documents.jsonl
│   │   ├── chunks.jsonl
│   │   └── download_report.json
│   └── vector_store/
│       ├── embeddings.npy
│       ├── chunks.jsonl
│       └── manifest.json
├── documents/
│   ├── project_description.md
│   ├── implementation_plan.md
│   └── Example Conversation Themes/
│       ├── 20260407_scenarios.docx
│       └── 20260413_additional_script.docx
└── src/
    ├── __init__.py
    ├── config.py
    ├── downloader.py
    ├── preprocessing.py
    ├── embeddings.py
    ├── vector_db.py
    ├── chat.py
    └── utils.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

## Current Usage

Start the chatbot:

```bash
python app.py
```

Show available commands:

```bash
python app.py --help
```

Run the embedding sanity check:

```bash
python app.py --check-embeddings
```

Search the saved vector store:

```bash
python app.py --search "How do deepfakes affect children?" --top-k 3
```

Search with an exact theme filter:

```bash
python app.py --search "How do deepfakes affect children?" --theme "Deepfakes & Manipulation"
```

Ask one source-grounded question with the RAG pipeline:

```bash
python app.py --ask "How do deepfakes affect children?" --top-k 3
```

Start the command-line chat loop:

```bash
python app.py --chat
```

Chat commands:

- `/help`
- `/reset`
- `/sources`
- `/exit`
- `/quit`

Build a small test subset of the source metadata and downloaded files:
This parses metadata, downloads or reuses files, extracts text, writes cleaned document records, creates retrieval chunks, and builds the vector store.

```bash
python app.py --build --limit 3
```

Force re-downloading for a small test subset:

```bash
python app.py --rebuild --limit 3
```

Build without `--limit` will process all 87 dataset records:

```bash
python app.py --build
```

For development, use `--limit` first to avoid repeatedly downloading all sources.

## Phase 1: Project Setup

Phase 1 created the base project structure, placeholder Python modules, data directories, and central configuration.

Important configuration values live in `src/config.py`, including:

- Dataset path
- Data directories
- Processed output paths
- Chunk size and overlap settings
- Embedding model name
- Ollama base URL and model name
- Retrieval top-k
- HTTP timeout and user-agent

Acceptance checks completed:

- `python3 app.py` starts without import errors.
- `python3 app.py --help` works.
- Data directories exist and are documented in `data/README.md`.
- No `.venv`, `__pycache__`, `.pyc`, or `.DS_Store` files are required.

## Phase 2: Dataset Parsing

The dataset parser reads `dataset/20260331_dataset2.xlsx` from `Sheet1`. `Sheet2` is empty and ignored.

Dataset columns:

- `id`
- `title`
- `url`
- `publisher`
- `date`
- `theme`
- `keywords`
- `bronvermelding`

The parser normalizes every row into records with these fields:

- `doc_id`
- `title`
- `url`
- `theme`
- `keywords`
- `source_type`
- `publisher`
- `date`
- `citation`
- `local_path`
- `status`
- `error`
- `duplicate_of`

Current dataset inspection:

- Rows: 87
- URLs: 87
- Missing URLs: 0
- Duplicate URLs: 0
- Direct PDF-like URLs: 21
- Web or content-type-dependent URLs: 66

Output:

```text
data/processed/documents.jsonl
```

## Phase 3: Document Acquisition

The downloader handles source acquisition from the URLs in the dataset.

Implemented behavior:

- Classifies `.pdf` URLs as PDFs.
- Uses HTTP `HEAD` content-type checks when the URL extension is unclear.
- Saves PDFs as `.pdf`.
- Saves HTML pages as `.html`.
- Uses stable filenames based on `doc_id`.
- Normalizes unsafe filename characters.
- Uses a configured timeout.
- Sends a user-agent header.
- Follows redirects.
- Skips files that already exist unless `--rebuild` is used.
- Logs failed and unsupported URLs without stopping the whole run.

Outputs:

```text
data/raw/
data/processed/documents.jsonl
data/processed/download_report.json
```

The current small test run downloaded or reused three sources:

```text
data/raw/R01-01.pdf
data/raw/R01_02.pdf
data/raw/R01-03.html
```

The download report contains:

- Total records processed
- Downloaded count
- Skipped existing count
- Failed count
- Unsupported count
- Duplicate count
- Missing URL count
- Details for failed or unsupported URLs

## Phase 4: Text Extraction and Cleaning

The preprocessing step extracts text from downloaded source files and saves the cleaned text back into `data/processed/documents.jsonl`.

Implemented behavior:

- PDF text extraction with `pypdf`.
- HTML text extraction with BeautifulSoup.
- PDF extraction reads every available page reported by the PDF parser.
- HTML extraction removes obvious non-content elements such as scripts, styles, navigation, headers, footers, forms, buttons, SVGs, and sidebars.
- Extracted text is cleaned into readable plain text.
- Excessive whitespace and repeated blank lines are normalized.
- Broken hyphenated line endings are repaired where practical.
- Missing local files, unsupported source types, extraction failures, empty text, and very short text are recorded in metadata instead of stopping the pipeline.

Each processed document record keeps the original document metadata and adds fields such as:

- `text`
- `text_char_count`
- `raw_text_char_count`
- `extraction_status`
- `extraction_error`
- `extraction_method`
- `page_count`

Possible `extraction_status` values include:

- `extracted`
- `too_short`
- `empty_text`
- `extraction_failed`
- `skipped`

Current small test result for `python app.py --build --limit 3`:

- Records processed: 3
- Extracted: 3
- Too short: 0
- Empty: 0
- Failed: 0
- Skipped: 0

The cleaned text is ready for Phase 5 chunking.

## Phase 5: Chunking

The chunking step splits cleaned document text into overlapping word-based chunks for later semantic retrieval.

Current settings from `src/config.py`:

- Chunk size: 400 words
- Chunk overlap: 80 words

Why this strategy is used:

- 400-word chunks keep retrieval focused on a smaller policy passage.
- 80-word overlap reduces the risk that important context is split across chunk boundaries.
- Word-based chunks are simple to explain and inspect for the assignment.

Each chunk is saved to:

```text
data/processed/chunks.jsonl
```

Each chunk record contains:

- `chunk_id`
- `doc_id`
- `title`
- `theme`
- `url`
- `publisher`
- `date`
- `citation`
- `source_type`
- `local_path`
- `chunk_index`
- `chunk_word_count`
- `text`

Current small test result for `python app.py --build --limit 3`:

- Documents processed for chunking: 3
- Chunks created: 22
- Minimum chunk length: 218 words
- Maximum chunk length: 400 words
- Average chunk length: 377.05 words
- Documents with zero chunks: 0

The chunks are ready for Phase 6 embedding generation.

## Phase 6: Embeddings

The embedding step converts text into numerical vectors for semantic search. The implementation uses a pre-trained Sentence Transformers model, not a custom embedding model.

Current model from `src/config.py`:

```text
BAAI/bge-small-en-v1.5
```

Current vector dimension:

```text
384
```

Implemented behavior:

- `EmbeddingModel.embed_text(text)` embeds a single string.
- `EmbeddingModel.embed_batch(texts)` embeds a list of strings.
- Returned embeddings are NumPy arrays.
- Non-empty embeddings are normalized to unit length.
- Empty strings return zero vectors safely.
- Zero vectors are handled without divide-by-zero errors.
- BGE query embeddings use the recommended retrieval prefix: `Represent this sentence for searching relevant passages:`.
- Document/chunk embeddings are not prefixed.
- A lightweight sanity check confirms identical text has higher similarity than unrelated text.

Run:

```bash
python app.py --check-embeddings
```

Current sanity check result:

- Model: `BAAI/bge-small-en-v1.5`
- Vector dimension: 384
- Self similarity: 0.956665
- Different-text similarity: 0.398063
- Empty vector norm: 0.0
- Sanity check passed: True

The vector dimension will also be written into the vector store manifest in Phase 7.

## Phase 7: Custom Vector Database

The vector database is implemented from scratch in `src/vector_db.py` using NumPy arrays and JSONL metadata. It does not use a pre-built vector database library.

The vector store is saved in:

```text
data/vector_store/
```

Files:

- `embeddings.npy`: NumPy array containing normalized chunk embeddings
- `chunks.jsonl`: human-readable chunk text and metadata
- `manifest.json`: vector store metadata

The manifest records:

- Creation timestamp
- Embedding model name
- Vector dimension
- Chunk count
- Embedding array shape
- File names
- Whether embeddings are normalized

Current small test vector store:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Vector dimension: 384
- Chunk count: 22
- Embedding shape: `[22, 384]`
- Normalized: True

Implemented behavior:

- Builds embeddings from `data/processed/chunks.jsonl`.
- Stores embeddings separately from chunk metadata.
- Saves and loads the vector store across runs.
- Loading the vector store does not recompute embeddings.
- Implements cosine similarity search with NumPy.
- Returns top-k results ordered by similarity.
- Each result includes chunk text, metadata, and similarity score.
- Handles an empty database gracefully.
- Handles `top_k` larger than the number of chunks.
- Supports optional exact metadata filtering, currently useful for `theme`.

Example:

```bash
python app.py --search "How do deepfakes affect children?" --top-k 3
```

The current small test search returns relevant chunks from `Children and Deepfakes`.

## Phase 8: RAG Pipeline and Chatbot Logic

The RAG pipeline is implemented in `src/chat.py`. It combines query embedding, vector search, prompt construction, local LLM generation through Ollama, conversation history, and source attribution.

Implemented behavior:

- Embeds user questions as retrieval queries.
- Retrieves relevant chunks from the custom `VectorDB`.
- Supports configurable `top_k`.
- Supports optional exact `theme` filtering.
- Handles empty questions with a helpful message.
- Builds a prompt containing the question, retrieved source excerpts, source metadata, and recent conversation history.
- Instructs the model to answer only from the provided EU policy source excerpts.
- Instructs the model to say when the sources are insufficient.
- Instructs the model to cite source labels such as `[Source 1]`.
- Calls a local Ollama model through the OpenAI-compatible SDK.
- Uses configurable Ollama settings from `src/config.py`.
- Handles Ollama connection/model errors without crashing.
- De-duplicates sources in the final source list.
- Keeps a bounded rolling conversation history.
- Supports `/reset`, `/sources`, and `/help` in the chat loop.

Current Ollama configuration:

```text
Base URL: http://localhost:11434/v1
Model: llama3.1
```

One-shot RAG command:

```bash
python app.py --ask "How do deepfakes affect children?" --top-k 3
```

If Ollama is not running or the configured model is missing, the system still retrieves sources and prints a clear model error instead of crashing.

Current local test result:

- Retrieval returned relevant chunks for `How do deepfakes affect children?`
- Top source: `Children and Deepfakes`
- Source attribution included title, document ID, theme, URL, citation, and similarity score.
- If Ollama is reachable, the chatbot produces a grounded answer with source labels.
- If Ollama is not reachable, the CLI reports: `I could retrieve relevant source material, but I could not contact the local Ollama model.`

## Phase 9: CLI Application

`app.py` is the main command-line entry point. Running `python app.py` starts the chatbot by default.

Implemented behavior:

- `python app.py` starts the chatbot.
- `python app.py --chat` also starts the chatbot.
- Users can type questions and receive source-grounded answers.
- The chatbot prints sources after each response.
- `/help` lists available chat commands.
- `/reset` clears conversation history and recent sources.
- `/sources` shows sources used in the most recent response.
- `/exit` and `/quit` close the chatbot.
- Empty input is handled without crashing.
- Keyboard interruption exits cleanly.
- Missing vector store errors include build instructions.

Build commands:

```bash
python app.py --build
python app.py --rebuild
python app.py --build --limit 10
```

Useful diagnostic commands:

```bash
python app.py --check-embeddings
python app.py --search "How do deepfakes affect children?" --top-k 3
python app.py --ask "How do deepfakes affect children?" --top-k 3
```

## Development Notes

Use small test mode while building later phases:

```bash
python app.py --build --limit 3
```

Use full mode only when needed:

```bash
python app.py --build
```

Raw downloaded PDFs should not be included in the final submission zip. The final submission should include processed text, chunks, and the vector store once those are generated.
