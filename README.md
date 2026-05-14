# NLP Policy Chatbot

Submission Date: 2026-06-05
Team Members:
- Name (Student Number)
- Name (Student Number)

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about EU AI policy, ethics, fairness, transparency, and responsibility. The chatbot grounds every response in real EU policy documents — it does not fabricate opinions or invent citations.

The knowledge base is built from 87 official EU publications and policy documents provided in `dataset/20260331_dataset2.xlsx`. Topics covered include: algorithmic bias, deepfakes, data protection, high-risk AI systems, healthcare AI, AI in law enforcement, AI literacy, labour market impacts, cloud sovereignty, and more.

The system can be used via a command-line interface or an optional Streamlit web application.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install and start Ollama, then pull the language model
#    https://ollama.com/
ollama pull llama3.2:3b

# 4. Start the chatbot (vector store is pre-built and included)
python app.py

# --- OR launch the Streamlit web interface ---
streamlit run streamlit_app.py
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running locally

### Steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

The processed text data and pre-built vector store are included in the repository. You do **not** need to re-download or re-process the source documents to run the chatbot.

---

## Usage

### Command-line interface

```bash
# Start the interactive chatbot
python app.py

# Ask a single question and exit
python app.py --ask "Who is responsible when an AI system causes harm?"

# Search the vector store directly (no LLM)
python app.py --search "algorithmic bias in hiring" --top-k 5

# Filter by theme
python app.py --search "transparency obligations" --theme "Human Oversight & Transparency"

# Run the embedding sanity check
python app.py --check-embeddings

# Run the retrieval evaluation (8 benchmark questions)
python app.py --evaluate

# Rebuild the vector store from cached chunks (re-embedding only)
python app.py --build

# Force full rebuild from scratch (re-download, re-chunk, re-embed)
python app.py --rebuild
```

### Chat commands (in interactive mode)

| Command | Effect |
|---------|--------|
| `/help` | List available commands |
| `/reset` | Clear conversation history and recent sources |
| `/sources` | Show sources used in the last response |
| `/exit` or `/quit` | Close the chatbot |

### Streamlit web interface

```bash
streamlit run streamlit_app.py
```

The web interface provides:
- **Chat** page: interactive multi-turn conversation with source cards
- **Search** page: direct vector search with similarity scores and text excerpts
- **Evaluate** page: run the retrieval benchmark
- **Build** page: rebuild the vector store without touching the terminal
- **Sources** page: browse all documents in the knowledge base

---

## Project Structure

```text
.
├── README.md
├── app.py                          # CLI entry point
├── streamlit_app.py                # Streamlit chat interface
├── requirements.txt
├── dataset/
│   └── 20260331_dataset2.xlsx      # Curated EU policy document list
├── data/
│   ├── raw/                        # Downloaded PDFs and HTML files
│   ├── processed/
│   │   ├── documents.jsonl         # Extracted and cleaned document records
│   │   ├── chunks.jsonl            # Retrieval-ready text chunks
│   │   └── download_report.json    # Acquisition summary
│   └── vector_store/
│       ├── embeddings.npy          # Normalised chunk embeddings (NumPy)
│       ├── chunks.jsonl            # Chunk text and metadata
│       └── manifest.json           # Store metadata (model, dimension, count)
├── pages/                          # Streamlit multi-page app pages
│   ├── 1_Search.py
│   ├── 2_Evaluate.py
│   ├── 3_Build.py
│   └── 4_Sources.py
├── src/
│   ├── config.py                   # Central configuration
│   ├── downloader.py               # Document acquisition pipeline
│   ├── preprocessing.py            # Text extraction, cleaning, chunking
│   ├── embeddings.py               # Embedding model wrapper
│   ├── vector_db.py                # Custom vector database
│   ├── chat.py                     # RAG pipeline and chatbot logic
│   ├── evaluate.py                 # Retrieval evaluation
│   └── utils.py                    # Shared utilities
└── documents/
    └── Example Conversation Themes/
        ├── 20260407_scenarios.docx
        └── 20260413_additional_script.docx
```

---

## Architecture

The system is organised into five sequential stages:

```
Excel dataset
     │
     ▼
[Downloader]  ──→  data/raw/          (PDFs and HTML files)
     │
     ▼
[Preprocessor] ──→  documents.jsonl   (extracted + cleaned text)
     │
     ▼
[Chunker]      ──→  chunks.jsonl      (overlapping text segments)
     │
     ▼
[EmbeddingModel] ──→  embeddings.npy  (dense vectors, 384-dim)
     │
     ▼
[VectorDB]     ──→  vector_store/     (persisted similarity index)
     │
     ▼
[PolicyChatbot]                       (RAG pipeline + Ollama LLM)
```

At query time:

```
User question
     │
     ▼
[EmbeddingModel.embed_text]   (query vector with BGE prefix)
     │
     ▼
[VectorDB.search]             (cosine similarity + MMR reranking)
     │
     ▼
[PolicyChatbot.build_messages] (system prompt + history + context)
     │
     ▼
[Ollama llama3.2:3b]          (grounded answer generation)
     │
     ▼
Answer + Source attribution
```

---

## NLP Approach

### Text Embeddings

The system uses [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) from the Sentence Transformers library. This model was chosen for three reasons:

1. **Retrieval-optimised**: BGE (BAAI General Embedding) models are fine-tuned specifically for asymmetric passage retrieval on the BEIR and MTEB benchmarks, making them more appropriate than general-purpose embeddings for a RAG use case.

2. **Asymmetric query prefix**: BGE-v1.5 uses different representations for queries and documents. At query time, the prefix `"Represent this sentence for searching relevant passages: "` is prepended to the question before embedding. Document chunks are embedded without any prefix. This asymmetry is standard practice for retrieval-focused models and meaningfully improves match quality.

3. **Efficiency**: The small variant (33M parameters, 384-dimensional vectors) embeds all 1,060 chunks quickly on CPU and fits comfortably in a 16 GB RAM environment.

All vectors are L2-normalised after embedding. This converts cosine similarity to a simple dot product at search time, which is both faster and numerically stable.

### Chunking Strategy

Raw document text is split into overlapping chunks using a sentence-aware strategy:

1. Text is first split on sentence boundaries (`re.compile(r"(?<=[.!?])\s+")`), preserving complete sentences.
2. Sentences are accumulated until the chunk reaches the target size (400 words).
3. The next chunk rewinds by 80 words worth of sentences (overlap), ensuring that information near a boundary appears in both adjacent chunks.

The 400-word target was chosen based on the context window that BGE-small handles well and the typical length of a self-contained policy argument. The 80-word overlap prevents information loss at boundaries — a sentence discussing a specific obligation should not be retrievable only if a user happens to phrase a query that aligns with the chunk start.

Statistics for the current knowledge base:
- **1,060 total chunks** from 85 documents
- **86.4%** fall between 300–500 words (tight, consistent distribution)
- Mean chunk length: **354 words**

### Similarity Search

The `VectorDB` class implements cosine similarity search from scratch using NumPy:

```
similarity(query, chunk) = query_vector · chunk_vector
```

This works because both vectors are L2-normalised at build and query time, so the dot product equals the cosine similarity directly. The implementation avoids any external vector search library.

### Maximal Marginal Relevance (MMR)

Naive top-k retrieval often returns several chunks from the same document, wasting context slots on redundant information. The system applies **MMR** (Maximal Marginal Relevance) to rerank results for diversity.

MMR works iteratively: at each step it selects the candidate that maximises:

```
score = λ × sim(chunk, query) − (1 − λ) × max_sim(chunk, already_selected)
```

With `λ = 0.7`, the selection weights relevance more than diversity (a standard value for retrieval tasks). The candidate pool is the top `4 × top_k` chunks by raw similarity, so the greedy loop runs over at most 20 candidates rather than all 1,060 — keeping it computationally cheap.

A per-document cap (`max_per_doc = 2`) ensures that no single document occupies more than two retrieval slots, complementing MMR's semantic diversity with source-level diversity.

### Evidence Threshold

Before calling the LLM, the system checks whether the top retrieved chunk scores above a minimum similarity threshold (`MIN_EVIDENCE_SIMILARITY = 0.55`). If no chunk exceeds this threshold, the LLM is not called and the user receives an "out of scope" message. This prevents the model from hallucinating answers when the knowledge base genuinely does not contain relevant information.

A separate lower floor (`RETRIEVAL_SIMILARITY_FLOOR = 0.35`) filters individual weak chunks from the context even when the overall query passes the evidence check.

### Context Management and Follow-up Handling

Conversation history is maintained as a rolling window of the last 4 turns (configurable). Each turn is passed to the LLM as proper `user`/`assistant` role messages rather than a pasted text block, matching the chat model's expected multi-turn format.

For short follow-up questions (under 15 words), the previous user question is prepended to the query before embedding. This enriches the query with domain vocabulary that short follow-ups typically lack, preventing them from scoring below the evidence threshold incorrectly.

Example: `"who is at fault?"` becomes `"Who is responsible when an AI system causes harm? who is at fault?"` for embedding purposes, while the LLM still receives only the original short question.

### Prompt Engineering

The system prompt instructs the model to act as an EU AI policy analyst and structure answers in four parts: overview, key obligations/roles, nuance/caveats, and conclusion. It is asked to synthesise across all retrieved sources rather than citing them in isolation.

Temperature is set to `0.4` — low enough to stay grounded in the sources, high enough to produce natural explanatory prose rather than flat bullet lists. `max_tokens = 800` prevents truncated responses on longer policy questions.

---

## Data Acquisition

### Pipeline

1. **Parse** `dataset/20260331_dataset2.xlsx` (Sheet1, 87 rows) to extract document ID, title, URL, theme, publisher, date, and citation.
2. **Classify** each URL as PDF or HTML by extension; fall back to HTTP `HEAD` content-type check for ambiguous URLs.
3. **Download** PDFs with `requests`, saving to `data/raw/<doc_id>.pdf`. HTML pages are downloaded and saved as `data/raw/<doc_id>.html`.
4. **Extract text**: PDFs use `pypdf` (all pages); HTML uses `BeautifulSoup` with removal of scripts, styles, navigation, headers, footers, and form elements.
5. **Clean** extracted text: normalise whitespace, repair broken hyphenated line endings, strip non-printable characters.
6. **Cache**: files already on disk are skipped on subsequent builds. `--rebuild` forces re-download.

### Issues Encountered

| Document | Issue | Handling |
|----------|-------|----------|
| R21-03 | HTTP 403 Forbidden — CDT Europe blocked automated access | Skipped; recorded in `download_report.json` |
| R18-01 | Extracted text under 500 characters (likely a redirect or login wall) | Marked `too_short`; excluded from chunks |

85 of 87 documents were successfully extracted and chunked. The two missing documents represent 2.3% of the dataset and do not cover any theme exclusively — their topics are addressed by other documents in the knowledge base.

---

## Known Limitations

- **Knowledge base coverage**: The dataset does not include the full text of every AI Act article. Questions about specific provider/deployer liability articles (Arts. 16–29) may receive partial answers because those articles are not directly in the corpus.

- **Language**: All documents are in English. Questions in other languages are not supported.

- **LLM dependency**: Answer generation requires a locally running Ollama instance. If Ollama is unavailable, the system still retrieves and displays source passages but cannot generate a synthesised answer.

- **Static knowledge base**: The vector store reflects the documents available at build time. New EU policy publications are not automatically included.

- **Out-of-scope detection**: The evidence threshold (0.55 cosine similarity) reliably rejects clearly unrelated queries but may occasionally flag genuinely in-scope questions that use non-standard terminology. Rephrasing with policy vocabulary typically resolves this.

- **PDF extraction quality**: A small number of PDFs contain scanned images rather than machine-readable text. `pypdf` cannot extract text from image-only PDFs; those documents produce very short or empty extractions and are excluded from the knowledge base.

---

## Configuration

All tuneable parameters are centralised in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence Transformers model |
| `DEFAULT_CHUNK_SIZE` | `400` | Target words per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `80` | Overlap words between chunks |
| `DEFAULT_RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `DEFAULT_MAX_CHUNKS_PER_DOC` | `2` | Max chunks per document in results |
| `MMR_LAMBDA` | `0.7` | MMR relevance weight (0=diversity, 1=relevance) |
| `MIN_EVIDENCE_SIMILARITY` | `0.55` | Minimum score to call the LLM |
| `OLLAMA_MODEL_NAME` | `llama3.2:3b` | Local Ollama model |
| `MAX_HISTORY_TURNS` | `4` | Conversation turns kept in context |
| `MAX_GENERATION_TOKENS` | `800` | Maximum LLM output tokens |
| `GENERATION_TEMPERATURE` | `0.4` | LLM sampling temperature |

---

## Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Pre-trained embedding model (BGE-small) |
| `numpy` | Vector storage, cosine similarity, MMR |
| `openai` | OpenAI-compatible SDK for Ollama |
| `requests` + `beautifulsoup4` | Document downloading and HTML extraction |
| `pypdf` | PDF text extraction |
| `pandas` + `openpyxl` | Excel dataset parsing |
| `streamlit` | Optional web interface |
| `torch` | PyTorch backend for sentence-transformers |
