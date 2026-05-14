# NLP Policy Chatbot

Submission Date: 2026-06-05
Team Members:
- Ziad Alatrash 722338
- M.D Tasnim Hassan Tabeeb 715660
- Mahmoud Farid Mahrous Farid 693428


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

This section explains the NLP concepts behind each component of the system: what each technique does, why it was chosen, and how the design decisions were made.

### Overview: What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system (finding relevant passages from a document corpus) with a generative language model (producing a natural language answer). The key insight is that a language model alone cannot be trusted for factual questions about specific documents — it may have never seen those documents during training, or may "hallucinate" plausible-sounding but incorrect details. RAG grounds the model's output in retrieved text, making answers traceable and verifiable.

The pipeline works in two phases:

**Offline (build time):** All documents are downloaded, cleaned, chunked, and embedded into vectors. These vectors are stored in the VectorDB.

**Online (query time):** The user's question is embedded into the same vector space. The most semantically similar document chunks are retrieved. Those chunks are injected into the LLM's prompt as context, and the model generates an answer grounded in that context.

---

### 1. Text Embeddings

#### What embeddings are

An embedding model converts a piece of text into a fixed-length numerical vector — for example, 384 numbers. The key property is that semantically similar texts produce vectors that are close together in this high-dimensional space. The word "transparency" and the phrase "obligation to disclose" will produce vectors that are nearby, even though they share no characters.

This is fundamentally different from keyword search (which matches exact words) or TF-IDF (which counts word frequencies). Embedding-based search understands *meaning*, so a query like "who is liable when AI goes wrong?" can match a document that uses the word "accountability" but never the word "liable".

#### How transformer-based embedding works

The model (`BAAI/bge-small-en-v1.5`) is a 33-million-parameter transformer trained using a contrastive learning objective. During training, it is shown triplets: a query, a relevant passage, and a non-relevant passage. It is trained to pull the query vector close to the relevant passage and push it away from the non-relevant one. After training, the final `[CLS]` token from the transformer's last layer is extracted as the document's representation.

The result is a 384-dimensional vector where proximity in vector space corresponds to semantic similarity across domains — including legal and policy language, which BGE was specifically trained on through the BEIR retrieval benchmark.

#### Why BGE-small was chosen

BGE (BAAI General Embedding) models are fine-tuned specifically for **asymmetric passage retrieval**: the situation where a short question must be matched against longer document passages. This is exactly the RAG use case. General-purpose sentence encoders (like `all-MiniLM`) are trained on symmetric pairs (sentences of similar length) and perform worse on retrieval tasks.

The BGE-v1.5 series introduces an important improvement: a **retrieval instruction prefix** applied only to queries:

```
"Represent this sentence for searching relevant passages: <question>"
```

Document chunks are embedded without any prefix. This asymmetry tells the model to map the query into the same part of the vector space that passage text occupies, even though queries and passages have different lengths and styles. Applying this prefix consistently at query time (but never at index time) is critical — if both query and document use the prefix, or neither does, the retrieval quality degrades.

This is implemented in `src/embeddings.py` via the `is_query` flag:

```python
def prepare_text(self, text: str, is_query: bool = False) -> str:
    if is_query and self.uses_bge_query_prefix:
        return f"{BGE_QUERY_PREFIX}{cleaned}"
    return cleaned
```

#### Vector normalisation

All vectors are L2-normalised immediately after encoding. A normalised vector has a Euclidean length (norm) of 1.0. This normalisation step converts cosine similarity into a simple dot product:

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
                        = a · b    (when |a| = |b| = 1)
```

This is significant because:
1. NumPy's matrix multiplication (`@`) can compute all pairwise similarities in a single operation
2. It eliminates a division, reducing floating-point error
3. All similarity scores are naturally bounded between -1 and 1

The normalisation code in `src/embeddings.py`:

```python
norms = np.linalg.norm(array, axis=1, keepdims=True)
safe_norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
return array / safe_norms
```

---

### 2. Chunking Strategy

#### Why chunking is necessary

Language models have a finite context window. Even if the full text of a 40-page document were provided, the model would struggle to focus on the relevant passage. More importantly, the embedding model must produce a single vector for each piece of text — a 40-page document would produce one averaged vector that captures everything and nothing specifically.

Chunking solves both problems by dividing documents into passages small enough to embed meaningfully and retrieve precisely.

#### The granularity tradeoff

- **Too small** (e.g., 50 words): each chunk captures only a fragment of a policy argument. A chunk might describe the penalty without mentioning which obligation it applies to.
- **Too large** (e.g., 1,000 words): the embedding averages over too much content. A chunk covering five different topics will be weakly similar to queries about any of them.

The 400-word target was chosen as the sweet spot: long enough to capture a complete policy argument with its context, short enough to embed a specific semantic theme.

#### Sentence-aware overlapping chunking

A naive word-count chunker splits text at arbitrary word positions, potentially cutting mid-sentence. This produces chunks starting with fragments like `"obligations, while the deployer must..."`, which are harder for the embedding model to anchor semantically.

The system instead splits text into sentences first using a regex that matches sentence-ending punctuation followed by whitespace:

```python
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
```

Sentences are then accumulated into chunks until the 400-word target is reached. The next chunk rewinds by 80 words worth of sentences before continuing. This **overlap** ensures that information near a chunk boundary appears in two consecutive chunks — a sentence about a specific legal obligation will not be unretrievable simply because the query aligns with the second half of the passage rather than the first.

Statistics for the current knowledge base:

| Metric | Value |
|--------|-------|
| Total chunks | 1,060 |
| Documents | 85 |
| In 300–500 word range | 86.4% |
| Mean chunk length | 354 words |
| Chunks starting mid-sentence | < 4% |

---

### 3. Semantic Similarity Search

The `VectorDB` class (`src/vector_db.py`) implements similarity search from scratch using only NumPy — no external vector database libraries.

#### How the search works

At query time, the user's question is embedded into a 384-dimensional vector (with the BGE retrieval prefix). All 1,060 stored chunk embeddings are already in memory as a NumPy array of shape `(1060, 384)`. A single matrix-vector multiplication computes all similarities at once:

```python
scores = candidate_embeddings @ query  # shape: (1060,)
```

Because both the stored embeddings and the query vector are L2-normalised, each element of `scores` is the cosine similarity between the query and that chunk. The top-k indices are selected with `np.argsort(scores)[::-1][:top_k]`.

This approach computes 1,060 similarity scores in a single vectorised NumPy operation — significantly faster than a Python loop, and sufficient for a corpus of this size without an approximate nearest-neighbour index.

#### Optional metadata filtering

Before computing similarities, the system can pre-filter chunks by theme using exact string matching on the `theme` field. This reduces the candidate set and improves precision for theme-specific queries.

---

### 4. Maximal Marginal Relevance (MMR)

#### The redundancy problem

Naive top-k retrieval ranks chunks by similarity to the query. In practice, several of the top-5 results are often from the same document — the document most closely matching the query has multiple relevant passages. This wastes context slots: the LLM receives five variations of the same argument instead of five complementary perspectives.

#### MMR algorithm

MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) solves this by greedily selecting chunks that are both *relevant to the query* and *different from what has already been selected*:

```
MMR(chunk_i) = λ × sim(chunk_i, query) − (1 − λ) × max_{j ∈ Selected} sim(chunk_i, chunk_j)
```

At each iteration, the chunk with the highest MMR score is selected. The second term penalises chunks that are semantically close to already-selected chunks, encouraging diversity. With `λ = 0.7`, relevance is weighted roughly twice as heavily as diversity.

#### Implementation detail

Running MMR over all 1,060 candidates on every query would be computationally wasteful. Instead, the top `4 × top_k = 20` candidates by raw similarity are selected first, and MMR runs only over that pool. This reduces the greedy loop to O(top_k × 20) rather than O(top_k × 1060) with negligible quality loss, since the globally optimal diverse set will almost always be found within the top-20 by similarity.

An additional per-document cap (`max_per_doc = 2`) enforces source diversity at the document level, complementing MMR's semantic diversity.

---

### 5. Evidence Threshold and Out-of-Scope Detection

LLMs are prone to confabulation — generating fluent but fabricated answers when the provided context is weak. To prevent this, the system evaluates the quality of retrieved evidence *before* calling the LLM.

The top retrieved chunk's cosine similarity score serves as a proxy for "is this question answerable from the knowledge base?". Two thresholds are applied:

| Threshold | Value | Effect |
|-----------|-------|--------|
| `RETRIEVAL_SIMILARITY_FLOOR` | 0.35 | Individual chunks below this score are excluded from the context |
| `MIN_EVIDENCE_SIMILARITY` | 0.55 | If the best chunk is below this, the LLM is not called at all |

The 0.55 cutoff was established empirically: on-topic policy questions score between 0.63 and 0.75, while clearly out-of-scope queries (e.g., a recipe question) score below 0.20. The threshold sits at 0.55 to create a buffer that catches weak but topically adjacent queries without blocking legitimate policy questions.

---

### 6. Context Management and Follow-up Handling

#### Conversation history

The chatbot maintains a rolling window of the last 4 conversation turns in memory. These are passed to the LLM as proper `user`/`assistant` role message pairs — not concatenated into a text blob. This matches the chat model's expected multi-turn format and allows the LLM to resolve pronouns and references across turns naturally (e.g., "what does *it* say about penalties?" after a question about the AI Act).

The history window is capped at 4 turns to keep the prompt size manageable and prevent older, irrelevant context from influencing new answers.

#### Contextual query enrichment for follow-ups

A key challenge in multi-turn retrieval is that follow-up questions are often short and ambiguous:

> "who is at fault?" (after asking about AI causing harm)

This 4-word query lacks the domain vocabulary needed to embed close to policy text. Submitted to the vector store in isolation, it may score below the evidence threshold even though it is clearly in scope.

The system detects short follow-up questions (under 15 words) and prepends the previous user question before embedding:

```python
def _retrieval_query(self, question: str) -> str:
    if self.history and len(question.split()) < 15:
        return f"{self.history[-1]['user']} {question}"
    return question
```

This enriched query is used *only for retrieval* — the LLM still receives the original short question plus the conversation history. The result is that `"Who is responsible when an AI system causes harm? who is at fault?"` embeds correctly into the liability/accountability cluster of the vector space.

---

### 7. Prompt Engineering

#### Context format

Retrieved chunks are formatted with the document title as a header, followed by the full excerpt text, and metadata (theme, URL) as a footer annotation. This ordering is intentional: transformer attention mechanisms weight earlier tokens more heavily, so the actual content appears before metadata noise.

```
[Source 1] AI Liability Directive – Legislative Train Overview
<excerpt text>
— Governance, Accountability & Liability | https://...
```

Placing the document ID, similarity score, and URL *after* the text ensures the model attends to the substantive content first.

#### System prompt design

The system prompt was designed to counteract two failure modes common in RAG systems:

1. **Minimal responses**: an instruction-following model asked only to "answer from sources" tends to produce short, hedged responses that merely summarise the sources.
2. **Source-level isolation**: without explicit instruction, models cite sources sentence by sentence rather than synthesising a coherent argument.

The prompt addresses both:

```
You are an expert EU AI policy analyst. [...] Write a thorough, structured answer
that synthesises across all relevant sources — do not treat each source in isolation.
Structure your response as follows:
(1) a brief overview of the core answer,
(2) the key obligations, roles, or mechanisms involved,
(3) nuances, exceptions, or areas of complexity,
(4) a concise conclusion.
Cite sources inline as [Source 1], [Source 2], etc.
Do not invent policy details, legal obligations, dates, or article numbers
not present in the sources.
```

The four-part structure guides the model toward analytical depth. The explicit "do not invent" instruction reduces hallucination of specific article numbers and dates.

#### Temperature and token budget

`temperature = 0.4`: Low temperature (near 0) produces flat, repetitive prose; higher temperature (above 0.7) risks generating ungrounded claims. 0.4 produces natural explanatory language while keeping the model anchored to the retrieved text.

`max_tokens = 800`: Without an explicit token limit, Llama 3.2 tends to truncate answers at a natural-sounding stopping point that is often too short for multi-source policy questions. Setting 800 tokens explicitly signals that a full-length response is expected.

---

### 8. Source Attribution

Every response is accompanied by a deduplicated list of sources. The system:

1. Collects all retrieved chunks that informed the answer
2. Groups them by document ID or URL (to avoid listing the same document twice when two chunks from it were retrieved)
3. Keeps the chunk with the highest similarity score as the representative for each document
4. Displays the document title, theme, URL, citation string, and best similarity score

This allows a reader to verify every claim made in the answer against the original EU policy document, satisfying the assignment's grounding requirement and reflecting real-world responsible AI disclosure practices.

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
