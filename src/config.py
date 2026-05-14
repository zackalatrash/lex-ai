"""Central configuration for the NLP Policy Chatbot."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "NLP Policy Chatbot"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "dataset" / "20260331_dataset2.xlsx"

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
SCENARIO_DIR = DOCUMENTS_DIR / "Example Conversation Themes"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

PROCESSED_DOCUMENTS_PATH = PROCESSED_DATA_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
DOWNLOAD_REPORT_PATH = PROCESSED_DATA_DIR / "download_report.json"

VECTOR_EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
VECTOR_CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.jsonl"
VECTOR_MANIFEST_PATH = VECTOR_STORE_DIR / "manifest.json"

EXCEL_SHEET_NAME = "Sheet1"

DEFAULT_BUILD_LIMIT = 10
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_MAX_CHUNKS_PER_DOC = 2

MMR_LAMBDA = 0.7            # 0 = pure diversity, 1 = pure relevance
MMR_FETCH_K_MULTIPLIER = 4  # candidates fetched before MMR = top_k × this
MIN_EVIDENCE_SIMILARITY = 0.55  # skip LLM call if best chunk scores below this

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BATCH_SIZE = 4  # lower = less MPS/GPU memory per batch; increase if you have headroom
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "llama3.1"
OPENAI_API_KEY_FOR_OLLAMA = "ollama"

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "NLPPolicyChatbot/1.0 (+student assignment)"

MAX_HISTORY_TURNS = 4
MAX_CONTEXT_CHARS_PER_CHUNK = 1600

RETRIEVAL_SIMILARITY_FLOOR = 0.35
RELEVANCE_THRESHOLD = 0.40
OUT_OF_SCOPE_THRESHOLD = 0.60
