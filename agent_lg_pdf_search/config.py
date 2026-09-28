from pathlib import Path

# --- llama-server ---

SERVER = "http://127.0.0.1:8081"

# --- pdf ---
# Every *.pdf in this folder goes into one index.
PDF_DIR = Path(__file__).with_name("pdf")

# --- embeddings ---
EMBED_SERVER = "http://127.0.0.1:8082"
# Without this, a hung server stalls the agent forever.
HTTP_TIMEOUT = 30
# EmbeddingGemma was trained with these prefixes on queries and passages; without them retrieval gets worse.
QUERY_PREFIX = "task: search result | query: "
PASSAGE_PREFIX = "title: none | text: "

# --- chunks ---
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 50

# --- index ---
# Built by "python index.py" and read by the agent; rebuild after changing the PDF,
# the chunk settings or the embedding model.
INDEX_PATH = Path(__file__).with_name("index.json")
# How many chunks reach the model; 5 chunks of 400 tokens is about 2000.
TOP_K = 5
