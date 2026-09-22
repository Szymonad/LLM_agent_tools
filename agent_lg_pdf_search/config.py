from pathlib import Path

# --- llama-server ---

SERVER = "http://127.0.0.1:8081"

# --- pdf ---
# Next to this file, not in whatever folder the terminal happens to be in.
PDF_PATH = Path(__file__).with_name("pdf") / "repeated_terms.pdf"

# --- embeddings ---
EMBED_SERVER = "http://127.0.0.1:8082"
# Without this, a hung server stalls the agent forever.
HTTP_TIMEOUT = 60
# EmbeddingGemma was trained with these prefixes on queries and passages; without them retrieval gets worse.
QUERY_PREFIX = "task: search result | query: "
PASSAGE_PREFIX = "title: none | text: "

# --- chunks ---
# Counted in EmbeddingGemma tokens: the server rejects inputs over 512, and formulas
# in the PDF take up to one token per character, so a character limit gives no guarantee.
CHUNK_TOKENS = 400
# Text shared by neighbouring chunks, so a sentence cut at the boundary is not lost from both.
CHUNK_OVERLAP = 50

# --- index ---
# Built by "python index.py" and read by the agent; rebuild after changing the PDF,
# the chunk settings or the embedding model.
INDEX_PATH = Path(__file__).with_name("index.json")
