"""Settings shared by the index, the agent and the evaluation script.

Addresses of the two llama-server instances, the folder with PDFs, the
chunk sizes and the location of the saved index.
"""

from pathlib import Path

# --- llama-server ---

#: Address of the chat server.
SERVER = "http://127.0.0.1:8081"

# --- pdf ---

#: Folder with the PDFs. Every ``*.pdf`` in it goes into one index.
PDF_DIR = Path(__file__).with_name("pdf")

# --- embeddings ---

#: Address of the embedding server (EmbeddingGemma).
EMBED_SERVER = "http://127.0.0.1:8082"
#: Seconds to wait for either server. Without it, a hung server stalls the agent forever.
HTTP_TIMEOUT = 30
#: Prefix put in front of every search query. EmbeddingGemma was trained with it; without it retrieval gets worse.
QUERY_PREFIX = "task: search result | query: "
#: Prefix put in front of every chunk. ``none`` stands for a missing document title.
PASSAGE_PREFIX = "title: none | text: "
EMBED_DIM = 768

# --- chunks ---

#: Maximum size of a chunk, in tokens of the embedding model.
CHUNK_TOKENS = 200
#: Tokens shared by two neighbouring chunks.
CHUNK_OVERLAP = 50

# --- index ---

#: File the index is saved to. Written by ``index.build`` and read by the agent;
#: rebuild after changing the PDFs, the chunk settings or the embedding model.
INDEX_PATH = Path(__file__).with_name("index.json")
#: How many chunks reach the model on each question.
TOP_K = 5
