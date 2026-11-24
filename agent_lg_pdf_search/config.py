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

#: Name of the parser, written to ``run.json`` of every evaluation run.
PARSER = "docling-rapidocr+qwen3vl4b+docname"
#: CPU threads for Docling; the GPU is taken by the chat model.
DOCLING_THREADS = 8
#: A page whose text layer has fewer characters than this counts as a scan (``origin`` ``ocr``).
SCAN_TEXT_CHARS = 20

# --- picture descriptions ---

#: Address of the vision server (Qwen3-VL-4B), needed only while the index is built.
VLM_SERVER = "http://127.0.0.1:8083"
#: Model file of the vision server; part of the description cache key.
VLM_MODEL = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
#: Longer side, in pixels, a picture is scaled down to before it goes to the model.
VLM_MAX_SIDE = 1024
VLM_MAX_TOKENS = 400
#: Seconds to wait for one description; reading the image is slow on a laptop GPU.
VLM_TIMEOUT = 600
#: Pictures smaller than this share of the page (logos, icons) get no description.
MIN_PICTURE_AREA = 0.02

# --- embeddings ---

#: Address of the embedding server (EmbeddingGemma).
EMBED_SERVER = "http://127.0.0.1:8082"
#: Seconds to wait for either server. Without it, a hung server stalls the agent forever.
HTTP_TIMEOUT = 120
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

# --- agent ---

#: Rewrite every question into a standalone search query before the search; off by default because it made retrieval about 8 times slower without finding more pages.
REPHRASE = False
