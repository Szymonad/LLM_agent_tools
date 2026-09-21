from pathlib import Path

# --- llama-server ---

SERVER = "http://127.0.0.1:8081"

# --- pdf ---
# Next to this file, not in whatever folder the terminal happens to be in.
PDF_PATH = Path(__file__).with_name("pdf") / "Selected_Filtration_Methods_of_ISO-16610.pdf"

# --- chunks ---
# The embedding server rejects inputs over 512 tokens; at ~4 characters per token
# 1200 characters leaves room for the prefix and for text that tokenizes worse.
CHUNK_CHARS = 1200
# Text shared by neighbouring chunks, so a sentence cut at the boundary is not lost from both.
CHUNK_OVERLAP = 150
