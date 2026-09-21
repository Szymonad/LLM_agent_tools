from pathlib import Path

# --- llama-server ---

SERVER = "http://127.0.0.1:8081"

# --- pdf ---
# Next to this file, not in whatever folder the terminal happens to be in.
PDF_PATH = Path(__file__).with_name("pdf") / "Selected_Filtration_Methods_of_ISO-16610.pdf"
