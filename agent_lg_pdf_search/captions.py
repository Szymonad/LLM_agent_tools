"""Descriptions of the pictures in the PDFs, written by a vision model.

The model (Qwen3-VL-4B) runs on its own llama-server, ``VLM_SERVER``,
started with ``start_vlm.bat`` while the index is built. Every description
is kept in a JSON file keyed by the SHA-256 of the image and the request
settings, so a rebuild asks the model only about pictures it has not seen,
and a changed prompt or model asks again.
"""

import base64
import hashlib
import io
import json
import os

import requests

from agent_lg_pdf_search.config import VLM_MAX_SIDE, VLM_MAX_TOKENS, VLM_MODEL, VLM_SERVER, VLM_TIMEOUT

PROMPT = """Describe this picture from a document so that it can be found by a text search.
If it is a chart, give its type, title, axes, legend and every value you can read.
If it is a table or a diagram, give its structure and all its text.
If it is a photo or an illustration, say what it shows.
Copy all visible text and numbers exactly. No introduction, at most 200 words."""
#: Part of every cache key: a description made with other settings is not reused.
SETTINGS = f"{VLM_MODEL}|{VLM_MAX_SIDE}|{VLM_MAX_TOKENS}|{PROMPT}".encode()


class CaptionCache:
    """Descriptions already written, loaded from and saved to one JSON file.

    Args:
        path (pathlib.Path): The JSON file; it need not exist yet.
    """

    def __init__(self, path):
        self.path = path
        self.items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.new = 0
        self.checked = False

    def caption(self, image):
        """Description of a picture, from the file or from the model.

        Args:
            image (PIL.Image.Image): The picture, scaled down here to
                ``VLM_MAX_SIDE`` pixels on its longer side.

        Returns:
            str: The description, stripped; empty when the model wrote
            nothing, which is not kept, so the next build asks again.
        """
        image = image.convert("RGB")
        image.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        data = buffer.getvalue()
        key = hashlib.sha256(SETTINGS + data).hexdigest()
        if key not in self.items:
            if not self.checked:
                check_server()
                self.checked = True
            caption = ask_model(data)
            if not caption:
                return ""
            self.items[key] = caption
            self.new += 1
            # Saved often: the build runs for hours and must not lose its work when stopped.
            if self.new % 10 == 0:
                self.save()
        return self.items[key]

    def save(self):
        """Write the file through a temporary one, so a stop while writing cannot cut it."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.items, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(temporary, self.path)


def ask_model(png):
    """Ask the vision model to describe one PNG image.

    Raises:
        requests.HTTPError: If the server answers with an error status.
        requests.ConnectionError: If the server cannot be reached.
    """
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    response = requests.post(
        f"{VLM_SERVER}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}
            ],
            "temperature": 0,
            "max_tokens": VLM_MAX_TOKENS,
        },
        timeout=VLM_TIMEOUT,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        print(f"description cut at {VLM_MAX_TOKENS} tokens")
    return (choice["message"]["content"] or "").strip()


def check_server():
    """Stop early, with a hint, when the vision server is not running."""
    try:
        requests.get(f"{VLM_SERVER}/health", timeout=10).raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"vision server at {VLM_SERVER} not reachable ({error}); start it with start_vlm.bat") from error
