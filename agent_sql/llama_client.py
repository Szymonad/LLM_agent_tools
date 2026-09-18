"""Talks to llama-server over HTTP, with no knowledge of what the conversation is about."""
import requests

from config import HTTP_TIMEOUT, SERVER


def context_full(response):
    """True when the server refused the request because the history no longer fits.

    exceed_context_size_error is the only 400 type recoverable by shortening the history,
    so every other 400 stays an error.
    """
    try:
        return response.json()["error"]["type"] == "exceed_context_size_error"
    except (ValueError, KeyError, TypeError):
        return False


class LlamaClient:
    def __init__(self, server=SERVER, tools=None, timeout=HTTP_TIMEOUT):
        self.server = server
        self.tools = tools
        self.timeout = timeout
        # Set by chat() from the server's "usage" field, sent by every OpenAI-compatible server.
        # The number is roughly 10 tokens behind, since it excludes the role markers of the turn
        # not sent yet.
        self.last_total_tokens = "?"

    def post(self, path, body):
        response = requests.post(self.server + path, json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get(self, path):
        response = requests.get(self.server + path, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def chat(self, messages):
        body = self.post("/v1/chat/completions", {"messages": messages, "tools": self.tools, "temperature": 0})
        self.last_total_tokens = body.get("usage", {}).get("total_tokens", "?")
        return body["choices"][0]

    def context_window(self):
        """Window size the server was started with (-c)"""
        return self.get("/props")["default_generation_settings"]["n_ctx"]

    def render_prompt(self, messages):
        """/apply-template is llama.cpp's own endpoint: it glues the messages into the flat
        text the model receives, tool definitions included, without generating anything."""
        return self.post("/apply-template", {"messages": messages, "tools": self.tools})["prompt"]
