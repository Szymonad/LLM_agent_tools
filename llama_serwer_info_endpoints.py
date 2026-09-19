"""Queries every informational endpoint of a running llama-server and prints a report.

Source of truth for endpoint shapes: llama.cpp tools/server/README.md
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Endpoints deliberately NOT called, and why:
  POST /props                          changes server global properties (needs --props anyway)
  POST /slots/{id}?action=save|restore|erase   mutates slot cache files on disk
  POST /lora-adapters                  changes the active LoRA scales
  POST /models/load, POST /models/unload       loads/unloads a model, changes server state
  POST /models                         downloads a model from the internet
  GET /models/sse                      server-sent events stream, would block forever
  POST /completion, /v1/completions, /v1/chat/completions, /v1/responses, /v1/messages,
  /embedding, /embeddings, /v1/embeddings, /reranking, /infill
                                        run inference on the GPU, not server information
"""

import requests

SERVER = "http://127.0.0.1:8081"
TIMEOUT = 30

SAMPLE_TEXT = "How many departments are there?"
SAMPLE_MESSAGES = [{"role": "user", "content": SAMPLE_TEXT}]
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all tables in the database.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


# --- helpers ---


def get(path):
    """Calls a GET endpoint. Returns {"ok": bool, "status": int, "data": ..., "error": str}."""
    try:
        response = requests.get(SERVER + path, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"ok": False, "status": None, "data": None, "error": str(exc)}
    if response.status_code != 200:
        return {"ok": False, "status": response.status_code, "data": None, "error": response.text}
    try:
        data = response.json()
    except ValueError:
        # /metrics returns Prometheus plain text, not JSON.
        data = response.text
    return {"ok": True, "status": response.status_code, "data": data, "error": ""}


def post(path, body):
    """Calls a POST endpoint. Same result shape as get()."""
    try:
        response = requests.post(SERVER + path, json=body, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return {"ok": False, "status": None, "data": None, "error": str(exc)}
    if response.status_code != 200:
        return {"ok": False, "status": response.status_code, "data": None, "error": response.text}
    try:
        data = response.json()
    except ValueError as exc:
        return {"ok": False, "status": response.status_code, "data": None, "error": str(exc)}
    return {"ok": True, "status": response.status_code, "data": data, "error": ""}


# --- GET endpoints ---


def health():
    """Information about server readiness: whether the model finished loading."""
    return get("/health")


def props():
    """Information about the server and the loaded model: model path, context size, chat
    template capabilities."""
    return get("/props")


def slots():
    """Information about the current task: per-slot processing state and context size."""
    return get("/slots")


def v1_models():
    """Information about the loaded model in OpenAI-compatible format: id, param count,
    training context size."""
    return get("/v1/models")


def models():
    """Information about the server in router mode: list of models known to the router."""
    return get("/models")


def lora_adapters():
    """Information about the loaded model: LoRA adapters currently attached."""
    return get("/lora-adapters")


def metrics():
    """Information about the current task: Prometheus metrics such as tokens processed and
    requests handled."""
    return get("/metrics")


# --- POST endpoints ---


def tokenize(text):
    """Information about the loaded model's tokenizer: token ids for a given text."""
    return post("/tokenize", {"content": text})


def detokenize(tokens):
    """Information about the loaded model's tokenizer: text reconstructed from token ids."""
    return post("/detokenize", {"tokens": tokens})


def apply_template(messages, tools):
    """Information about the current task: prompt string produced by the model's chat
    template for a given conversation."""
    return post("/apply-template", {"messages": messages, "tools": tools})


def chat_input_tokens(messages):
    """Information about the current task: number of prompt tokens a chat completion request
    would consume."""
    return post("/v1/chat/completions/input_tokens", {"model": "local", "messages": messages})


def responses_input_tokens(text):
    """Information about the current task: number of prompt tokens a responses request would
    consume."""
    return post("/v1/responses/input_tokens", {"model": "local", "input": text})


def messages_count_tokens(messages):
    """Information about the current task: number of prompt tokens an Anthropic-style messages
    request would consume."""
    return post("/v1/messages/count_tokens", {"model": "local", "messages": messages})


# --- report ---


def print_result(name, result):
    if not result["ok"]:
        print("FAILED %s %s %s" % (name, result["status"], result["error"]))
    else:
        print("OK %s" % name)


def main():
    health_result = health()
    print_result("/health", health_result)
    if health_result["ok"]:
        print("  status:", health_result["data"]["status"])

    props_result = props()
    print_result("/props", props_result)
    if props_result["ok"]:
        data = props_result["data"]
        print("  model_path:", data.get("model_path"))
        print("  n_ctx:", data.get("default_generation_settings", {}).get("n_ctx"))
        print("  total_slots:", data.get("total_slots"))
        print("  chat_template_caps:", data.get("chat_template_caps"))

    slots_result = slots()
    print_result("/slots", slots_result)
    if slots_result["ok"]:
        for slot in slots_result["data"]:
            print("  slot", slot["id"], "n_ctx:", slot["n_ctx"], "is_processing:", slot["is_processing"])

    v1_models_result = v1_models()
    print_result("/v1/models", v1_models_result)
    if v1_models_result["ok"]:
        model_info = v1_models_result["data"]["data"][0]
        print("  id:", model_info["id"])
        meta = model_info.get("meta") or {}
        print("  n_params:", meta.get("n_params"))
        print("  n_ctx_train:", meta.get("n_ctx_train"))
        print("  n_vocab:", meta.get("n_vocab"))

    models_result = models()
    print_result("/models", models_result)
    if models_result["ok"]:
        print("  count:", len(models_result["data"]["data"]))

    lora_result = lora_adapters()
    print_result("/lora-adapters", lora_result)
    if lora_result["ok"]:
        print("  count:", len(lora_result["data"]))

    metrics_result = metrics()
    print_result("/metrics", metrics_result)
    if metrics_result["ok"]:
        print("  lines:", len(metrics_result["data"].splitlines()))

    tokenize_result = tokenize(SAMPLE_TEXT)
    print_result("/tokenize", tokenize_result)
    tokens = []
    if tokenize_result["ok"]:
        tokens = tokenize_result["data"]["tokens"]
        print("  token count:", len(tokens))

    detokenize_result = detokenize(tokens)
    print_result("/detokenize", detokenize_result)
    if detokenize_result["ok"]:
        print("  text:", detokenize_result["data"]["content"])

    apply_template_result = apply_template(SAMPLE_MESSAGES, SAMPLE_TOOLS)
    print_result("/apply-template", apply_template_result)
    if apply_template_result["ok"]:
        prompt = apply_template_result["data"]["prompt"]
        prompt_tokens = tokenize(prompt)
        prompt_token_count = len(prompt_tokens["data"]["tokens"]) if prompt_tokens["ok"] else None
        print("  prompt length (chars):", len(prompt))
        print("  prompt length (tokens):", prompt_token_count)

    chat_input_tokens_result = chat_input_tokens(SAMPLE_MESSAGES)
    print_result("/v1/chat/completions/input_tokens", chat_input_tokens_result)
    if chat_input_tokens_result["ok"]:
        print("  input_tokens:", chat_input_tokens_result["data"]["input_tokens"])

    responses_input_tokens_result = responses_input_tokens(SAMPLE_TEXT)
    print_result("/v1/responses/input_tokens", responses_input_tokens_result)
    if responses_input_tokens_result["ok"]:
        print("  input_tokens:", responses_input_tokens_result["data"]["input_tokens"])

    messages_count_tokens_result = messages_count_tokens(SAMPLE_MESSAGES)
    print_result("/v1/messages/count_tokens", messages_count_tokens_result)
    if messages_count_tokens_result["ok"]:
        print("  input_tokens:", messages_count_tokens_result["data"]["input_tokens"])


if __name__ == "__main__":
    main()
