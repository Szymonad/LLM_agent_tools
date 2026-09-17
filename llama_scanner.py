import struct
import subprocess

import requests

from llama_serwer_info_endpoints import apply_template, props, slots, tokenize, v1_models

SERVER = "http://127.0.0.1:8080"
TIMEOUT = 30

PRZYKLADOWE_NARZEDZIE = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

GGUF_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


# --- server ---


def data(result):
    """Unwraps a result from llama_serwer_info_endpoints, raising when the call failed."""
    if not result["ok"]:
        raise RuntimeError(f"{result['status']}: {result['error']}")
    return result["data"]


def capabilities():
    """What the model's chat template supports."""
    return data(props())["chat_template_caps"]


def supports_tools():
    """The one field that decides whether a model can drive an agent.

    Measured False on Llama-PLLuM and Bielik - tool definitions vanished from the
    rendered prompt without any error.
    """
    return capabilities().get("supports_tools", False)


def context_window():
    """Window size the server was started with (-c)."""
    return data(props())["default_generation_settings"]["n_ctx"]


def count_tokens(text):
    return len(data(tokenize(text))["tokens"])


def tools_reach_model(messages, tools):
    """Compares the prompt rendered with and without tools.

    Equal strings mean the template silently drops the tool definitions - measured on
    Llama-PLLuM, where both renders were 157 tokens.
    """
    with_tools = count_tokens(data(apply_template(messages, tools))["prompt"])
    without_tools = count_tokens(data(apply_template(messages, None))["prompt"])
    return {"with": with_tools, "without": without_tools, "reaches": with_tools != without_tools}


def template_accepts(messages, tools=None):
    """Whether the chat template accepts this shape of history.

    Measured rejections on the Llama 3.1 template: history with no user message, and a
    trailing assistant message holding tool calls.
    """
    result = apply_template(messages, tools)
    return result["ok"], result["error"]


def chat(messages, tools=None, temperature=0):
    """One real generation. Returns the choice plus the usage block."""
    payload = {"model": "local", "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    response = requests.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    body = response.json()
    return {"choice": body["choices"][0], "usage": body.get("usage", {})}


# --- gguf file ---


def _read_gguf_string(data, position):
    length = struct.unpack_from("<Q", data, position)[0]
    position += 8
    text = data[position: position + length].decode("utf-8")
    return text, position + length


def _read_gguf_value(data, position):
    value_type = struct.unpack_from("<I", data, position)[0]
    position += 4
    if value_type == 8:
        return _read_gguf_string(data, position)
    if value_type == 9:
        element_type = struct.unpack_from("<I", data, position)[0]
        position += 4
        count = struct.unpack_from("<Q", data, position)[0]
        position += 8
        for _ in range(count):
            if element_type == 8:
                _, position = _read_gguf_string(data, position)
            else:
                position += struct.calcsize(GGUF_SCALAR_FORMATS[element_type])
        return None, position
    fmt = GGUF_SCALAR_FORMATS[value_type]
    size = struct.calcsize(fmt)
    value = struct.unpack_from(fmt, data, position)[0]
    return value, position + size


def gguf_metadata(source, max_mb=24):
    """Key-value header of a GGUF file. Works on a local path or on a URL.

    For a URL only the first max_mb are fetched with a Range request, which is enough for
    the header and avoids downloading gigabytes.
    """
    if source.startswith("http://") or source.startswith("https://"):
        headers = {"Range": f"bytes=0-{max_mb * 1024 * 1024 - 1}"}
        response = requests.get(source, headers=headers, timeout=TIMEOUT)
        data = response.content
    else:
        with open(source, "rb") as gguf_file:
            data = gguf_file.read(max_mb * 1024 * 1024)

    assert data[:4] == b"GGUF"
    position = 8  # skip magic and uint32 version
    tensor_count, kv_count = struct.unpack_from("<QQ", data, position)
    position += 16
    del tensor_count

    metadata = {}
    for _ in range(kv_count):
        key, position = _read_gguf_string(data, position)
        value, position = _read_gguf_value(data, position)
        metadata[key] = value
    return metadata


def chat_template(source):
    """The Jinja chat template baked into a GGUF file."""
    return gguf_metadata(source).get("tokenizer.chat_template", "")


def template_has_tools(source):
    """True when the template has a branch for tool definitions.

    This is checkable before downloading the file, because the GGUF header sits at its start.
    """
    template = chat_template(source)
    return "tools" in template or "tool_call" in template


def kv_bytes_per_token(meta):
    """Bytes one token costs in the KV cache, from the GGUF header.

    layers * 2 * kv_heads * head_dim * 2 bytes for the f16 cache.
    Measured: 128 KiB for Llama 3.1 8B, 144 KiB for Qwen3-8B, 132 KiB for Qwen3.5-4B.
    """
    arch = meta["general.architecture"]
    layers = meta[f"{arch}.block_count"]
    kv_heads = meta[f"{arch}.attention.head_count_kv"]
    head_dim = meta.get(
        f"{arch}.attention.key_length",
        meta[f"{arch}.embedding_length"] // meta[f"{arch}.attention.head_count"],
    )
    return layers * 2 * kv_heads * head_dim * 2


def vram_needed(file_gib, n_ctx, kv_per_token, overhead_gib=0.3):
    """Total VRAM for weights plus KV cache plus compute buffers, in GiB."""
    kv_cache_gib = (n_ctx * kv_per_token) / (1024 ** 3)
    return file_gib + kv_cache_gib + overhead_gib


def largest_context(file_gib, kv_per_token, free_gib, overhead_gib=0.3):
    """How many tokens of context still fit on the card. Returns an int, possibly 0."""
    kv_budget_gib = free_gib - file_gib - overhead_gib
    if kv_budget_gib <= 0:
        return 0
    return int((kv_budget_gib * (1024 ** 3)) / kv_per_token)


# --- huggingface ---


def hf_chat_template(repo):
    """Chat template from a model repo, without downloading any weights.

    Returns None when the repo is gated - HTTP 401 - which is the case for Bielik,
    meta-llama and google. For those the template can still be read from a GGUF mirror
    with gguf_metadata on the file URL.
    """
    url = f"https://huggingface.co/{repo}/raw/main/tokenizer_config.json"
    response = requests.get(url, timeout=TIMEOUT)
    if not response.ok:
        return None
    return response.json().get("chat_template")


def hf_has_tools(repo):
    """True/False/None. None means the repo could not be read."""
    template = hf_chat_template(repo)
    if template is None:
        return None
    return "tools" in template or "tool_call" in template


def hf_gguf_files(repo, max_gib=4.6):
    """GGUF files in a repo that fit the budget, as (name, size_gib), smallest last."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    response = requests.get(url, timeout=TIMEOUT)
    siblings = response.json()["siblings"]
    files = []
    for sibling in siblings:
        name = sibling["rfilename"]
        size = sibling.get("size")
        if not name.endswith(".gguf") or not size:
            continue
        size_gib = size / (1024 ** 3)
        if size_gib <= max_gib:
            files.append((name, size_gib))
    files.sort(key=lambda item: item[1], reverse=True)
    return files


# --- graphics card ---


def vram():
    """Total, used and free VRAM in MiB from nvidia-smi. None when the tool is missing."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=TIMEOUT,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    total, used, free = (int(value) for value in output.decode().strip().split(","))
    return {"total_mib": total, "used_mib": used, "free_mib": free}


# --- main ---


def main():
    props_result = props()
    if not props_result["ok"]:
        print("Server is not reachable on", SERVER)
        return

    print("Server is alive on", SERVER)

    info = data(v1_models())["data"][0]["meta"]
    slot_list = data(slots())
    n_ctx = props_result["data"]["default_generation_settings"]["n_ctx"]
    slot_count = len(slot_list)
    print(f"\nModel: {info.get('n_params')} params, trained context {info.get('n_ctx_train')}")
    print(f"Server context window: {n_ctx}")
    # Read from the slot, never n_ctx divided by the slot count: with kv_unified every
    # slot sees the whole window, so the division reports a quarter of the truth.
    per_slot = slot_list[0]["n_ctx"] if slot_list else "?"
    print(f"Slots: {slot_count}, context per slot: {per_slot}")

    print("\nCapabilities:")
    for key, value in capabilities().items():
        marker = " <--" if key == "supports_tools" else ""
        print(f"  {key}: {value}{marker}")

    print("\nVRAM:")
    gpu_memory = vram()
    if gpu_memory is None:
        print("  nvidia-smi not available")
    else:
        print(f"  total {gpu_memory['total_mib']} MiB, used {gpu_memory['used_mib']} MiB, "
              f"free {gpu_memory['free_mib']} MiB")

    print("\nTools reaching the model:")
    sample_messages = [{"role": "user", "content": "What is the weather in Warsaw?"}]
    reach = tools_reach_model(sample_messages, PRZYKLADOWE_NARZEDZIE)
    print(f"  with tools: {reach['with']} tokens, without: {reach['without']} tokens, "
          f"reaches: {reach['reaches']}")

    print("\nTemplate acceptance:")
    no_user = [{"role": "system", "content": "You are a helpful assistant."}]
    accepted, message = template_accepts(no_user)
    print(f"  history with no user message: accepted={accepted} {message}")

    trailing_tool_calls = [
        {"role": "user", "content": "What is the weather in Warsaw?"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'}}
        ]},
    ]
    accepted, message = template_accepts(trailing_tool_calls, PRZYKLADOWE_NARZEDZIE)
    print(f"  assistant with trailing tool_calls: accepted={accepted} {message}")

    tool_call_and_result = trailing_tool_calls + [
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 18}'}
    ]
    accepted, message = template_accepts(tool_call_and_result, PRZYKLADOWE_NARZEDZIE)
    print(f"  tool_call followed by tool result: accepted={accepted} {message}")


if __name__ == "__main__":
    main()
