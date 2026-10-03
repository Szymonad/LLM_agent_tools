"""Prints what the running llama-server holds: context, KV cache cost, model, GPU memory."""
import json
import struct
import subprocess
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8080"
MODEL_FILE = Path(r"C:\Users\szymo\Desktop\kodzik\stacjonarny llm\models\llm\Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf")
KV_BYTES = 2  # f16, llama.cpp default for the KV cache

MIB = 1024 * 1024


def get(path):
    with urllib.request.urlopen(SERVER + path, timeout=5) as response:
        return json.load(response)


def read_gguf_metadata(path):
    """Reads the key-value header of a GGUF file, stops before the tensor data."""
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    with open(path, "rb") as f:
        def read(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def read_string():
            return f.read(read("<Q")).decode("utf-8", errors="replace")

        def read_value(value_type):
            if value_type == 8:
                return read_string()
            if value_type == 9:
                item_type, count = read("<I"), read("<Q")
                # Arrays like the 128k-token vocabulary are skipped, only their length is kept.
                for _ in range(count):
                    read_value(item_type)
                return f"<array of {count}>"
            return read(scalar[value_type])

        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        read("<I")  # version
        read("<Q")  # tensor count
        kv_count = read("<Q")

        metadata = {}
        for _ in range(kv_count):
            key = read_string()
            metadata[key] = read_value(read("<I"))
        return metadata


def gpu_memory():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    total, used, free = (int(x) for x in out.strip().split(","))
    return total, used, free


def main():
    print("health:", get("/health")["status"])

    props = get("/props")
    slots = get("/slots")
    meta = get("/v1/models")["data"][0]["meta"]
    gguf = read_gguf_metadata(MODEL_FILE)

    print(gguf['1'])

    arch = gguf["general.architecture"]
    layers = gguf[f"{arch}.block_count"]
    heads = gguf[f"{arch}.attention.head_count"]
    kv_heads = gguf[f"{arch}.attention.head_count_kv"]
    head_dim = gguf.get(f"{arch}.attention.key_length", meta["n_embd"] // heads)

    n_ctx = props["default_generation_settings"]["n_ctx"]
    kv_per_token = layers * 2 * kv_heads * head_dim * KV_BYTES

    # print("\n--- model ---")
    # print(f"file:              {props['model_path']}")
    # print(f"quantization:      {props['model_ftype']}")
    # print(f"parameters:        {meta['n_params'] / 1e9:.2f} B")
    # print(f"weights size:      {meta['size'] / MIB:.0f} MiB")
    # print(f"vocabulary:        {meta['n_vocab']} tokens")
    # print(f"trained context:   {meta['n_ctx_train']} tokens")

    # print("\n--- architecture (from GGUF header) ---")
    # print(f"layers:            {layers}")
    # print(f"attention heads:   {heads}  (Q)")
    # print(f"KV heads:          {kv_heads}  (GQA: {heads // kv_heads} Q heads share one KV pair)")
    # print(f"head dim:          {head_dim}")
    # print(f"embedding:         {meta['n_embd']}")

    # print("\n--- context and KV cache ---")
    # print(f"context (-c):      {n_ctx} tokens")
    # print(f"slots:             {props['total_slots']}  (parallel conversations sharing that context)")
    # print(f"KV per token:      {kv_per_token / 1024:.0f} KiB  = {layers} x 2 x {kv_heads} x {head_dim} x {KV_BYTES} B")
    # print(f"KV for {n_ctx:>5}:     {kv_per_token * n_ctx / MIB:.0f} MiB  (reserved at startup)")
    # busy = [s["id"] for s in slots if s.get("is_processing")]
    # print(f"busy slots:        {busy or 'none'}")

    # params = props["default_generation_settings"]["params"]
    # print("\n--- default sampling (overridden per request) ---")
    # for key in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "n_predict"):
    #     print(f"{key + ':':<19}{params[key]:g}" if isinstance(params[key], float) else f"{key + ':':<19}{params[key]}")

    # caps = props.get("chat_template_caps", {})
    # print("\n--- chat template ---")
    # print(f"system role:       {caps.get('supports_system_role')}")
    # print(f"parallel tools:    {caps.get('supports_parallel_tool_calls')}")

    # total, used, free = gpu_memory()
    # print("\n--- GPU (nvidia-smi, whole card) ---")
    # print(f"total:             {total} MiB")
    # print(f"used:              {used} MiB")
    # print(f"free:              {free} MiB")
    # print("\nexact per-buffer sizes: start the server with -lv 4")


if __name__ == "__main__":
    main()
