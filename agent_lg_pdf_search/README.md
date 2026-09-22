# agent_lg_pdf_search

Terminal agent that answers questions about a PDF document.

## Requirements

```powershell
pip install langchain langchain-openai langchain-text-splitters pypdf numpy requests
```

| Package | Used for |
|---|---|
| `langchain`, `langchain-openai` | the agent loop, `OpenAIEmbeddings` against llama-server |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` in `split()` |
| `pypdf` | reading the PDF in `read_pages()` |
| `numpy` | similarity search inside `InMemoryVectorStore` |
| `requests` | the `/tokenize` endpoint in `count_tokens()` |

## Servers

Two llama-server processes, each in its own PowerShell window, both started from the repo root.

| Port | Role | Model | Runs on |
|---|---|---|---|
| 8081 | chat | Qwen3-8B `IQ4_XS` | GPU |
| 8082 | embeddings | EmbeddingGemma 300M `Q8_0` | CPU |

8080 is taken at every Windows start by MiniTool ShadowMaker's MTAgentService.

### Chat: Qwen3-8B

Download (4.56 GB): https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-IQ4_XS.gguf

```powershell
.\LIama\llama-server.exe -m .\Qwen_Qwen3-8B-IQ4_XS.gguf -c 8192 --port 8081
```

### Embeddings: EmbeddingGemma 300M

Download (334 MB): https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf

```powershell
.\LIama\llama-server.exe -m .\embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 0 --port 8082
```

`-ngl 0` keeps the whole EmbeddingGemma 300M on the CPU
