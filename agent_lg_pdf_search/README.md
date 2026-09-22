# agent_lg_pdf_search

Terminal agent that answers questions about a PDF document.

## Requirements

Python 3.14. Everything this agent needs:

```powershell
pip install langchain
pip install langchain-openai
pip install langchain-text-splitters
pip install pypdf
pip install numpy
pip install requests
```

Or in one line:

```powershell
pip install langchain langchain-openai langchain-text-splitters pypdf numpy requests
```

| Package | Version it was run with | Used for |
|---|---|---|
| `langchain` | 1.4.2 | `create_agent` and the middleware in `agent.py` |
| `langchain-openai` | 1.6.2 | `OpenAIEmbeddings` and `ChatOpenAI` against llama-server |
| `langchain-text-splitters` | 1.1.2 | `RecursiveCharacterTextSplitter` in `split()` |
| `pypdf` | 6.14.2 | reading the PDF in `read_pages()` |
| `numpy` | 2.5.3 | similarity search inside `InMemoryVectorStore` |
| `requests` | 2.34.2 | the `/tokenize` endpoint in `count_tokens()` |

Pulled in automatically, no need to install them by hand: `langchain-core`, `langgraph`,
`pydantic` (by `langchain`), `openai`, `tiktoken` (by `langchain-openai`).

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
