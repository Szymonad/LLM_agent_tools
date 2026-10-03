# Local LLM agents

Agents and tools built on a local llama-server. Everything is started from this folder.

## Layout

| Folder | What it is |
|---|---|
| `agent_lg_pdf_search/` | LangGraph agent that answers questions about PDF documents |
| `agent_sql/` | agent that answers questions about the Oracle database |
| `mcp_oracle/` | notes on reaching Oracle through MCP (`cline/`, `llama/`) |
| `narzedzia/` | scripts that inspect a running llama-server |
| `archiwum/` | first experiments, no longer used |
| `pdf/` | notes |
| `modele/` | llama-server binaries (`LIama/`), chat models (`llm/`), embedding models (`embeddingi/`); not in git |

## Starting

Servers, chat on port 8081 and embeddings on port 8082:

```powershell
.\start_llama.bat
```

PDF agent:

```powershell
python -m agent_lg_pdf_search.index
python -m agent_lg_pdf_search.agent
python -m agent_lg_pdf_search.eval.evaluate --compare
```

SQL agent:

```powershell
python -m agent_sql.agent
```

Server tools:

```powershell
python -m narzedzia.llama_scanner
python -m narzedzia.llama_serwer_info_endpoints
python -m narzedzia.show_tokens
python -m narzedzia.server_info
```

Details, requirements and credentials are in the README of each agent.
