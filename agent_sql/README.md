# agent_sql

Terminal agent that answers questions about the Oracle database.
The model writes the SQL itself. Two helper tools let it look at the schema first:

| Tool | What it does |
|---|---|
| `list_tables` | names of all tables |
| `describe_table` | columns and data types of one table |
| `run_query` | runs one read-only `SELECT` |
| `answer_user` | final answer to the user |

## Files

- `agent.py` - terminal loop and the conversation with the model
- `tools.py` - the contract with the model: system prompt, tool declarations, and the map to `db.py`
- `db.py` - database connection, the query functions and the read-only guard
- `check.sql` - queries for auditing an answer against the database by hand
- `agent.log` - created on first run: questions, tool calls and every SQL sent to the database

## Requirements

- llama-server running on `http://127.0.0.1:8080`
- `pip install oracledb`
- Oracle XE with service `xepdb1` on port 1521

## Which models work

A model only works here if its chat template declares tools. Weights, size and language
do not decide this - the template does, and it is a text file the publisher writes.

| Model | Template | Tools | Verdict |
|---|---|---|---|
| Qwen3-8B `IQ4_XS` (4.25 GiB) | ChatML with tools | yes | best tested; fluent Polish, correct Oracle syntax |
| Meta-Llama-3.1-8B-Instruct `IQ4_XS` (4.45 GiB) | Llama 3.1 | yes | works; answers in English, weaker Polish |
| Llama-PLLuM-8B-instruct `IQ4_XS` | Mistral `[INST]` | **no** | unusable - tool definitions are silently dropped |
| Bielik 4.5B / 7B / 11B v3.0 | ChatML without tools | **no** | unusable for the same reason |



```powershell
.\llama-server.exe -m ".\Qwen_Qwen3-8B-IQ4_XS.gguf" -c 4096
```

Check a model before downloading it. The template lives in the GGUF header.

### Qwen3 and the token counter

Qwen3 writes a reasoning block before every answer and returns it in `reasoning_content`,
separate from `content`. Measured: 385 of 405 generated tokens were reasoning. Those tokens
count towards `usage.total_tokens`, which is what the prompt shows, but the template drops
the reasoning of earlier turns - so the counter jumps up and then falls back by hundreds.
`--chat-template-kwargs "{\"enable_thinking\":false}"` turns it off (need to test).

## Run

In one PowerShell window:

```powershell
$env:ORACLE_USER = "your_user"
$env:ORACLE_PASSWORD = Read-Host "password"
python agent_sql/agent.py
```

The text after `Read-Host` is only the prompt - type the password after `password:` appears.

Example questions:

```
> which tables are there?
> what columns does PRACOWNICY have?
> how many employees are in each department?
> who earns more than 10000?
```

## Read-only safety

The model writes arbitrary SQL, so two layers in `db.py` stop any data change:

1. `check_query` lets through only a single `SELECT` or `WITH` and rejects words like `INSERT`, `DELETE`, `DROP`, `BEGIN`. Later it will also be used as an extra layer that counts how many read operations the LLM does.
2. The safest setup is still a separate database account that has only `SELECT` rights. (not done yet - a separate Oracle account with read-only permissions is the safest setup). Not added at this point, because this solution is meant to write to the database later.

## No guessing of names

The model tends to invent names like `employees` or `salary`. Asking it not to in the prompt was not enough,
so `agent.py` refuses the first `run_query` until `describe_table` has been called in the conversation.
The refusal message tells the model to look at the schema first, and it does. Using temperature = 0, don't want to randomize output.

## Limits

- An 8B model makes SQL mistakes (need to detect them and repair them in the backend).
- A wrong query that runs is worse than one that fails. Qwen3 asked for the IT department with
  `UPPER(nazwa) LIKE '%it%'` - uppercase compared against a lowercase pattern never matches - got
  no rows, and answered that the department does not exist. It found it only after being pushed.
- Asked to explain the schema, the model described tables it had not queried and invented column
  names. Anything it did not read through a tool is a guess, however confident it sounds.
- Up to 12 steps per query—a typical sequence is a refusal, `list_tables`, `describe_table`, one failed query, a corrected query, and a response. Llama tends to get stuck in a loop, while Qwen performs better and is more likely to respond that it couldn't find anything.
- The model can misread a result and state numbers that are not in it. Check the real SQL and row count in `agent.log`.
- At most 20 rows per query, long text cut to 200 characters - the context window is only 4096 tokens. Later I will increase it, my PC can handle Llama at 10 times more tokens (limit not measured). Qwen caps at 32768, its training context.
- When the conversation fills the context, older questions are dropped automatically.
