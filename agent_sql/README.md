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
- `db.py` - database connection, tools and the read-only guard
- `agent.log` - created on first run: questions, tool calls and every SQL sent to the database

## Requirements

- llama-server running on `http://127.0.0.1:8080`
- `pip install oracledb`
- Oracle XE with service `xepdb1` on port 1521

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

1. `check_query` lets through only a single `SELECT` or `WITH` and rejects words like `INSERT`, `DELETE`, `DROP`, `BEGIN`.
2. Every connection starts with `SET TRANSACTION READ ONLY` and nothing is ever committed.

The safest setup is still a separate database account that has only `SELECT` rights.

## No guessing of names

The model tends to invent names like `employees` or `salary`. Asking it not to in the prompt was not enough,
so `agent.py` refuses the first `run_query` until `describe_table` has been called in the conversation.
The refusal message tells the model to look at the schema first, and it does.

## Limits

- An 8B model makes SQL mistakes. Errors go back to the model, which usually fixes the query on the next step.
- Up to 12 steps per question - a typical path is refusal, `list_tables`, `describe_table`, one failed query, the fixed query, answer.
- The model can misread a result and state numbers that are not in it. Check the real SQL and row count in `agent.log`.
- At most 20 rows per query, long text cut to 200 characters - the context window is only 4096 tokens.
- When the conversation fills the context, older questions are dropped automatically.
