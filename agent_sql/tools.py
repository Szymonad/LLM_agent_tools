"""Contract between the model and the database: system prompt, tool declarations, and their implementations."""
from agent_sql import db

ANSWER_TOOL = "answer_user"
SCHEMA_TOOL = "describe_table"
QUERY_TOOL = "run_query"

BEHAVIOUR = """You are a read-only assistant for an Oracle database. Always answer in English.

Pick one function for every step:
- list_tables to see which tables exist
- describe_table to see the columns of one table
- run_query to run one SELECT statement
- answer_user to give the final answer

Never guess table or column names. Before the first run_query, call list_tables and
describe_table for every table the query uses, unless their results are already in this conversation.

Oracle SQL rules:
- use FETCH FIRST n ROWS ONLY, never LIMIT
- no semicolon at the end
- never write text/number/date values in quotes inside sql; use :name placeholders and pass
  the values in params, e.g. sql "SELECT * FROM PRACOWNICY WHERE UPPER(IMIE) = UPPER(:imie)",
  params {"imie": "Marek"}
If run_query returns an ERROR, fix the SQL and call run_query again."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "Return the names of all tables in the database.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Return the columns and data types of one table.",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string", "description": "Exact table name returned by list_tables."}},
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run one Oracle SELECT query and return the rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single Oracle SELECT statement. Never put text/number/date values "
                            "directly in the SQL; use :name placeholders and pass the values in params."
                        ),
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Values for the :name placeholders in sql"
                        ),
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_user",
            "description": "Reply to the user with a new, original answer. Never copy or repeat the user's message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {"type": "string", "description": "The assistant's own reply, written from scratch."},
                },
                "required": ["reply"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "list_tables": db.list_tables,
    "describe_table": db.describe_table,
    "run_query": db.run_query,
}
