"""Terminal agent that answers questions about the Oracle database.

Run:
    python agent_sql/agent.py
"""
import json
import logging
import os
import urllib.request
from pathlib import Path

import db

SERVER = "http://127.0.0.1:8080"
LLM_URL = SERVER + "/v1/chat/completions"
# Measured path: refused query, list_tables, describe_table, a failed query, the fixed query, answer.
MAX_STEPS = 12

# Next to this script, not in whatever folder the terminal happens to be in.
logging.basicConfig(
    filename=Path(__file__).with_name("agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("agent_sql")

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
- compare text with UPPER(column) = UPPER('value')
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
                "properties": {"sql": {"type": "string", "description": "A single Oracle SELECT statement."}},
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


def post(path, body):
    request = urllib.request.Request(
        SERVER + path, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def ask_model(messages):
    return post("/v1/chat/completions", {"messages": messages, "tools": TOOLS, "temperature": 0})["choices"][0]["message"]


def render_prompt(messages):
    """The flat text the model really receives, tool definitions included."""
    return post("/apply-template", {"messages": messages, "tools": TOOLS})["prompt"]


def context_size():
    """Window size the server was started with (-c), so the counter is not hard-coded."""
    with urllib.request.urlopen(SERVER + "/props", timeout=5) as response:
        return json.load(response)["default_generation_settings"]["n_ctx"]


def used_tokens(messages):
    """Tokens the next request would take, counted on a history the template accepts.

    Two shapes are rejected by the Llama 3.1 template and both occur here:
    tools without any user message (before the first question), and a trailing
    assistant tool call with no result (after a turn ended by answer_user).
    """
    messages = list(messages)
    if not any(message.get("role") == "user" for message in messages):
        messages.append({"role": "user", "content": ""})

    last = messages[-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        messages.append({"role": "tool", "tool_call_id": last["tool_calls"][0]["id"], "content": ""})

    try:
        return len(post("/tokenize", {"content": render_prompt(messages)})["tokens"])
    except urllib.error.URLError:
        # A counter is not worth crashing the program for.
        return "?"


previous_prompt = ""


def show_prompt(messages):
    """Prints the flat text the model really receives, with the role markers.

    The head with the tool definitions never changes, so only the new tail is printed
    after the first call - that tail is exactly what one turn adds to the prompt.
    """
    global previous_prompt

    prompt = render_prompt(messages)
    if previous_prompt and prompt.startswith(previous_prompt):
        print("----- PROMPT: new part -----")
        print(prompt[len(previous_prompt):])
    else:
        print("----- PROMPT: full -----")
        print(prompt)
    print("----- END -----")
    previous_prompt = prompt


def run_tool(name, args):
    log.info("TOOL: %s | args: %s", name, args)
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"ERROR: unknown tool {name}"
    try:
        return function(**args)
    except Exception as error:
        # The error goes back to the model instead of stopping the program.
        return f"ERROR: {type(error).__name__}: {error}"


def schema_checked(messages):
    """True if describe_table was already called in the conversation still held in the history."""
    return any(
        call["function"]["name"] == "describe_table"
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    )


def answer(messages):
    for step in range(1, MAX_STEPS + 1):
        # show_prompt(messages)
        message = ask_model(messages)
        messages.append(message)
        # print(f"message ===== {message}")
        calls = message.get("tool_calls")
        if not calls:
            print(message.get("content") or "(empty answer)", "\n")
            return

        call = calls[0]
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")

        if name == "answer_user":
            log.info("ANSWER: %s", args.get("reply"))
            print(args.get("reply", "(no reply)"), "\n")
            return

        print(f"  [{step}] {name} {args}")
        # The prompt alone did not stop the model from guessing names like "employees",
        # so the first query is refused until the schema has been looked at.
        if name == "run_query" and not schema_checked(messages):
            result = (
                "ERROR: never guess table or column names. Call list_tables, "
                "then describe_table only for the tables this query needs."
            )
        else:
            result = run_tool(name, args)

        # default=str: Oracle dates are not JSON types on their own.
        content = json.dumps(result, ensure_ascii=False, default=str)
        print(f"      -> {content[:300]}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    log.info("STOPPED after %d steps", MAX_STEPS)
    print(f"stopped: {MAX_STEPS} steps without a final answer\n")


def main():
    missing = [name for name in ("ORACLE_USER", "ORACLE_PASSWORD") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set these environment variables in this terminal first: {', '.join(missing)}")

    context = context_size()
    messages = [{"role": "system", "content": BEHAVIOUR}]
    print("SQL agent - ask about the database, Ctrl+C to quit\n")

    try:
        while True:
            question = input(f"tokens {context}/{used_tokens(messages)} > ").strip()
            if not question:
                continue
            log.info("QUESTION: %s", question)
            messages.append({"role": "user", "content": question})
            answer(messages)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
