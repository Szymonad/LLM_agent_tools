"""Terminal agent that answers questions about the Oracle database.

Run:
    python agent_sql/agent.py
"""
import json
import os
import urllib.request

import db

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_STEPS = 6

BEHAVIOUR = """You are a read-only assistant for an Oracle database. Always answer in English.

Pick one function for every step:
- list_tables to see which tables exist
- describe_table to see the columns of one table
- answer_user to give the final answer"""

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
}


def ask_model(messages):
    body = json.dumps({"messages": messages, "tools": TOOLS, "temperature": 0}).encode("utf-8")
    request = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]


def run_tool(name, args):
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"ERROR: unknown tool {name}"
    try:
        return function(**args)
    except Exception as error:
        # The error goes back to the model instead of stopping the program.
        return f"ERROR: {type(error).__name__}: {error}"


def answer(messages):
    for step in range(1, MAX_STEPS + 1):
        message = ask_model(messages)
        messages.append(message)

        calls = message.get("tool_calls")
        if not calls:
            print(message.get("content") or "(empty answer)", "\n")
            return

        call = calls[0]
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")

        if name == "answer_user":
            print(args.get("reply", "(no reply)"), "\n")
            return

        print(f"  [{step}] {name} {args}")
        result = run_tool(name, args)
        content = json.dumps(result, ensure_ascii=False, default=str)
        print(f"      -> {content[:300]}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    print(f"stopped: {MAX_STEPS} steps without a final answer\n")


def main():
    missing = [name for name in ("ORACLE_USER", "ORACLE_PASSWORD") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set these environment variables in this terminal first: {', '.join(missing)}")

    messages = [{"role": "system", "content": BEHAVIOUR}]
    print("SQL agent - ask about the database, Ctrl+C to quit\n")

    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            messages.append({"role": "user", "content": question})
            answer(messages)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
