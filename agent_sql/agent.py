"""Terminal agent that answers questions about the Oracle database.

Run:
    python agent_sql/agent.py
"""
import json
import logging
import os
from pathlib import Path

import requests

from tools import BEHAVIOUR, TOOLS, TOOL_FUNCTIONS

SERVER = "http://127.0.0.1:8080"
# MAX_STEPS = 12: measured path - refused query, list_tables, describe_table, a failed
# query, the fixed query, answer.
MAX_STEPS = 12
# How many of the oldest messages one round of trimming removes.
DROP_OLDEST = 10
# Without this, a hung server stalls the agent forever.
HTTP_TIMEOUT = 60

previous_prompt = ""
# Set by ask_model from the server's "usage" field, sent by every OpenAI-compatible server.
# The number is roughly 10 tokens behind, since it excludes the role markers of the turn
# not sent yet.
last_total_tokens = "?"


# Next to this script, not in whatever folder the terminal happens to be in.
logging.basicConfig(
    filename=Path(__file__).with_name("agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("agent_sql")


# --- HTTP ---


def post(path, body):
    response = requests.post(SERVER + path, json=body, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def context_size():
    """Window size the server was started with (-c), so the counter is not hard-coded."""
    response = requests.get(SERVER + "/props", timeout=5)
    response.raise_for_status()
    return response.json()["default_generation_settings"]["n_ctx"]


def render_prompt(messages):
    """The flat text the model really receives, tool definitions included."""
    return post("/apply-template", {"messages": messages, "tools": TOOLS})["prompt"]


# --- model ---


def context_full(response):
    """True when the server refused the request because the history no longer fits.

    exceed_context_size_error is the only 400 type recoverable by shortening the history,
    so every other 400 stays an error.
    """
    try:
        return response.json()["error"]["type"] == "exceed_context_size_error"
    except (ValueError, KeyError, TypeError):
        return False


def ask_model(messages):
    """Returns the whole choice, not just the message: finish_reason tells a finished answer from a cut-off one."""
    global last_total_tokens
    body = post("/v1/chat/completions", {"messages": messages, "tools": TOOLS, "temperature": 0})
    last_total_tokens = body.get("usage", {}).get("total_tokens", "?")
    return body["choices"][0]


def trim(messages):
    """Drops the oldest messages and returns how many went. None means there is nothing left to drop.

    messages[0] is the system prompt and is never touched. The cut is pushed forward to the
    next user message, because that is the only place where no tool call is separated from
    its result.
    """
    cut = 1 + DROP_OLDEST
    while cut < len(messages) and messages[cut].get("role") != "user":
        cut += 1
    if cut >= len(messages):
        return None
    del messages[1:cut]
    return cut - 1


def hard_reset(messages):
    """Last resort before giving up: keeps the system prompt and the question being answered."""
    question = next((message for message in reversed(messages) if message.get("role") == "user"), None)
    del messages[1:]
    if question is not None:
        messages.append(question)


def make_room(messages, reason):
    """Frees space in the history. False when there is nothing left to free."""
    dropped = trim(messages)
    if dropped:
        print(f"[{reason} - dropped the {dropped} oldest messages, retrying]")
        return True
    if len(messages) > 2:
        hard_reset(messages)
        print(f"[{reason} - hard reset: only the system prompt and your question were kept]")
        return True
    return False


def ask_model_within_context(messages):
    """Asks the model, shrinking the history whenever there is no room left.

    Running out of context arrives in two shapes: HTTP 400 when the history alone is
    bigger than the window, or finish_reason "length" when the reply gets cut off mid-token.
    The step is retried after every trim, so the question is answered rather than lost.
    """
    while True:
        try:
            choice = ask_model(messages)
        except requests.HTTPError as error:
            if not context_full(error.response):
                raise
            if not make_room(messages, "context full"):
                return None
            continue
        if choice.get("finish_reason") == "length":
            if not make_room(messages, "no room left to answer"):
                return None
            continue
        return choice["message"]


# --- debug ---


def show_prompt(messages):
    """Prints the flat text the model really receives.

    The head with the tool definitions never changes, so only the new tail is printed
    after the first call.
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


# --- tools ---


def run_tool(name, args):
    log.info("TOOL: %s | args: %s", name, args)
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"ERROR: unknown tool {name}"
    try:
        return function(**args)
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


def schema_checked(messages):
    """True if describe_table was already called in the conversation still held in the history."""
    return any(
        call["function"]["name"] == "describe_table"
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    )


# --- loop ---


def handle_call(messages, call, step):
    """Runs one tool call and appends its result. True when the turn is over."""
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError as error:
        # The model writes these arguments as plain text and gets them wrong now and then;
        # a bad call goes back to it to fix, like any failed query.
        result = f"ERROR: the arguments of your {name} call were not valid JSON ({error}). Send the call again."
        print(f"  [{step}] {name} <malformed arguments>")
        print(f"      -> {result}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        return False

    if name == "answer_user":
        log.info("ANSWER: %s", args.get("reply"))
        print(args.get("reply", "(no reply)"), "\n")
        return True

    print(f"  [{step}] {name} {args}")
    # The prompt alone did not stop the model from guessing table names, so the first
    # query is refused until the schema has been looked at.
    if name == "run_query" and not schema_checked(messages):
        result = (
            "ERROR: never guess table or column names. Call list_tables, "
            "then describe_table only for the tables this query needs."
        )
    else:
        result = run_tool(name, args)

    content = json.dumps(result, ensure_ascii=False, default=str)  # Oracle dates are not JSON types on their own.
    print(f"      -> {content[:300]}")
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    return False


def answer(messages):
    for step in range(1, MAX_STEPS + 1):
        # show_prompt(messages)
        message = ask_model_within_context(messages)
        if message is None:
            notice = (
                "Unavailable: this request does not fit the server context window. "
                "Ask a narrower question."
            )
            log.info(notice)
            print(notice, "\n")
            return
        messages.append(message)
        calls = message.get("tool_calls")
        # stop when model responded with text without tool
        if not calls:
            print(message.get("content") or "(empty answer)", "\n")
            return

        if handle_call(messages, calls[0], step):
            return

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
            question = input(f"tokens {context}/{last_total_tokens} > ").strip()
            if not question:
                continue
            log.info("QUESTION: %s", question)
            before = len(messages)
            messages.append({"role": "user", "content": question})
            try:
                answer(messages)
            except (requests.RequestException, KeyError) as error:
                log.info("SERVER ERROR: %s", error)
                print(f"server error: {error}")
                print("the conversation is unaffected, try again\n")
                # Drop the unanswered question so a failed turn does not leave half a message.
                del messages[before:]
    except KeyboardInterrupt:
        print("\nbye")

if __name__ == "__main__":
    main()