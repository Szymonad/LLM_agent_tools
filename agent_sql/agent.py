"""Terminal agent that answers questions about the Oracle database.

Run:
    python agent_sql/agent.py
"""
import json
import logging
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_sql.config import DROP_OLDEST, MAX_STEPS
from agent_sql.llama_client import LlamaClient, context_full
from agent_sql.tools import ANSWER_TOOL, BEHAVIOUR, QUERY_TOOL, SCHEMA_TOOL, TOOL_FUNCTIONS, TOOLS


# Next to this script, not in whatever folder the terminal happens to be in.
logging.basicConfig(
    filename=Path(__file__).with_name("agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("agent_sql")


class Agent:
    def __init__(self, client):
        self.client = client
        self.messages = [{"role": "system", "content": BEHAVIOUR}]
        self.previous_prompt = ""

    # --- conversation ---

    def ask(self, question):
        log.info("QUESTION: %s", question)
        before = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        try:
            self.run_turn()
        except (requests.RequestException, KeyError) as error:
            log.info("SERVER ERROR: %s", error)
            print(f"server error: {error}")
            print("the conversation is unaffected, try again\n")
            # Drop the unanswered question so a failed turn does not leave half a message.
            del self.messages[before:]

    def run_turn(self):
        for step in range(1, MAX_STEPS + 1):
            # self.show_prompt()

            message = self.call_model()
            print("======================================================================================")
            self.print_message(self.messages)
            print("======================================================================================")
            if message is None:
                notice = (
                    "Unavailable: this request does not fit the server context window. "
                    "Ask a narrower question."
                )
                log.info(notice)
                print(notice, "\n")
                return
            self.messages.append(message)
            calls = message.get("tool_calls")

            # stop when model responded with text without tool
            if not calls:
                # Qwen keeps its thinking in reasoning_content, so content can be empty
                # while the model did produce text.
                text = message.get("content")
                log.info("NO TOOL CALL: %s", message)
                print(text or f"(empty answer) {message}", "\n")
                return

            if self.handle_call(calls[0], step):
                return

        log.info("STOPPED after %d steps", MAX_STEPS)
        print(f"stopped: {MAX_STEPS} steps without a final answer\n")

    def call_model(self):
        """Asks the model, shrinking the history whenever there is no room left.

        Running out of context arrives in two shapes: HTTP 400 when the history alone is
        bigger than the window, or finish_reason "length" when the reply gets cut off mid-token.
        The step is retried after every trim, so the question is answered rather than lost.
        """
        while True:
            try:
                choice = self.client.chat(self.messages)
            except requests.HTTPError as error:
                if not context_full(error.response):
                    raise
                if not self.make_room("context full"):
                    return None
                continue
            if choice.get("finish_reason") == "length":
                if not self.make_room("no room left to answer"):
                    return None
                continue
            return choice["message"]

    # --- context ---

    def make_room(self, reason):
        """Frees space in the history. False when there is nothing left to free."""
        dropped = self.trim()
        if dropped:
            print(f"[{reason} - dropped the {dropped} oldest messages, retrying]")
            return True
        if len(self.messages) > 2:
            self.hard_reset()
            print(f"[{reason} - hard reset: only the system prompt and your question were kept]")
            return True
        return False

    def trim(self):
        """Drops the oldest messages and returns how many went. None means there is nothing left to drop.

        messages[0] is the system prompt and is never touched. The cut is pushed forward to the
        next user message, because that is the only place where no tool call is separated from
        its result.
        """
        cut = 1 + DROP_OLDEST
        while cut < len(self.messages) and self.messages[cut].get("role") != "user":
            cut += 1
        if cut >= len(self.messages):
            return None
        del self.messages[1:cut]
        return cut - 1

    def hard_reset(self):
        """Last resort before giving up: keeps the system prompt and the question being answered."""
        question = next((message for message in reversed(self.messages) if message.get("role") == "user"), None)
        del self.messages[1:]
        if question is not None:
            self.messages.append(question)

    # --- tools ---

    def handle_call(self, call, step):
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
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            return False

        if name == ANSWER_TOOL:
            log.info("ANSWER: %s", args.get("reply"))
            print(args.get("reply", "(no reply)"), "\n")
            return True

        print(f"  [{step}] {name} {args}")
        refusal = self.check_policy(name, args)
        if refusal is not None:
            result = refusal
        else:
            result = run_tool(name, args)

        content = json.dumps(result, ensure_ascii=False, default=str)  # Oracle dates are not JSON types on their own.
        print(f"      -> {content[:300]}")
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return False

    def check_policy(self, name, args):
        """Returns an error string when a call must be refused, None when it may run."""
        # The prompt alone did not stop the model from guessing table names, so the first
        # query is refused until the schema has been looked at.
        if name == QUERY_TOOL and not self.schema_checked():
            return (
                "ERROR: never guess table or column names. Call list_tables, "
                "then describe_table only for the tables this query needs."
            )
        return None

    def schema_checked(self):
        """True if describe_table was already called in the conversation still held in the history."""
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call["function"]["name"] == SCHEMA_TOOL:
                    return True
        return False

    # --- debug ---

    def show_prompt(self):
        """Prints the flat text the model really receives.

        The head with the tool definitions never changes, so only the new tail is printed
        after the first call.
        """
        prompt = self.client.render_prompt(self.messages)
        if self.previous_prompt and prompt.startswith(self.previous_prompt):
            print("----- PROMPT: new part -----")
            print(prompt[len(self.previous_prompt):])
        else:
            print("----- PROMPT: full -----")
            print(prompt)
        print("----- END -----")
        self.previous_prompt = prompt

    def print_message(self, message):
        """Prints a model message as indented JSON, with tool arguments unpacked for reading."""
        if isinstance(message, list):
            for item in message:
                self.print_message(item)
            return
        if message is None:
            print("None")
            return
        shown = json.loads(json.dumps(message))
        for call in shown.get("tool_calls") or []:
            try:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            except (ValueError, TypeError):
                pass
        print(json.dumps(shown, indent=4, ensure_ascii=False))


def run_tool(name, args):
    log.info("TOOL: %s | args: %s", name, args)
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"ERROR: unknown tool {name}"
    try:
        return function(**args)
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


def main():
    missing = [name for name in ("ORACLE_USER", "ORACLE_PASSWORD") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set these environment variables in this terminal first: {', '.join(missing)}")

    client = LlamaClient(tools=TOOLS)
    context = client.context_window()
    agent = Agent(client)
    print("SQL agent - ask about the database, Ctrl+C to quit\n")

    try:
        while True:
            question = input(f"tokens {context}/{client.last_total_tokens} > ").strip()
            if question:
                agent.ask(question)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
