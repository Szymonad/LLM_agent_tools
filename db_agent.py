"""Read-only database agent: lists tables and finds a person by name.

Requires:
    pip install oracledb

Credentials come from environment variables, never from this file:
    $env:ORACLE_USER = "..."
    $env:ORACLE_PASSWORD = "..."
"""
import json
import logging
import os
import urllib.request
from pathlib import Path

import oracledb

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
ORACLE_DSN = "localhost:1521/xepdb1"
MAX_STEPS = 5
MAX_ROWS = 10

PERSON_TABLE = "PRACOWNICY"
# Assumed column names - check with:
#   SELECT column_name FROM user_tab_columns WHERE table_name = 'PRACOWNICY'
FIRST_NAME_COLUMN = "IMIE"
LAST_NAME_COLUMN = "NAZWISKO"

# Next to this script, not in whatever folder the terminal happens to be in.
LOG_FILE = Path(__file__).with_name("agent.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("db_agent")

BEHAVIOUR = """You are a read-only database assistant. Always answer in English.

Pick one function for every step:
- list_tables when the user asks what data or tables are available
- find_person when the user asks about a person by first and last name
- answer_user_with_text to give the final answer to the user"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "Return the names of all tables available in the database.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_person",
            "description": "Find an employee by first name and last name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                },
                "required": ["first_name", "last_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_user_with_text",
            # "text" was ambiguous - the model copied the user's message into it.
            "description": "Reply to the user with a new, original answer. Never copy or repeat the user's message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {
                        "type": "string",
                        "description": "The assistant's own reply, written from scratch.",
                    },
                },
                "required": ["reply"],
            },
        },
    },
]


def connect():
    return oracledb.connect(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=ORACLE_DSN,
    )


def run_sql(cursor, sql, **params):
    # Every query goes through here, so the log shows exactly what reached the database.
    # Result rows are not logged - they hold personal data.
    log.info("SQL: %s | params: %s", sql, params)
    cursor.execute(sql, **params)


def list_tables():
    with connect() as connection:
        with connection.cursor() as cursor:
            run_sql(cursor, "SELECT table_name FROM user_tables ORDER BY table_name")
            return [row[0] for row in cursor.fetchall()]


def find_person(first_name, last_name):
    # Table and column names are constants from this file, never text from the model.
    # Names from the model go only through bind variables (:first_name, :last_name).
    sql = (
        f"SELECT * FROM {PERSON_TABLE} "
        f"WHERE UPPER({FIRST_NAME_COLUMN}) = UPPER(:first_name) "
        f"AND UPPER({LAST_NAME_COLUMN}) = UPPER(:last_name) "
        f"FETCH FIRST {MAX_ROWS} ROWS ONLY"
    )
    with connect() as connection:
        with connection.cursor() as cursor:
            run_sql(cursor, sql, first_name=first_name, last_name=last_name)
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    # An explicit sentence instead of an empty list - a small model may invent a person from [].
    if not rows:
        return f"No employee named {first_name} {last_name} exists in the database."
    return rows


def run_tool(name, args):
    log.info("TOOL: %s | args: %s", name, args)
    if name == "list_tables":
        return list_tables()
    if name == "find_person":
        return find_person(args["first_name"], args["last_name"])
    return f"unknown tool: {name}"


def ask_model(messages):
    body = json.dumps(
        {"messages": messages, "tools": TOOLS, "temperature": 0}
    ).encode("utf-8")
    request = urllib.request.Request(
        LLM_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]


missing = [name for name in ("ORACLE_USER", "ORACLE_PASSWORD") if not os.environ.get(name)]
if missing:
    raise SystemExit(f"set these environment variables in this terminal first: {', '.join(missing)}")

messages = [{"role": "system", "content": BEHAVIOUR}]

while True:
    prompt = input("> ").strip()
    if not prompt:
        continue
    messages.append({"role": "user", "content": prompt})

    for step in range(MAX_STEPS):
        message = ask_model(messages)
        messages.append(message)

        if not message.get("tool_calls"):
            print(message.get("content"), "\n")
            break

        call = message["tool_calls"][0]
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")
        print(f"  [{step + 1}] {name}")

        if name == "answer_user_with_text":
            print(args["reply"], "\n")
            break

        try:
            result = run_tool(name, args)
        except Exception as error:
            # Type name included - some exceptions have an empty message.
            result = f"ERROR: {type(error).__name__}: {error}"

        # default=str: Oracle dates and numbers are not JSON types on their own.
        content = json.dumps(result, ensure_ascii=False, default=str)
        print(f"      -> {content[:200]}")
        # The model sees the result only because it is added to the history.
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    else:
        print(f"stopped: {MAX_STEPS} steps without a final answer\n")




# output:
# > is there marek nowak
#   [1] find_person
#       -> [{"ID": 2, "IMIE": "Marek", "NAZWISKO": "Nowak", "EMAIL": "m.nowak@firma.pl", "ZATRUDNIONY": "2016-06-15 00:00:00", "PENSJA": 18500.0, "DZIAL_ID": 10, "STANOWISKO_ID": 2, "KIEROWNIK_ID": 1}]
#   [2] answer_user_with_text
# There is an employee named MAREK NOWAK in the database.

# > do you have more information abut him?
#   [1] answer_user_with_text
# Marek Nowak has been working at the company since 2016, his email is m.nowak@firma.pl, and his salary is 18500.0 PLN.

# >
