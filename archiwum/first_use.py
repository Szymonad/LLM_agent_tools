import json
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
DATA_DIR = Path(r"C:\Users\szymo\Desktop\kodzik\stacjonarny llm\dane")

# This model doesn't remember tools, to make it rementer write (Remember tools whith you use: answear_user_with_text and write_file)

BEHAVIOUR = """You are a chat assistant. Remember tools whith you use: answear_user_with_text and write_file.

Pick one function for every message:
- answer_user_with_text for conversation, questions, etc
- write_file only when the user asks to save something to a file"""

# Fixed by the API (these keys must be spelled exactly like this):
#   type          - always "function"
#   parameters    - holds a JSON Schema:
#                   type "object", properties, required (use when needed for tools)
#   type inside properties - one of: string, number, integer, boolean, array, object
#
# To choose:
#   name          - tool name, must match the if-branch below
#   description   - the MODEL reads this and picks the tool based on it
#   argument names in properties and their list in required
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer_user_with_text",
            "description": "Reply to the user in conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                    },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save text and file to its file on disk. Only when the user asks to save.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "text": {"type": "string"}
                    },
                "required": ["name", "text"],
            },
        },
    },
]

# Grows with every turn - the model has no memory of its own.
messages = [{"role": "system", "content": BEHAVIOUR}]

while True:
    prompt = input("> ").strip()
    if not prompt:
        continue

    prompt_message = {"role": "user", "content": prompt}
    messages.append(prompt_message)


    request_body = {
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0,
    }

    body_bytes = json.dumps(request_body).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request) as response:  # POST, same as requests.post
        message = json.load(response)["choices"][0]["message"]
        print(f"message: ===={message}")

    # print("\nRAW MODEL OUTPUT:")
    # print(json.dumps(message, indent=2, ensure_ascii=False))

    messages.append(message)

    if message.get("tool_calls"):
        call = message["tool_calls"][0]["function"]
        name = call["name"]
        args = json.loads(call["arguments"])

        if name == "answer_user_with_text":
            print("\n" + args["text"])
        elif name == "write_file":
            DATA_DIR.mkdir(exist_ok=True)
            file = DATA_DIR / args["name"]
            file.write_text(args["text"], encoding="utf-8")
            print("\nsaved:", file)
        else:
            print("\nunknown tool:", name)
    else:
        print("\nmodel answered without a tool:", message["content"])

    print()

# working prompt
# Create a text file named "cars.txt". Inside the file, write text that LOOKS like JSON, but does not need to be parsed as JSON. Just output plain text formatted like JSON. The text must contain a non-empty object with an array called "cars". The array must contain at least four items. Each item must include the following fields: brand, model, year, engine, horsepower, price, features. Use the brands BMW, Mercedes, Audi, and Renault. Fill every field with example values. Do NOT leave anything empty. Do NOT output an empty object. Do NOT output an empty array.The result must be plain text that visually resembles JSON
