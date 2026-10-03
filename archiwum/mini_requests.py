"""To samo co mini.py, ale na bibliotece requests.

Wymaga instalacji:
    pip install requests
"""
import requests
from pathlib import Path

URL = "http://127.0.0.1:8081/v1/chat/completions"

NARZEDZIE = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Save text to a file.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
            "required": ["name", "text"],
        },
    },
}

polecenie = input("> ")

odp = requests.post(
    URL,
    json={
        "messages": [{"role": "user", "content": polecenie}],
        "tools": [NARZEDZIE],
        "temperature": 0,
    },
    timeout=120,
)
odp.raise_for_status()
wiadomosc = odp.json()["choices"][0]["message"]

print("\nCO WROCILO OD MODELU:")
print(odp.text)

if wiadomosc.get("tool_calls"):
    import json

    argumenty = json.loads(wiadomosc["tool_calls"][0]["function"]["arguments"])
    Path(argumenty["name"]).write_text(argumenty["text"], encoding="utf-8")
    print("\nzapisano:", argumenty["name"])
else:
    print("\nmodel nie chcial narzedzia, odpowiedzial:", wiadomosc["content"])
