"""Model zapisuje plik. Jedno narzędzie, jeden krok.

Uruchom najpierw serwer:
    .\\llama\\llama-server.exe -m .\\Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf -c 4096
"""
import json
import re
import urllib.request
from pathlib import Path

SERWER = "http://127.0.0.1:8081/v1/chat/completions"
PIASKOWNICA = Path(__file__).with_name("dane").resolve()

# Wyjście modelu to dane z zewnątrz — znaki sterujące rozjeżdżają terminal.
KONTROLNE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

SYSTEM = "You save files. Call write_file when asked to save something. Answer in Polish."

NARZEDZIA = [
    {
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
]


def zapisz(nazwa, tekst):
    plik = (PIASKOWNICA / nazwa).resolve()
    if not plik.is_relative_to(PIASKOWNICA):
        return f"odmowa: {nazwa} jest poza dane/"
    plik.write_text(tekst, encoding="utf-8")
    return f"zapisano {plik.name}"


def zapytaj(polecenie):
    dane = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": polecenie},
            ],
            "tools": NARZEDZIA,
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    zadanie = urllib.request.Request(
        SERWER, data=dane, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(zadanie, timeout=120) as odp:
        return json.load(odp)["choices"][0]["message"]


def main():
    PIASKOWNICA.mkdir(exist_ok=True)
    print(f"piaskownica: {PIASKOWNICA}\nCtrl+C kończy\n")
    while True:
        polecenie = input("> ").strip()
        if not polecenie:
            continue

        odpowiedz = zapytaj(polecenie)
        wywolania = odpowiedz.get("tool_calls")

        if not wywolania:
            print(KONTROLNE.sub("", odpowiedz.get("content") or ""), "\n")
            continue

        args = json.loads(wywolania[0]["function"]["arguments"])
        print(f"  write_file({args})")
        print(" ", zapisz(args["name"], args["text"]), "\n")


if __name__ == "__main__":
    main()
