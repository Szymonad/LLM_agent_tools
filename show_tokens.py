
import json
import urllib.request

text = input("> ")

body = json.dumps({"content": text, "with_pieces": True}).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8080/tokenize",
    data=body,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(request) as response:
    tokens = json.load(response)["tokens"]

for token in tokens:
    print(token["id"], repr(token["piece"]))
print("tokens:", len(tokens))