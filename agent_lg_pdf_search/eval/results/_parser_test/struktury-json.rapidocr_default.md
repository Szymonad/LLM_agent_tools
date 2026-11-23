## Struktury danych agenta PDF

agent\_lg\_pdf\_search. Nazwy pól sprawdzone na żywych obiektach przez model\_dump() i na pliku index.json (161 wpisów). W miejscach z długą treścią stoi opis zamiast tekstu.

## 1. state, czyli to, co trzyma checkpointer

Słownik o trzech kluczach. messages rośnie z każdą turą, context i sources są nadpisywane przy każdym pytaniu.

```
{ "messages": [ { /* HumanMessage, punkt 2 */ }, { /* AIMessage, punkt 3 */ } ], "context": "<fragmenty PDF sklejone w jeden napis, ok. 6000 znakow>", "sources": [ "EmbeddingGemma_technical_report.pdf p. 10, 11", "EmbeddingGemma_technical_report.pdf p. 9, 10" ] }
```

## 2. HumanMessage

id jest null, dopóki wiadomość nie przejdzie przez graf; wtedy LangGraph nadaje jej uuid.

```
{ "additional_kwargs": {}, "response_metadata": {}, "type": "human", "name": null, "id": null }
```

## 3. AlMessage, czyli odpowiedź modelu

Pola tool\_calls i invalid\_tool\_calls są puste, bo nasz agent nie daje modelowi narzędzi.

```
{ "content": "EmbeddingGemma has 308M parameters [EmbeddingGemma_technical_report.pdf p. 10, 11].", "additional_kwargs":{ "refusal": null }, "response_metadata": { "token_usage": { "completion_tokens": 291, "prompt_tokens": 2015, "total_tokens": 2306, "prompt_tokens_details": { "cached_tokens": 1872 } }, "model_provider": "openai", "model_name": ".\\Qwen_Qwen3-8B-IQ4_XS.gguf", "system_fingerprint": "b10842-473599738", "id": "chatcmpl-...", "finish_reason": "stop", "logprobs": null }, "type": "ai", "name": null, "id": "lc_run--01a0d437-...", "tool_calls": [], "invalid_tool_calls": [], "usage_metadata": { "input_tokens": 2015, "output_tokens": 291, "total_tokens": 2306, "input_token_details": { "cache_read": 1872 }, "output_token_details": {} } }
```

---

## A.srdr z rdd zss )

Na turę przychodzą dwa kroki, po jednym na węzeł. Klucz to nazwa węzła, wartość to jego zwrot.

```
// najpierw { "retrieve": { "context": "<fragmenty...>", "sources": ["plik.pdf p. 10, 11", "..."] } } // potem { "answer": { "messages": [ { /* AIMessage z punktu 3 */ } ] } }
```

## 5. Document, czyli jeden fragment z search()

id jest nadawane przy dodaniu do indeksu, więc fragmenty wczytane z index.json mają uuid.

```
{ 899-99--9-992.:6.. "metadata": { "source": "EmbeddingGemma_technical_report.pdf", "pages": [2, 3] }, "page_content": "<tekst fragmentu PDF, do 400 tokenow>", "type": "Document" }
```

## 6. index.json na dysku

Słownik: klucz to uuid wpisu, wartość to fragment razem z wektorem. U nas 161 wpisów, plik ma ok. 3,7 MB.

```
{ "f12e57ec-1e6...": { "id": "f12e57ec-1e6...", "vector": [-0.058, -0.031, 0.009, /* ... razem 768 liczb ... */ 0.074], "text": "<tekst fragmentu PDF>", "metadata": { "source": "CPIK W06.pdf", "pages": [1] } }, "...kolejne 160 wpisow...": {} }
```

## 7. state["context"] w środku

To      i          s  o..

```
[1] <tekst fragmentu 1> [2] <tekst fragmentu 2> ... az do [5]
```

## 8. Prompt wysłany do modelu

Budowany w answer przy każdej turze i nigdzie nietrzymany. Klasy wiadomości zamieniają się na role z APl OpenAI.

```
] { "role": "system", "content": "<BEHAVIOUR, czyli zasady, 441 znakow>" }, { "role": "system", "content": "<state['context'], czyli fragmenty>" }, { "role": "user", "content": "ile parametrow ma gemma" }, {   d : :e } { "role": "user", "content": "<najnowsze pytanie>" } ]
```

---

## 9. Jak sięgać po te dane w kodzie

Słownik po kluczu w nawiasach, lista po indeksie, obiekt po kropce.

```
state["messages"] # list[BaseMessage] state["messages"][-1] # AIMessage albo HumanMessage stat[a[]tent # str state["messages"][-1].usage_metadata #dict state["sources"][0] # str: "plik.pdf p. 10, 11" step["answer"]["messages"][-1].content # str chunk.metadata["pages"] # list[int]
```