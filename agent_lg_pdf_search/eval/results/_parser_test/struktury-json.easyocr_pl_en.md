## Struktury danych agenta PDF

agent\_Ig\_pdf\_search. Nazwy pól sprawdzone na żywych obiektach przez model\_dump() na pliku index json (161 wpisów) . W miejscach z długą treścią stoi opis zamiast tekstu.

## 1. state, czyli co trzyma checkpointer to,

Słownik 0 trzech kluczach. messages rośnie z każdą turą, context sources są nadpisywane przy każdym pytaniu.

```
'messages [ { /* HumanMessage, punkt { /* AIMessage, punkt 3 */ } 1, context" "<fragmenty PDF sklejone w jeden napis, ok . 6000 znakow> 'sources" EmbeddingGemma_technical_report. p. 10, 11" EmbeddingGemma_technical_report.pdf p. 9 , 10" pdf
```

## 2. HumanMessage

id jest null, dopóki wiadomość nie przejdzie przez graf; wtedy LangGraph nadaje jej uuid.

```
content" "ile parametrow ma gemma "additional_kwargs "response_metadata {} , "type' "human" "name null, 'id" null
```

## 3. AlMessage, czyli odpowiedź modelu

Pola tool\_calls invalid tool\_calls są puste, bo nasz agent nie daje modelowi narzędzi.

```
content" 'EmbeddingGemma has 308M parameters [EmbeddingGemma technical_report. P. 11] "additional_kwargs { "refusal" null } "response_metadata "token_usage "completion tokens 291 , "prompt tokens 2015 , "total tokens 2306 , "prompt tokens details { cached tokens 1872 } } , 'model_provider "openai model_name IQwen_Qwen3-8B-IQ4_XS . system_fingerprint "b10842-473599738 "id" chatcmpl "finish_reason stop "logprobs null "type ai" "name null, "id" run - 01a0d437 "tool calls [], 'invalid tool calls" : [], "usage_metadata 'input tokens 2015 , "output_tokens 291, "total tokens 2306 'input token_details { "cache_read" "output_token_details pdf 10, gguf"
```

---

## 4. czyli jedna porcja ze stream(stream mode="updates") step,

Na turę przychodzą dwa kroki; po jednym na węzeł. Klucz to nazwa węzła; wartość to jego zwrot.

```
{ "retrieve { "context "<fragmenty sources ["plik. 10, 11" "] } } 1/ potem { answer { 'messages [ { /* AIMessage punktu 3 */ } ] } } pdf
```

## 5. Document; czyli jeden fragment z search()

id jest nadawane przy dodaniu do indeksu; więc fragmenty wczytane z index json mają uuid.

```
"id" '217b9ac7-dal9-477a-9694-65bac8698326" 'metadata { source 'EmbeddingGemma_technical_report. 'pages [2, 3] } page content <tekst fragmentu PDF , do 400 tokenow> "type "Document pdf"
```

## 6. index json na dysku

```
"id" "vector [-0.058, -0.031, 0.009 , razem 768 liczb */ 0.074], "text <tekst fragmentu PDF > "metadata source "CPIK W06 'pages [1] } }, kolejne 160 wpisow {} Pdf"
```

## 7 . state["context"] w środku

To jeden napis, a nie lista. Numery są tym, co cytuje model, zanim kod podmieni je na nazwy plików.

```
[1] <tekst fragmentu 1> [2] <tekst fragmentu 2> az do [5]
```

## 8. Prompt wysłany do modelu

Budowany w answer przy każdej turze nigdzie nietrzymany. Klasy wiadomości zamieniają się na role z API OpenAI.

```
{ "role 'system "content <BEHAVIOUR , czyli zasady , 441 znakow> } , { "role 'system content <state[ 'context' ], czyli fragmenty> } , { role user content 'ile parametrow Ma gemma { "role "assistant" content <poprzednia odpowiedz, jesli rozmowa trwa> } { "role user "content '<najnowsze pytanie>
```

---

## 9. Jak sięgać po te dane w kodzie

Słownik po kluczu w nawiasach; lista po indeksie; obiekt po kropce.

```
state[ "messages" ] list[BaseMessage] state[ "messages"][-1] AIMessage albo HumanMessage state[ "messages"][-1].content str state[ "messages"][-1].usage_metadata # dict state[ sources"][0] str: "plik.pdf 10, 11" step[ "answer" ]["messages"][-1].content str chunk.metadata[ "pages"] list[int]
```