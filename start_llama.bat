@echo off
rem Starts both llama-server processes, each in its own window, from the repo root.
cd /d "%~dp0"

start "chat 8081 - Qwen3-8B" cmd /k .\modele\LIama\llama-server.exe -m .\modele\llm\Qwen_Qwen3-8B-IQ4_XS.gguf -c 8192 -ngl all --port 8081
start "embeddings 8082 - EmbeddingGemma" cmd /k .\modele\LIama\llama-server.exe -m .\modele\embeddingi\embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 0 --port 8082
