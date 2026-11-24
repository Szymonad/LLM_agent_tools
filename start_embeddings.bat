@echo off
rem Starts only the embedding server, on the CPU, for building the index with start_vlm.bat.
cd /d "%~dp0"

start "embeddings 8082 - EmbeddingGemma" cmd /k .\models\LIama\llama-server.exe -m .\models\embeddings\embeddinggemma-300M-Q8_0.gguf --embeddings -ngl 0 --port 8082
