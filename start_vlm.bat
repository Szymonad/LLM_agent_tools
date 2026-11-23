@echo off
rem Starts the vision server used while the PDF index is built (picture descriptions).
rem It needs the GPU, so stop the chat server (8081) first; start_llama.bat brings it back.
cd /d "%~dp0"

start "vision 8083 - Qwen3-VL-4B" cmd /k .\models\LIama\llama-server.exe -m .\models\vlm\Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj .\models\vlm\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf -c 8192 -ngl all --port 8083
