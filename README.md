# Local LLM with tools

## Server start

powershell
.\llama\llama-server.exe -m .\Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf -c 4096

## Database credentials (db_agent.py)

db_agent.py reads the Oracle login from environment variables.
Set them in the same PowerShell window that runs the script:

```powershell
$env:ORACLE_USER = "your_user"
$env:ORACLE_PASSWORD = Read-Host "password"
python db_agent.py
```

- The text after `Read-Host` is only the prompt - type the password after `password:` appears.
- Variables live only in that window and disappear when it closes.
- Never put the password in a file.


