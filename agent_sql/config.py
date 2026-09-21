"""Settings worth changing between runs. Secrets stay in environment variables."""

# --- llama-server ---
# 8080 is taken at every Windows start by MiniTool ShadowMaker's MTAgentService.
SERVER = "http://127.0.0.1:8081"
# Without this, a hung server stalls the agent forever.
HTTP_TIMEOUT = 60

# --- conversation ---
# MAX_STEPS = 12: measured path - refused query, list_tables, describe_table, a failed
# query, the fixed query, answer.
MAX_STEPS = 12
# How many of the oldest messages one round of trimming removes.
DROP_OLDEST = 10

# --- database ---
ORACLE_DSN = "localhost:1521/xepdb1"
# One query result has to leave room in the 4096 token window for the rest of the conversation.
MAX_ROWS = 20
