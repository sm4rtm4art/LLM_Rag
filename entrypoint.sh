#!/bin/bash
set -e

# If the first argument is "api", execute uvicorn
if [ "$1" = "api" ]; then
    exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000
else
    # Otherwise, execute the given command
    exec "$@"
fi
