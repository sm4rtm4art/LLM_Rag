#!/bin/bash

if [ "$1" = "api" ]; then
    exec uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8008
else
    exec "$@"
fi
