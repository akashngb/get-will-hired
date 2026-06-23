#!/usr/bin/env bash
# Launch the FastAPI inference server.
set -euo pipefail

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8080}"

exec uvicorn src.serving.api:app --host "$HOST" --port "$PORT"
