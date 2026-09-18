#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PORT="${1:-8765}"
cd "$APP_ROOT"
export NEBULAX_LOCAL_PATHS=1

APP_PYTHON="${NEBULAX_PYTHON:-python}"
if ! "$APP_PYTHON" -c 'import fastapi, openpyxl, uvicorn' >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
    CANDIDATE="$CONDA_ROOT/envs/nebulax/bin/python"
    if [[ -x "$CANDIDATE" ]] && "$CANDIDATE" -c 'import fastapi, openpyxl, uvicorn' >/dev/null 2>&1; then
      APP_PYTHON="$CANDIDATE"
    fi
  fi
fi
if ! "$APP_PYTHON" -c 'import fastapi, openpyxl, uvicorn' >/dev/null 2>&1; then
  printf '%s\n' 'NEBULA X needs the Python dependencies in requirements.lock.txt (including openpyxl).' >&2
  printf '%s\n' 'Activate that environment, or set NEBULAX_PYTHON to its python executable.' >&2
  exit 1
fi
exec "$APP_PYTHON" -m uvicorn nebulax.api.main:app --host 127.0.0.1 --port "$PORT"
