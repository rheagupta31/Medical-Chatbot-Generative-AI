#!/usr/bin/env bash
# Start the MediBot dev server.
# --reload watches only project code, not the huge .venv directory --
# watching .venv causes an endless restart loop.
cd "$(dirname "$0")"
source .venv/bin/activate
exec uvicorn app:app --reload \
  --reload-dir src \
  --reload-dir templates \
  --reload-dir static \
  --reload-include 'app.py' \
  --reload-exclude '.venv/*' \
  "$@"
