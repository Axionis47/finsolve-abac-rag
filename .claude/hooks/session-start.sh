#!/bin/bash
# Runs when a Claude Code cloud session starts. On a laptop it does nothing.
#
# Three jobs:
#   1. The Vertex AI credentials. The environment holds strings, not files, so the credential JSON
#      (a service-account key, or a copy of gcloud's application_default_credentials.json) rides in
#      GOOGLE_SERVICE_ACCOUNT_JSON; this writes it under artifacts/ (git-ignored) and points
#      GOOGLE_APPLICATION_CREDENTIALS at it. VERTEX_PROJECT falls back to the key's project_id.
#   2. The dependencies: the package, the tests, the web app and the Vertex client.
#   3. The offline indexes, so the tests pass before any service is reachable.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

if [ -n "${GOOGLE_SERVICE_ACCOUNT_JSON:-}" ]; then
  mkdir -p artifacts
  key="artifacts/gcp-key.json"
  umask 077
  printf '%s' "$GOOGLE_SERVICE_ACCOUNT_JSON" > "$key"
  umask 022
  echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$CLAUDE_PROJECT_DIR/$key\"" >> "$CLAUDE_ENV_FILE"
  if [ -z "${VERTEX_PROJECT:-}" ]; then
    project=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("project_id",""))' "$key")
    [ -n "$project" ] && echo "export VERTEX_PROJECT=\"$project\"" >> "$CLAUDE_ENV_FILE"
  fi
  echo "session-start: wrote the Vertex AI credentials to $key"
else
  echo "session-start: GOOGLE_SERVICE_ACCOUNT_JSON is not set; the tests and the offline commands work, Vertex does not"
fi

uv sync --frozen --group dev --group ui --group vertex

if [ ! -f artifacts/chunks.jsonl ]; then
  FINRAG_EMBEDDER=fake FINRAG_LLM=fake uv run --no-sync finrag ingest
fi
