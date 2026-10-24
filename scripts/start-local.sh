#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.local"
LOCAL_EMBEDDINGS="false"

usage() {
  cat <<'EOF'
Uso: ./scripts/start-local.sh [--local-embeddings]

  Sin opciones          Inicia la app en http://127.0.0.1:8000
  --local-embeddings    Inicia también Ollama para embeddings locales
EOF
}

while (($#)); do
  case "$1" in
    --local-embeddings) LOCAL_EMBEDDINGS="true" ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Opción desconocida: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  printf 'Instala Docker Engine o Docker Desktop con Docker Compose v2.\n' >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

IFS=$'\t' read -r PROJECT_DIR WEB_PORT < <(python3 - "$ENV_FILE" "$ROOT_DIR" "$LOCAL_EMBEDDINGS" <<'PY'
from __future__ import annotations

import base64
import os
import re
import secrets
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
local_embeddings = sys.argv[3] == "true"
lines = env_path.read_text(encoding="utf-8").splitlines()
values: dict[str, str] = {}
for line in lines:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
    if match:
        values[match.group(1)] = match.group(2)

project_dir = Path(values.get("DISCOURSELENS_PROJECT_DIR", "./data/local-project"))
if not project_dir.is_absolute():
    project_dir = root / project_dir
project_dir = project_dir.resolve()
session_secret = values.get("DISCOURSELENS_WEB_SESSION_SECRET", "").strip()
credential_key = values.get("DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", "").strip()
if len(session_secret.encode("utf-8")) < 32:
    session_secret = secrets.token_urlsafe(48)
if not credential_key:
    credential_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")

updates = {
    "COMPOSE_PROJECT_NAME": "discourselens-local",
    "DISCOURSELENS_PROJECT_DIR": str(project_dir),
    "DISCOURSELENS_WEB_SESSION_SECRET": session_secret,
    "DISCOURSELENS_WEB_ALLOW_REGISTRATION": (
        values.get("DISCOURSELENS_WEB_ALLOW_REGISTRATION", "true") or "true"
    ),
    "DISCOURSELENS_WEB_COOKIE_SECURE": "false",
    "DISCOURSELENS_WEB_PUBLIC_ORIGIN": "",
    "DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY": credential_key,
    "DISCOURSELENS_WEB_PORT": values.get("DISCOURSELENS_WEB_PORT", "8000") or "8000",
    "DISCOURSELENS_UID": str(os.getuid()),
    "DISCOURSELENS_GID": str(os.getgid()),
    "DISCOURSELENS_PROVIDER_ENV_FILE": values.get("DISCOURSELENS_PROVIDER_ENV_FILE", "./provider.env") or "./provider.env",
}
if local_embeddings:
    updates["DISCOURSELENS_PROVIDER_ENV_FILE"] = "./provider.env"

output: list[str] = []
seen: set[str] = set()
for line in lines:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
    if match and match.group(1) in updates:
        key = match.group(1)
        output.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        output.append(line)
for key, value in updates.items():
    if key not in seen:
        output.append(f"{key}={value}")
env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
print(f"{project_dir}\t{updates['DISCOURSELENS_WEB_PORT']}")
PY
)

mkdir -p "$PROJECT_DIR"
chmod 700 "$PROJECT_DIR"

COMPOSE=(docker compose --project-name discourselens-local --env-file "$ENV_FILE" -f "$ROOT_DIR/compose.web.yaml")
if [[ "$LOCAL_EMBEDDINGS" == "true" ]]; then
  if ! docker network inspect discourselens-local-embeddings >/dev/null 2>&1; then
    docker network create discourselens-local-embeddings >/dev/null
  fi
  if ! docker volume inspect discourselens-ollama-models >/dev/null 2>&1; then
    docker volume create discourselens-ollama-models >/dev/null
  fi
  if [[ ! -f "$ROOT_DIR/provider.env" ]]; then
    printf 'DL_OLLAMA_API_KEY=ollama\n' > "$ROOT_DIR/provider.env"
    chmod 600 "$ROOT_DIR/provider.env"
  fi
  docker compose --project-name discourselens-ollama -f "$ROOT_DIR/compose.ollama.yaml" up -d
  COMPOSE+=(-f "$ROOT_DIR/compose.local-embeddings.yaml")
fi

"${COMPOSE[@]}" up -d --build web

if [[ "$LOCAL_EMBEDDINGS" == "true" ]]; then
  OLLAMA=(docker compose --project-name discourselens-ollama -f "$ROOT_DIR/compose.ollama.yaml")
  printf 'Descargando el modelo de embeddings local si hace falta…\n'
  "${OLLAMA[@]}" exec -T ollama ollama pull embeddinggemma:300m-qat-q4_0
fi

printf 'Abre la app en http://127.0.0.1:%s\n' "$WEB_PORT"

STOP=(docker compose --project-name discourselens-local --env-file "$ENV_FILE" -f "$ROOT_DIR/compose.web.yaml")
if [[ "$LOCAL_EMBEDDINGS" == "true" ]]; then
  STOP+=(-f "$ROOT_DIR/compose.local-embeddings.yaml")
fi
STOP+=(down)
printf 'Para detenerla: '
printf '%q ' "${STOP[@]}"
printf '\n'
