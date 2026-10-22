#!/usr/bin/env bash
set -euo pipefail
MANDOL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MANDOL_ROOT"
export UV_CACHE_DIR="$MANDOL_ROOT/.cache/uv" UV_PYTHON_INSTALL_DIR="$MANDOL_ROOT/.toolchains"
export npm_config_cache="$MANDOL_ROOT/.cache/npm"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Explicit process environment wins over the local file, including credentials.
declare -A MANDOL_CALLER_ENV=()
while IFS= read -r MANDOL_ENV_NAME; do
  case "$MANDOL_ENV_NAME" in CHINASOFT_*|DEEPSEEK_API_KEY|MANDOL_*|HF_TOKEN|HF_HUB_DISABLE_XET) MANDOL_CALLER_ENV["$MANDOL_ENV_NAME"]="${!MANDOL_ENV_NAME}";; esac
done < <(compgen -e)
if [[ -f .env ]]; then set -a; source .env; set +a; fi
for MANDOL_ENV_NAME in "${!MANDOL_CALLER_ENV[@]}"; do export "$MANDOL_ENV_NAME=${MANDOL_CALLER_ENV[$MANDOL_ENV_NAME]}"; done
unset MANDOL_CALLER_ENV MANDOL_ENV_NAME
export CHINASOFT_DATA_DIR="${CHINASOFT_DATA_DIR:-$MANDOL_ROOT/data/native}"
export CHINASOFT_HOST="${CHINASOFT_HOST:-127.0.0.1}" CHINASOFT_PORT="${CHINASOFT_PORT:-20600}"
# A different project name cannot silently target an unrelated Compose stack.
export MANDOL_COMPOSE_PROJECT="${MANDOL_COMPOSE_PROJECT:-mandol-os-native}"
case "$MANDOL_COMPOSE_PROJECT" in mandol-os-*) ;; *) echo 'MANDOL_COMPOSE_PROJECT must start with mandol-os-' >&2; exit 2;; esac
