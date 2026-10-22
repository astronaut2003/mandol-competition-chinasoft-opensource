#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
for tool in uv node npm docker; do command -v "$tool" >/dev/null || { echo "Missing $tool. See docs/DEPLOYMENT.md" >&2; exit 2; }; done
docker compose version >/dev/null
[[ -f .env ]] || { cp .env.example .env; chmod 600 .env; echo 'Created .env; existing configuration is never overwritten.'; }
echo '[1/4] Install the pinned Python environment'
uv sync --locked --python 3.12 --group dev
echo '[2/4] Install frontend dependencies from lockfile'
npm --prefix studio ci --no-audit --no-fund
echo '[3/4] Build the frontend'
npm --prefix studio run build
echo '[4/4] Initialize independent SQLite schema'
.venv/bin/python scripts/manage.py init
printf '%s\n' 'Setup complete. Configure model endpoints and DeepSeek in .env.' 'No GPU model or application service was started.' 'For an explicit basic/degraded check use ./scripts/start.sh --basic.'
