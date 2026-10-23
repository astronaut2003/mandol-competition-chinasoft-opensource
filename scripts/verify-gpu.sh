#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
[[ -x .venv/bin/python ]] || { echo 'Run ./scripts/setup.sh first.' >&2; exit 2; }
exec .venv/bin/python scripts/verify-gpu.py "$@"
