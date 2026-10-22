#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
[[ -x .venv-vllm-029/bin/python ]] || { echo 'Run ./scripts/inference-setup.sh first.' >&2; exit 2; }
exec .venv-vllm-029/bin/python scripts/download_models.py "$@"
