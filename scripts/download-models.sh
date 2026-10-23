#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
[[ -x .venv-vllm-unified/bin/python ]] || { echo 'Run ./scripts/inference-setup.sh first.' >&2; exit 2; }
exec .venv-vllm-unified/bin/python scripts/download_models.py "$@"
