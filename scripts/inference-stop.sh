#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec .venv/bin/python scripts/chinasoft_vllm.py stop "$@"
