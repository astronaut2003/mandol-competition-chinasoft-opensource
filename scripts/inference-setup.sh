#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
command -v uv >/dev/null || { echo 'Install uv first. See docs/DEPLOYMENT.md.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/GPU is unavailable. Configure remote HTTP endpoints instead.' >&2; exit 2; }
case "${1:-unified}" in
  all|unified) ;;
  *) echo 'Usage: inference-setup.sh [unified]' >&2; exit 2;;
esac
python3 - <<'PY'
import platform
if platform.machine() != 'x86_64':
    raise SystemExit('The qualified lock targets Linux x86_64. Other platforms need a separately validated lock.')
if tuple(map(int, platform.libc_ver()[1].split('.')[:2])) < (2, 31):
    raise SystemExit('The official llguidance wheel requires glibc >=2.31.')
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
MANDOL_TARGET="$MANDOL_ROOT/.venv-vllm-unified"
MANDOL_LOCK="$MANDOL_ROOT/deploy/inference/vllm-unified.lock"
echo 'Installing vLLM 0.32.0 / Python 3.12 from hash-locked official wheels.'
[[ -x "$MANDOL_TARGET/bin/python" ]] || uv venv --no-config --no-project --python 3.12 "$MANDOL_TARGET"
# The Web workspace pins protobuf 7. Its constraints must never leak into the
# model environment. --no-config isolates configuration, not dependency checks.
uv pip sync --no-config --python "$MANDOL_TARGET/bin/python" --only-binary :all: --require-hashes "$MANDOL_LOCK"
uv pip check --no-config --python "$MANDOL_TARGET/bin/python"
"$MANDOL_TARGET/bin/python" - <<'PY'
import importlib.metadata as m,json,torch
import sys
assert sys.version_info[:2] == (3, 12) and m.version('vllm') == '0.32.0'
if not torch.cuda.is_available():raise SystemExit('Installed runtime cannot access CUDA. Check your driver/device permissions.')
value=(torch.ones(8,device='cuda')*2).sum().item()
assert value==16
print(json.dumps({'vllm':m.version('vllm'),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'protobuf':m.version('protobuf'),'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'cuda_compute':True}))
PY
printf '%s\n' 'Unified GPU environment installed; no model service was started.' 'Run ./scripts/download-models.sh, configure *_MANAGED=true, then ./scripts/inference-start.sh.'
