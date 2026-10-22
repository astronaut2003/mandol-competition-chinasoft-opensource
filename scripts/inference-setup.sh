#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
command -v uv >/dev/null || { echo 'Install uv first. See docs/DEPLOYMENT.md.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/GPU is unavailable. Configure remote HTTP endpoints instead.' >&2; exit 2; }
case "${1:-all}" in
  all) MANDOL_RECIPES=(029 025);;
  029|025) MANDOL_RECIPES=("$1");;
  *) echo 'Usage: inference-setup.sh [all|029|025]' >&2; exit 2;;
esac
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
for MANDOL_RECIPE in "${MANDOL_RECIPES[@]}"; do
  MANDOL_TARGET="$MANDOL_ROOT/.venv-vllm-$MANDOL_RECIPE"
  MANDOL_LOCK="$MANDOL_ROOT/deploy/inference/vllm-$MANDOL_RECIPE.lock"
  echo "Installing isolated GPU runtime $MANDOL_RECIPE from its hash-locked dependencies."
  [[ -x "$MANDOL_TARGET/bin/python" ]] || uv venv --no-config --python 3.12 "$MANDOL_TARGET"
  # The Web workspace pins protobuf 7. Its constraints must never leak into the
  # model environment. --no-config isolates configuration, not dependency checks.
  uv pip sync --no-config --python "$MANDOL_TARGET/bin/python" --require-hashes "$MANDOL_LOCK"
  uv pip check --no-config --python "$MANDOL_TARGET/bin/python"
  "$MANDOL_TARGET/bin/python" - <<'PY'
import importlib.metadata as m,json,torch
if not torch.cuda.is_available():raise SystemExit('Installed runtime cannot access CUDA. Check your driver/device permissions.')
value=(torch.ones(8,device='cuda')*2).sum().item()
assert value==16
print(json.dumps({'vllm':m.version('vllm'),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'protobuf':m.version('protobuf'),'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'cuda_compute':True}))
PY
done
printf '%s\n' 'GPU environments installed; no model service was started.' 'Run ./scripts/download-models.sh, configure *_MANAGED=true, then ./scripts/inference-start.sh.'
