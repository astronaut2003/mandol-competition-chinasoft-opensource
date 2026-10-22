#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/GPU is unavailable.' >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
for recipe in 029:0.29.0 025:0.25.1; do
  suffix="${recipe%%:*}"; version="${recipe#*:}"; target=".venv-vllm-$suffix"
  echo "Install isolated vLLM $version in $target (CUDA dependencies can require several GB)."
  [[ -x "$target/bin/python" ]] || uv venv --python 3.12 "$target"
  uv pip install --python "$target/bin/python" "vllm==$version"
  "$target/bin/python" -c 'import importlib.metadata,torch; print("vLLM",importlib.metadata.version("vllm"),"CUDA",torch.cuda.is_available())'
done
printf '%s\n' 'Model weights are not downloaded automatically.' 'Run ./scripts/download-models.sh, configure *_MANAGED=true, then explicitly run ./scripts/inference-start.sh.'
