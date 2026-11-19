#!/usr/bin/env bash
# Linux x86_64. Requires Conda, a CUDA 12.4 toolkit, and a C++17 compiler.
set -euo pipefail

case "${1:-}" in
  -h|--help)
    echo 'Usage: bash setup.sh [--check]'
    echo 'Create the nonl Conda environment and run the GPU regression suite.'
    echo 'Use --check to test an existing environment without installing packages.'
    echo 'Set CUDA_HOME if nvcc is outside PATH. Set CUDA_VISIBLE_DEVICES to select a GPU.'
    exit 0
    ;;
  ""|--check) ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash setup.sh [--check]' >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v conda >/dev/null
command -v "${CXX:-c++}" >/dev/null
if [[ -z "${CUDA_HOME:-}" ]]; then
  nvcc_path="$(command -v nvcc)" || {
    echo 'Set CUDA_HOME to the CUDA 12.4 toolkit directory.' >&2
    exit 1
  }
  CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$nvcc_path")")")"
fi
export CUDA_HOME
if [[ "$("$CUDA_HOME/bin/nvcc" --version)" != *"release 12.4,"* ]]; then
  echo 'The reproduced environment requires a CUDA 12.4 toolkit.' >&2
  exit 1
fi
nvidia-smi -L

run=(conda run -n nonl --no-capture-output)
if [[ "${1:-}" != --check ]]; then
  if "${run[@]}" python --version >/dev/null 2>&1; then
    echo 'The nonl environment already exists. Use bash setup.sh --check.' >&2
    exit 1
  fi
  conda create --yes --name nonl python=3.10.18 pip
  "${run[@]}" python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
  "${run[@]}" python -m pip install \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    vllm==0.8.5 safetensors==0.8.0 pytest==9.1.1 ninja==1.13.0
fi

"${run[@]}" python -m pip check
"${run[@]}" python -c '
import torch
from vllm import _custom_ops

assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
assert torch.cuda.get_device_capability()[0] >= 8, "An SM80 or newer GPU is required"
assert hasattr(torch.ops._C, "awq_marlin_repack"), "vLLM AWQ operators are unavailable"
torch.empty(1, device="cuda").fill_(1)
torch.cuda.synchronize()
print(torch.cuda.get_device_name())
'
"${run[@]}" python -m pytest -q test/fused_rmsnorm_linear_w4a16_marlin
