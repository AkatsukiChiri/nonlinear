#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16"}
HIDDEN_DIM=${HIDDEN_DIM:-4096}
OUT_DIM=${OUT_DIM:-4097}
WARMUP=${WARMUP:-10}
ITERATIONS=${ITERATIONS:-50}

if [[ "${CONDA_DEFAULT_ENV:-}" == "nonl" ]]; then
  PYTHON_CMD=(python)
  PYTEST_CMD=(python -m pytest)
else
  PYTHON_CMD=(conda run -n nonl --no-capture-output python)
  PYTEST_CMD=(conda run -n nonl --no-capture-output python -m pytest)
fi

echo "Running fused square-and-linear correctness tests"
"${PYTEST_CMD[@]}" -q test/fused_square_linear/test_fused_square_linear.py

echo
echo "Running performance sweep"
echo "Batches: ${BATCH_SIZES}"
echo "K=${HIDDEN_DIM}, N=${OUT_DIM}, warmup=${WARMUP}, iterations=${ITERATIONS}"

for batch in ${BATCH_SIZES}; do
  echo
  echo "========== Batch ${batch} =========="
  "${PYTHON_CMD[@]}" test/fused_square_linear/bench_fused_square_linear.py \
    --batch "${batch}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --out-dim "${OUT_DIM}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}"
done
