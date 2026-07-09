# Non-Linear Project Guide

## Project Overview

Non-Linear is a high-performance deep learning project focused on low-bit LLM
quantization and custom GPU kernels. The current project scope is limited to
normalization layers, specifically RMSNorm and LayerNorm, for inference and
training efficiency.

Do not implement or optimize unrelated layer types such as activations, SwiGLU,
GeLU, GLU, Hadamard transforms, attention, or GEMM unless the user explicitly
requests that work.

## Directory Structure

Keep the repository root clean. Place new project files in first-level
directories according to their purpose:

- `models/`: LLM architectures, model configs, and network definitions.
- `layers/`: PyTorch modules and wrapper layers, especially RMSNorm and
  LayerNorm implementations.
- `kernels/`: CUDA, CuTe, CUTLASS, and C++ GPU kernel implementations.
- `test/`: Correctness validation, unit tests, benchmark drivers, and profiling
  scripts.

Do not place tests, kernels, model code, generated logs, build outputs, or
experiments directly in the repository root.

## Environment

Use the `nonl` conda environment for all Python execution and tests:

```bash
conda run -n nonl --no-capture-output pytest test/
conda run -n nonl --no-capture-output python kernels/bench.py
```

Do not assume dependencies are installed globally. Do not install, upgrade, or
modify packages in the environment unless the user explicitly asks for that.

## Coding Style

### Python

- Follow PEP 8.
- Use descriptive names for modules, functions, classes, variables, and tests.
- Add explicit type hints to all function signatures.
- Keep PyTorch eager reference implementations simple and readable before
  adding optimized kernels.
- Support mixed precision where relevant, especially `torch.float16` and
  `torch.bfloat16`.
- Handle underflow and overflow risks in normalization math and quantization
  scales.
- Document tensor shape transforms in docstrings or comments using the form
  `[Batch, SeqLen, HiddenDim]`.

### CUDA, CuTe, and C++

- Prefer CUDA and CuTe for new kernel work.
- Avoid Triton for new kernels unless the user explicitly requests it or
  CUDA/CuTe is infeasible for the task.
- Decouple host C++ bindings from device `.cu` kernel code.
- Use error-checking macros for every CUDA allocation, kernel launch, and
  synchronization point.
- Use CuTe layouts and tiling where appropriate.
- Use Tensor Cores or CUTLASS where they fit the normalization workload.

## Testing

Place all tests and verification scripts under `test/`. Do not add tests in the
repository root or next to implementation files.

When adding or modifying an RMSNorm or LayerNorm layer or kernel:

1. Add or update a focused correctness test in `test/`.
2. Compare optimized results against a clean PyTorch eager reference.
3. Use `torch.allclose` with `atol=1e-3` and `rtol=1e-3` for FP16 correctness
   unless the user specifies stricter tolerances.
4. Run tests through the project conda environment:

```bash
conda run -n nonl --no-capture-output pytest test/
```

Benchmarking or profiling scripts may also live in `test/` when they validate
normalization behavior or performance. Keep generated logs and binary build
artifacts out of the repository unless explicitly requested.

## Development Workflow

- Work test-first for normalization layers and kernels.
- Establish correctness with a PyTorch reference before optimizing CUDA/CuTe.
- Make incremental changes. Avoid broad rewrites of working code unless the
  user asks for them.
- Keep changes scoped to the requested normalization work.
- Preserve existing user changes in the working tree.
- Do not introduce new top-level directories unless they match the structure
  above or the user explicitly approves them.
