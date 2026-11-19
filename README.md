# Non-Linear

本项目实现 Llama decode 阶段的 W4A16 RMSNorm + Linear 融合。当前主线支持 FP16 激活、group size 128 的 AWQ 格式 W4 权重，以及 `[Batch, 1, HiddenDim]` 输入，batch 范围为 1–8。

模型加载时，每个输出 stripe 预填一列全 1 权重，并按 MMA fragment 顺序重排 W4 数据。推理时，CTA 内生成 `x * gamma` 和 `x²`，使用 FP16 Tensor Core 计算线性输出和平方和，再在同一次 CUDA launch 内完成 RMSNorm 缩放。K-split 使用 FP32 workspace。当前 kernel 复用了 Marlin U4 解包原语，并使用本项目的权重布局和调度。

## 环境搭建

准备 Linux x86_64、Conda、CUDA 12.4 Toolkit、兼容的 NVIDIA 驱动、C++17 编译器，以及计算能力 8.0 或更高的 NVIDIA GPU。在项目根目录运行：

```bash
bash setup.sh
```

脚本创建 `nonl` 环境，安装 Python 3.10.18、PyTorch 2.6.0+cu124、vLLM 0.8.5 等依赖，并运行 GPU 回归。已有环境可使用 `bash setup.sh --check`。若 `nvcc` 不在 PATH 中，设置 `CUDA_HOME`。用 `CUDA_VISIBLE_DEVICES` 选择 GPU。

```bash
CUDA_HOME=/path/to/cuda-12.4 CUDA_VISIBLE_DEVICES=0 bash setup.sh --check
```

## 运行测试

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nonl --no-capture-output \
  pytest -q test/fused_rmsnorm_linear_w4a16_marlin
```

首次执行 CUDA 路径时，PyTorch 会 JIT 编译扩展。可用 `TORCH_EXTENSIONS_DIR` 指定编译缓存目录。验证 GPU 时优先运行 `setup.sh --check`，它会显式检查 CUDA 是否可用。

## 复现性能测试

模型权重需单独准备，并通过 `--model` 指定本地 checkpoint 目录。

**全层对比**。对 Llama2-7B 或 Llama3-8B 的 Hugging Face safetensors checkpoint 运行下列命令。两个模型分别指定各自的 `--model` 和新的 CSV 文件名。

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nonl --no-capture-output python -m \
  benchmark.fused_rmsnorm_linear_w4a16_marlin.bench_llama_all_layers \
  --model /path/to/llama3-8b --csv /tmp/llama3-8b-decode.csv
```

脚本测试每个 decoder block 的 RMSNorm + 合并 QKV 和 RMSNorm + 合并 gate-up，覆盖 bs=1–8。三条路径使用同一份离线量化的 W4 权重。PyTorch eager 使用预先反量化的 FP16 权重，vLLM 使用 AWQ-Marlin，融合版使用默认 N tile=64。CSV 记录各层 p50 延迟和输出余弦相似度。测量范围为单个融合层的 decode 延迟。

**单投影对比**。使用已有 AWQ checkpoint，可调整 batch 和 CTA N tile。这里的 `gate` 表示单独的 `gate_proj`。

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nonl --no-capture-output python -m \
  benchmark.fused_rmsnorm_linear_w4a16_marlin.bench_real_awq_projection \
  --model /path/to/awq-checkpoint --projection gate --batch 4 --stripe-n 128
```

## 目录

主线子目录均名为 `fused_rmsnorm_linear_w4a16_marlin`：

- `kernels/` 存放 CUDA kernel、C++ binding 和 Marlin U4 原语。
- `layers/` 存放 PyTorch layer 和离线权重打包。
- `test/` 验证打包、CUDA 输出和 workspace 复用。
- `benchmark/` 存放测试脚本、vLLM 基线辅助代码和历史测量记录。
