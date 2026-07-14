# Fused RMSNorm + Linear 实现设计

本文档描述当前项目中小 batch、单 token 场景下的
`FusedRMSNormLinear` 推理实现，包括数学变换、计算流、CUDA kernel
设计、Torch 层封装、精度处理、性能优化和已知限制。

相关文件：

- `layers/fused_rmsnorm_linear/__init__.py`：Torch 层及权重预融合。
- `layers/fused_rmsnorm_linear/cuda.py`：CUDA 扩展的 JIT 加载器。
- `kernels/fused_rmsnorm_linear/fused_rmsnorm_linear.cpp`：PyTorch C++ binding 与参数检查。
- `kernels/fused_rmsnorm_linear/fused_rmsnorm_linear.cu`：projection、平方和与 RMS scaling kernel。
- `test/fused_rmsnorm_linear/test_fused_rmsnorm_linear.py`：正确性测试。
- `test/fused_rmsnorm_linear/bench_fused_rmsnorm_linear.py`：与原生 PyTorch 实现的性能对比。

## 1. 目标计算

输入和权重定义如下：

```text
x:                 [Batch, 1, HiddenDim]
rmsnorm_weight:    [HiddenDim]
linear_weight:     [OutDim, HiddenDim]
output:            [Batch, 1, OutDim]
```

当前实现限定：

- `x`、RMSNorm weight 和 Linear weight 均为 FP16。
- Linear 可以带或不带 FP16 bias。
- `SeqLen == 1`。
- 优化 CUDA 路径支持 `1 <= Batch <= 8`。
- 目标 GPU 为 Ampere，包括 A100（SM80）和 RTX 3090（SM86）。
- 该层面向固定权重推理，不支持训练时更新融合权重。

标准 RMSNorm + Linear 的数学表达式为：

```text
variance[b] = sum_k(x[b, k]^2) / HiddenDim
inverse_rms[b] = rsqrt(variance[b] + eps)

normalized[b, k] = x[b, k] * inverse_rms[b] * rmsnorm_weight[k]

output[b, n] = sum_k(
    normalized[b, k] * linear_weight[n, k]
) + linear_bias[n]
```

将 RMSNorm weight 展开后可得：

```text
output[b, n] = inverse_rms[b] * sum_k(
    x[b, k]
    * rmsnorm_weight[k]
    * linear_weight[n, k]
)
```

`inverse_rms[b]` 对同一 batch 内的所有 K、N 元素都是标量，因此可以将
RMSNorm weight 提前吸收到 Linear weight 中。

## 2. 权重预融合

Torch 层初始化时计算：

```python
fused_weight = linear.weight * rmsnorm.weight.unsqueeze(0)
```

其形状保持为：

```text
fused_weight: [OutDim, HiddenDim]
```

最终计算可重写为：

```text
numerator[b, n] = sum_k(x[b, k] * fused_weight[n, k])
square_sum[b] = sum_k(x[b, k]^2)
inverse_rms[b] = rsqrt(square_sum[b] / HiddenDim + eps)
output[b, n] = numerator[b, n] * inverse_rms[b] + linear_bias[n]
```

预融合只发生在 `FusedRMSNormLinear.__init__` 中，不进入 forward 的性能计时。
融合后的权重通过 `register_buffer` 保存，具有以下性质：

- 会进入层的 `state_dict`。
- 会随 `module.to(device)` 和 dtype/device 迁移操作移动。
- 不作为可训练参数出现在 `named_parameters()` 中。
- 原始 RMSNorm 和 Linear 模块不会被该层继续持有。
- 原始模块权重后续发生变化时，buffer 不会自动更新。

## 3. 旧计算流及其问题

早期方案复用了 `fused_square_linear` kernel。该方案在 fused weight 后追加一行
全 1 权重：

```text
augmented_weight = concat([fused_weight, ones], dim=0)
```

然后计算：

```text
stack([x, x^2], dim=1) @ augmented_weight.T
```

输出形状为 `[Batch, 2, OutDim + 1]`，其中真正需要的内容只有：

```text
output[:, 0, :OutDim]  = x @ fused_weight.T
output[:, 1, OutDim]   = sum(x^2)
```

其余内容全部无用：

```text
output[:, 0, OutDim]   = sum(x)
output[:, 1, :OutDim]  = x^2 @ fused_weight.T
```

这导致以下性能浪费：

1. 为得到一个平方和，额外执行了完整的 `x^2 @ fused_weight.T`。
2. 计算并写回了无用的 `sum(x)`。
3. `[Batch, 2, OutDim + 1]` 中接近一半的数据不会被最终层使用。
4. 平方和经过 FP16 输出，存在精度损失及大数溢出风险。
5. Python 端还需分别启动类型转换、除法、`rsqrt`、乘法和 cast kernel。

因此当前实现不再通过全 1 行和 `fused_square_linear` 间接计算，而是使用专用
RMSNorm + Linear CUDA 扩展。

## 4. 当前计算流

当前 forward 包含两个 CUDA kernel：

```text
                       ┌─────────────────────────┐
x ────────────────────▶│ projection + square sum │
fused_weight ─────────▶│                         │
                       └───────────┬─────────────┘
                                   │
                    ┌──────────────┴─────────────┐
                    │                            │
              numerator FP16              square_sum FP32
              [B, 1, N]                       [B]
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                         ┌───────────────────┐
                         │ RMS scale epilogue │
                         └─────────┬─────────┘
                                   ▼
                           output [B, 1, N]
```

### Kernel 1：Projection + Square Sum

第一个 kernel 同时完成：

```text
numerator = x @ fused_weight.T
square_sum = sum(x.float()^2)
```

但两部分采用不同的并行方式：

- 所有 CTA 共同计算 projection。
- 仅 `blockIdx.x == 0` 的 CTA 额外计算每个 batch 的平方和。

这样可以避免为平方和单独启动第三个 kernel，同时不会在每个输出 tile 中重复计算
平方和。

### Kernel 2：RMS Scaling Epilogue

第二个 kernel 对 numerator 原位缩放：

```text
inverse_rms[b] = rsqrtf(square_sum[b] / HiddenDim + eps)
output[b, n] = half(float(numerator[b, n]) * inverse_rms[b])
```

它使用一维线程映射遍历 `[Batch * OutDim]`，每个线程处理一个输出元素。

## 5. Projection Kernel 的线程映射

当前 projection 采用 CUDA Core SIMT，而不是 Tensor Core。原因是主要应用场景为
小 batch：逻辑 GEMM 的 M 等于 Batch，在 B=1 等场景下远小于 Ampere FP16 Tensor
Core 常用的 `m16n8k16` 最小 M tile。

kernel 配置为：

```text
warps per CTA:    8
threads per CTA:  256
one warp:         one output channel n
one lane:         a strided portion of K
```

N 维映射：

```text
output_index = blockIdx.x * 8 + warp_id
```

K 维映射在 HiddenDim 为偶数时使用 `half2`：

```text
pair_index = lane_id + iteration * 32
```

因此一个 warp 的32个 lane 会连续读取64个 FP16 元素：

```text
lane 0  → K[0:2]
lane 1  → K[2:4]
...
lane 31 → K[62:64]
```

这种布局使对同一个 weight row 的读取保持合并访存。每次加载的 weight pair 会在
寄存器中复用于模板中所有 batch：

```text
load weight[n, k:k+2] once

for b in Batch:
    load x[b, k:k+2]
    accumulator[b] += x[b, k:k+2] * weight[n, k:k+2]
```

每个线程维护 `Batch` 个 FP32 accumulator。K 循环完成后，通过五步 warp shuffle
完成归约：

```text
offset = 16, 8, 4, 2, 1
```

最终由 lane 0 将每个 batch 的 projection 输出转换为 FP16 并写回。

### Batch 模板实例化

为了让编译器展开 batch 循环并将 accumulator 固定放入寄存器，当前生成了
B=1～8 的独立模板实例：

```cpp
fused_projection_square_sum_kernel<1>
...
fused_projection_square_sum_kernel<8>
```

host launcher 根据输入 batch 选择对应实例。这样避免了动态 batch 循环导致的
local-memory 数组和额外索引开销。

### 奇数 HiddenDim

当 HiddenDim 为偶数时走 `half2` 快速路径。当 HiddenDim 为奇数时，batch row
之间可能不满足 `half2` 对齐要求，因此自动切换为标量 FP16 load，保证任意 K 的
正确性。

## 6. FP32 Square-Sum 归约

平方和由第一个 CTA 的全部256个线程完成：

```text
thread t processes k = t, t+256, t+512, ...
```

每个线程为所有 batch 保存独立 FP32 partial sum：

```text
local_square_sum[b] += float(x[b, k]) * float(x[b, k])
```

归约分两级进行：

1. 每个 warp 使用 `__shfl_down_sync` 得到 warp-level sum。
2. 每个 warp 的 lane 0 将结果写入极小的 shared memory。
3. 前 Batch 个线程分别汇总8个 warp 的 partial sum。

shared memory 大小为：

```text
Batch * 8 * sizeof(float)
```

例如 B=1 时仅32字节，B=8 时也只有256字节。

与旧实现相比，square-sum 从输入到最终统计值始终使用 FP32，不再经过 FP16
中间输出。这既提高精度，也避免 `sum(x^2)` 超过 FP16 最大有限值时直接变成
`inf`。

## 7. RMS Scaling Epilogue

epilogue 使用256线程的一维 block：

```text
index = blockIdx.x * 256 + threadIdx.x
batch_index = index / OutDim
```

每个线程读取对应 batch 的 FP32 square-sum，计算 inverse RMS，并缩放一个 FP16
numerator 元素。

当前每个线程都会执行一次 `rsqrtf`。对 B=1、N=4096，这意味着4096次相同的
`rsqrtf`。这样设计保持 kernel 简单且无同步，但仍存在进一步优化空间：可以由每个
block 只计算一次 inverse RMS，并通过 shared memory 或 kernel 参数缓存给该 block
中的线程。

## 8. 数值精度

当前数值路径如下：

```text
weight fusion:       FP16 multiply and FP16 storage
projection input:    FP16
projection product:  converted to FP32 FMA
projection sum:      FP32 accumulator
numerator storage:   FP16
square operation:    FP32
square sum:          FP32
inverse RMS:         FP32 rsqrtf
final scaling:       FP32 multiply, then FP16 storage
```

与原生 PyTorch RMSNorm + Linear 不会逐位一致，主要差异来自：

- RMSNorm weight 被提前乘入 Linear weight，并舍入到 FP16。
- projection 的归约顺序不同。
- numerator 在 RMS scaling 前暂存为 FP16。

测试使用适合 FP16 推理的误差容限与 PyTorch reference 对比。

## 9. Torch 层与 C++ Binding

### Torch 层

`FusedRMSNormLinear` 初始化时执行：

1. 若 Linear 带 bias，则将其快照保存为非参数 buffer。
2. 检查 RMSNorm HiddenDim 与 Linear in_features 一致。
3. 检查权重 device 和 dtype 一致。
4. 从 `variance_epsilon`、`eps` 或 dtype epsilon 中提取 RMS epsilon。
5. 预计算并保存 `[OutDim, HiddenDim]` fused weight。

forward 检查输入形状、device 和 dtype，然后根据 bias 是否存在分派到无 bias 或
带 bias 的专用扩展，返回 `[Batch, 1, OutDim]`。带 bias 路径将加法融合进 RMS
scale epilogue，不会额外启动 PyTorch elementwise kernel。

### C++ Binding

C++ binding 负责：

- CUDA、FP16、contiguous 和 shape 检查。
- 限制优化路径 Batch 在 `[1, 8]`。
- 设置正确的 CUDA device guard。
- 使用 PyTorch 当前 CUDA stream。
- 分配最终 FP16 output 和临时 FP32 square-sum。
- 顺序启动 projection 和无 bias scale kernel，或 projection 和 scale+bias kernel。
- 对每次 kernel launch 执行 CUDA error check。

临时 `square_sum` 只有 `Batch * sizeof(float)` 字节，其分配仍可能产生少量 allocator
开销。未来可以考虑将它改为 workspace、复用 buffer，或在支持条件下使用更深度的
kernel fusion。

## 10. 编译目标与资源使用

Python loader 默认设置：

```text
TORCH_CUDA_ARCH_LIST=8.0;8.6+PTX
```

对应：

- SM80：NVIDIA A100。
- SM86：NVIDIA RTX 3090。

CUDA 12.4 编译观察到：

```text
B=1 projection:
  SM80: 30 registers/thread
  SM86: 37 registers/thread
  shared memory: 32 bytes

B=8 projection:
  SM80/SM86: 40 registers/thread
  shared memory: 256 bytes

RMS scale epilogue:
  12 registers/thread

所有实例:
  stack frame: 0 bytes
  spill stores: 0 bytes
  spill loads: 0 bytes
```

## 11. 正确性测试

运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
pytest -q test/fused_rmsnorm_linear/test_fused_rmsnorm_linear.py
```

测试覆盖：

- fused weight 是否在 init 时正确生成。
- 层中是否没有残留 trainable parameter。
- 无 bias 与带 bias Linear 的 buffer 和数值行为。
- B=1、2、4、8 的 CUDA 输出。
- 与干净的 PyTorch RMSNorm + Linear reference 对比。
- 最终输出形状是否为 `[Batch, 1, OutDim]`。

## 12. 性能测试

默认场景为 B=1、K=N=4096：

```bash
CUDA_VISIBLE_DEVICES=0 \
python test/fused_rmsnorm_linear/bench_fused_rmsnorm_linear.py
```

更稳定的测试可增加 warmup 和 iteration：

```bash
CUDA_VISIBLE_DEVICES=0 \
python test/fused_rmsnorm_linear/bench_fused_rmsnorm_linear.py \
  --batch 1 \
  --hidden-dim 4096 \
  --out-dim 4096 \
  --eps 1e-6 \
  --warmup 20 \
  --iterations 100
```

benchmark 在计时区域之外完成：

- 原始层构造。
- fused weight 预计算。
- JIT extension 首次编译。
- 输入和权重初始化。
- 计时前正确性检查。

对比对象为共享同一组输入和原始权重的：

```text
FusedRMSNormLinear
PyTorch nn.RMSNorm + nn.Linear（由 benchmark 的 `--bias/--no-bias` 控制）
```

报告 mean、P50、P95、有效 TFLOPS 和相对 PyTorch speedup。

## 13. 当前限制

1. 只支持 FP16，不支持 BF16。
2. 只支持 Ampere compute capability 8.x。
3. 只支持 SeqLen=1。
4. 专用路径只支持 Batch 1～8。
5. 只面向 inference，不实现 backward。
6. fused weight 和 bias 是原始权重的快照，不会随源模块更新。
7. numerator 在 scaling 前暂存为 FP16，仍有一次中间舍入。
8. forward 会为 FP32 square-sum 申请一个极小临时 tensor。

## 14. 后续优化方向

按优先级可继续尝试：

1. **缓存 inverse RMS**：每个 scale block 只执行一次 `rsqrtf`，通过 shared memory
   广播给线程。
2. **向量化 epilogue**：在 OutDim 对齐时使用 `half2` 或更宽的向量化 load/store。
3. **复用 square-sum workspace**：减少 PyTorch allocator 参与 forward 的固定开销。
4. **合并最终 scaling**：探索 cooperative launch、persistent kernel 或其他可安全完成
   跨 CTA 同步的方案。
5. **FP32 numerator workspace**：评估避免 scaling 前 FP16 舍入的精度收益与额外带宽。
6. **BF16 specialization**：增加 BF16 input、weight 和输出支持。
7. **按 K/N 调整 CTA warps**：为较小或较大的 OutDim 选择4、8或更多 warp。
8. **CUDA Graph benchmark**：区分 kernel 本体耗时与 Python/dispatcher launch 开销。
9. **Nsight Compute 分析**：重点检查 DRAM throughput、L2 hit rate、FP32 pipe 利用率、
   warp occupancy 和 epilogue 的 special-function-unit 使用情况。

当前设计的核心原则是：针对小 batch、单 token 的真实形状，将问题视为共享权重读取
的多向量 GEMV，并只计算 RMSNorm + Linear 最终结果真正需要的数据，避免为了复用
通用 GEMM 接口而引入无效矩阵行、无效平方投影和额外全局内存流量。
