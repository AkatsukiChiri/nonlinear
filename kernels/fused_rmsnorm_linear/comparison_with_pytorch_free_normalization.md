# 与 PyTorch “Towards Free Normalization” 实现的对比

本文对比当前项目的 `FusedRMSNormLinear` 与 PyTorch 官方博客
[Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)
中介绍的归一化融合方法。

对比依据：

- 文章发表于 2026 年 7 月 10 日。
- 文章公开代码指向 Meta Ads Model Kernel Library 中的
  `multi_cta_norm_fusion` 和 `gdpa_megakernel`。
- 本项目实现以当前仓库中的
  `layers/fused_rmsnorm_linear/__init__.py`、
  `kernels/fused_rmsnorm_linear/fused_rmsnorm_linear.cu` 和
  `kernels/fused_rmsnorm_linear/fused_rmsnorm_linear.cpp` 为准。

## 1. 结论摘要

两者共享相同的核心数学观察：RMSNorm 的 inverse RMS 是 row-wise 标量，可以把
它从 GEMM 输入端延迟到 GEMM 输出端：

```text
(A * rstd[:, None]) @ B = (A @ B) * rstd[:, None]
```

PyTorch 文章将这一方法称为 **Lazy Pre-Norm**。其目标是在标准 Tensor Core GEMM
内部一边执行 K-loop，一边累计输入平方和，最后在同一个 kernel 的 epilogue 中应用
`rstd`。更广泛的方案还包括基于 CTA cluster 和 Distributed Shared Memory 的
Multi-CTA Norm，以及把多个 norm、residual 和 attention 融合为一个 megakernel 的
FlashNormAttention。

本项目采用相同代数变换，但针对完全不同的工作负载做了更窄的工程选择：

```text
Batch:       1～8
SeqLen:      1
K/N:         常见为 4096
GPU:         RTX 3090 / A100，Ampere
dtype:       FP16
模式:        固定权重推理
```

当前实现将问题视为多向量 GEMV，而不是标准大 M GEMM：

1. 初始化时把 RMSNorm affine weight 吸收到 Linear weight。
2. 第一个 SIMT kernel 计算 projection，并由第一个 CTA 计算一次 FP32 square-sum。
3. 第二个轻量 kernel 应用 inverse RMS scaling。

因此，当前实现相对文章方案的主要优势是：更适合极小 batch、支持静态 RMSNorm
affine、平方和不在 N-direction CTA 间重复计算、Ampere 上实现简单且资源占用低。

主要劣势是：还不是单 kernel 的“近乎免费” normalization，无法把 CUDA Core norm
计算与 Tensor Core GEMM 重叠；需要中间 numerator HBM 写回和第二次 launch；不支持
训练、backward、LayerNorm、post-norm、attention fusion、BF16 或大 batch。

## 2. PyTorch 文章提出的方案

### 2.1 问题背景

文章认为 LayerNorm/RMSNorm 的主要性能问题是：

- normalization 是低算术强度、memory-bound 的 reduction + elementwise 操作；
- normalization 不利用 Tensor Core；
- 在其训练模型中，normalization 可能占约 10%～20% 总延迟；
- norm 通常紧邻 GEMM 或 attention，因此适合通过 fusion 减少 HBM IO，并把
  CUDA Core 工作与 Tensor Core 工作重叠。

文章的主要实验环境为：

```text
GPU:       NVIDIA B200
dtype:     BF16
power cap: 750 W
DSL:       TLX 和 Helion
场景:      Meta ads/recommendation training workloads
```

这与我们的 Ampere、FP16、小 batch inference 环境明显不同，文章性能数字不能直接
与本项目 benchmark 横向比较。

### 2.2 Naive GEMM + Norm Fusion

文章首先讨论一种直接方案：令 GEMM tile 在 normalization reduction 维度上覆盖
完整一行，使同一个 CTA 拿到 normalization 所需的所有数据。

问题是：

- GEMM tile 被 normalization 强制改变，偏离纯 GEMM 的最佳 tiling；
- cache 和 pipeline 效率下降；
- shared memory 对完整 N tile 的容量形成硬限制；
- 小 N（例如64、128）可能受益，N 增大后会退化甚至不可行。

文章报告该方案在小形状上可节省约17%～32%的 LayerNorm latency，但随着 K/N
超过128，收益逐渐消失，甚至出现明显 regression。

### 2.3 Lazy Pre-Norm

Lazy Pre-Norm 面向：

```text
C = rmsnorm(A) @ B
rmsnorm(A) = A * rstd(A)[:, None]
```

普通 prologue fusion 存在循环依赖：

- 只有扫描完整个 K-loop 后才能得到 `rstd`；
- 但如果先显式 normalize A，每个 K tile 在进入 GEMM 前又需要 `rstd`。

Lazy Pre-Norm 将问题拆成：

- reduction：在扫描 A 的 K tile 时累计 square-sum；
- row-wise scaling：利用矩阵乘法结合律延迟到 GEMM epilogue。

文章伪代码可概括为：

```text
square_sum = 0
acc = 0

for k_tile:
    tile_A = load A
    tile_B = load B
    acc += tile_A @ tile_B
    square_sum += sum(tile_A * tile_A)

rstd = rsqrt(square_sum / K + eps)
acc *= rstd[:, None]
store acc
```

文章特别强调通过 warp specialization，让 square-sum 的 CUDA Core 计算尽量与
Tensor Core GEMM 重叠。

该方案仍有一个已知缺点：计算同一组 M rows、不同 N tiles 的多个 CTA 会重复计算
相同的 square-sum。文章认为只要这部分工作能被 Tensor Core computation 隐藏，重复
计算仍然可以接受。

文章列出的 Lazy Pre-Norm 限制包括：

- 不容易支持 elementwise affine，因为 affine 是 column-wise scaling；
- 不适用于 LayerNorm，因为 LayerNorm 还包含依赖 mean 的减法；
- backward 较复杂，因为 forward 没有 materialize normalized A。

### 2.4 Multi-CTA Norm Fusion

对于 post-norm，文章利用 Blackwell 的 CTA cluster 和 Distributed Shared Memory：

- 多个 CTA 分担同一 row 的 normalization reduction；
- CTA 在 cluster 内通过 DSMEM 交换很小的 reduction result；
- 不需要为了完整 reduction 强迫单个 CTA 覆盖整个 N；
- 可以把 multi-CTA norm 放入 GEMM epilogue。

代价包括：

- 必须采用 persistent kernel；
- 必须使用2D swizzling；
- 无法使用常见 GEMM tile super-grouping；
- cluster/DSMEM 通信和约束可能损害 base GEMM。

文章的主要 forward benchmark 使用大 M 的 ads model 形状，例如 M≈256K、K/N≈512，
而不是本项目的 M=Batch≤8。

### 2.5 Backward Fusion Regrouping

文章还处理训练 backward：forward 的 epilogue fusion 通常会在 backward 中变成性能
不佳的 prologue fusion。其策略不是强行保持 forward/backward 融合对象一致，而是
让 normalization 在 forward 与一个 GEMM 融合，在 backward 与另一个 GEMM 融合，
使两边都保持 epilogue fusion 结构。

这属于模型图级别的 fusion regrouping，超出当前 inference-only 层的范围。

### 2.6 FlashNormAttention

文章最终把 LayerNorm、RMSNorm、residual 和 GDPA/FlashAttention-like kernel 融合成
FlashNormAttention。关键技术包括：

- CTA cluster 和 DSMEM multi-CTA reduction；
- SMEM/TMEM buffer reuse；
- register subtiling；
- load、MMA、activation、norm、epilogue warp specialization；
- backward recomputation，避免 forward 保存大型中间 tensor；
- TMA_REDUCE_ADD；
- Tensor Core accumulate；
- register preloading。

文章称 GEMM fusion 最多能隐藏约90%的 normalization kernel latency，
FlashNormAttention 在其测试中可获得最高约35%的 kernel speedup。

这些结果针对 B200 BF16 训练 workload，不代表3090/A100小 batch inference 的预期
收益。

## 3. 当前项目实现

### 3.1 数学变换

当前层计算：

```text
output = Linear(RMSNorm(x))
```

设 RMSNorm affine weight 为 `gamma`，Linear weight 为 `W`：

```text
output[b, n]
  = rstd[b] * sum_k(x[b, k] * gamma[k] * W[n, k])
```

固定权重推理时可以预计算：

```text
fused_weight[n, k] = W[n, k] * gamma[k]
```

forward 变成：

```text
numerator = x @ fused_weight.T
square_sum = sum(x.float()^2)
rstd = rsqrt(square_sum / K + eps)
output = numerator * rstd
```

### 3.2 Kernel 结构

当前 forward 包含两个 CUDA kernel。

第一个 kernel：

- 每个 CTA 有8个 warp、256个线程；
- 每个 warp 负责一个 output channel；
- lane 沿 K 做连续、合并的 `half2` load；
- weight pair 在寄存器中复用于所有 batch；
- 使用 FP32 FMA accumulator；
- 使用 warp shuffle 完成 dot-product reduction；
- 仅 `blockIdx.x == 0` 的 CTA 计算一次 FP32 square-sum。

第二个 kernel：

- 读取 FP16 numerator 与 FP32 square-sum；
- 计算 `rsqrtf(square_sum / K + eps)`；
- 原位缩放并写回 FP16 output。

### 3.3 设计定位

当前实现没有使用 Tensor Core。对于 B=1，逻辑 GEMM M=1；即使把原始旧算法的
`x/x²` 两行一起考虑，M也仅为2。Ampere 常用 FP16 Tensor Core atom 的 M tile 为16，
会产生严重 padding 和无效计算。

项目实测表明，小 batch SIMT kernel 相比此前 CuTe Tensor Core kernel 更适合这一
形状。因此当前优化目标不是把 normalization 隐藏在高吞吐大 M GEMM 内，而是让
小 M GEMV 的全局内存读取、CUDA Core FMA 和 launch overhead 尽可能小。

## 4. 核心相同点

### 4.1 使用同一个 Lazy Pre-Norm 恒等式

两者都利用：

```text
(A * row_scale) @ B = (A @ B) * row_scale
```

都没有在 GEMM/projection 前显式 materialize RMS-normalized activation。

### 4.2 Reduction 与 row-wise scaling 分离

两者都把 RMSNorm 分成：

```text
reduction:       square_sum → rstd
elementwise:     projection output * rstd
```

并将 row-wise scaling 延迟到 projection 之后。

### 4.3 在读取 x 时顺便计算平方和

两者都避免额外完整读取 x 的独立 RMSNorm kernel，square-sum 与 projection/GEMM 的
输入扫描发生在同一主 kernel 中。

### 4.4 都面临 reduction 与输出 tiles 的协作问题

当 N 被多个 CTA 分割时，同一 row 的 square-sum 如何共享，是两种实现都必须处理
的问题：

- 文章 Lazy Pre-Norm：每个 N tile CTA 重复计算 square-sum。
- 本项目：只允许第一个 CTA 计算 square-sum，依靠下一次 kernel launch 的 stream
  ordering 在 scaling epilogue 中消费结果。

## 5. 当前实现相对文章方案的优势

### 5.1 更适合极小 batch、单 token decode

文章的核心 GEMM fusion 依赖有足够 Tensor Core 工作来隐藏 normalization。其主要
benchmark 是 B200 上的大 M training shapes。

本项目针对 M=1～8：

- 不承担 `m16n8k16` 的 M padding；
- 不需要为极瘦矩阵构造复杂 GEMM pipeline；
- warp-per-output GEMV 映射直接匹配 row-major `[N,K]` weight；
- 对 B=1～8 使用静态模板，让 accumulator 固定进入寄存器。

对于当前明确的应用场景，这种专用性是优势，而不是功能缺失的偶然结果。

### 5.2 支持 RMSNorm elementwise affine

文章的 Lazy Pre-Norm 明确指出 elementwise affine 难以支持，因为 `gamma[k]` 是
column-wise scaling，不能像 `rstd[b]` 一样简单移到 GEMM 输出端。

本项目利用固定权重 inference 的额外条件：

```text
fused_weight[n, k] = linear_weight[n, k] * gamma[k]
```

这样既保留 RMSNorm affine，又不在 forward 增加 column-wise multiplication。

这是当前实现相对文章 Lazy Pre-Norm 最明确的算法优势之一。但它依赖 Linear weight
固定，不能直接推广到训练。

### 5.3 Square-sum 只计算一次

文章 Lazy Pre-Norm 中，同一 M tile、不同 N tile 的 CTA 会重复累计相同 square-sum。

本项目只由第一个 CTA 计算：

```text
square_sum computation count = 1 per forward
```

对 N=4096、每 CTA 负责8个 output channels 的情况，可以避免数百个 CTA 的重复
reduction。

代价是需要第二次 kernel launch；这项权衡在极小 batch 下可能优于重复 reduction，
但必须由实际 benchmark 验证。

### 5.4 平方和保持 FP32

当前 square-sum 从乘法到最终统计值均为 FP32：

```text
float(x) * float(x) → FP32 partial sum → FP32 square_sum
```

它不会像项目早期复用 `fused_square_linear` 的方案那样，把统计结果先压回 FP16。
因此具有更好的精度和 overflow 安全性。

### 5.5 不依赖 Blackwell 特性

当前实现只使用 Ampere 已有的：

- CUDA Core FP32 FMA；
- FP16/half2 load；
- warp shuffle；
- 少量 shared memory；
- 普通 CUDA stream ordering。

不依赖：

- CTA cluster；
- DSMEM；
- TMEM；
- tcgen05；
- TMA_REDUCE_ADD；
- Blackwell warp-specialized GEMM skeleton。

因此可直接运行在 RTX 3090 和 A100 上。

### 5.6 实现与调试复杂度较低

当前核心 kernel 只有：

- warp-per-output projection；
- block0 square reduction；
- 单独 scaling epilogue。

相比 Multi-CTA Norm 或 FlashNormAttention，无需处理 cluster launch、DSMEM 生命周期、
TMEM buffer reuse、多组 specialized warps 和复杂 pipeline dependency，更容易验证非法
访存、数值误差和资源使用。

## 6. 当前实现相对文章方案的劣势

### 6.1 不是单 kernel fusion

文章 Lazy Pre-Norm 在同一个 GEMM kernel 内完成：

```text
GEMM + square-sum + rstd + output scaling
```

当前项目使用两个 kernel：

```text
projection + square-sum
RMS scaling epilogue
```

因此仍有：

- 第二次 CUDA launch；
- 临时 FP32 square-sum tensor 分配；
- numerator 写入 HBM；
- scaling kernel 从 HBM 重新读取并写回 numerator。

文章方案可在 accumulator 仍位于寄存器/TMEM 时直接 scaling，避免这部分 HBM round
trip。

### 6.2 无法把 normalization 与 Tensor Core 工作重叠

文章的核心目标是让 CUDA Core normalization 与 Tensor Core GEMM/attention overlap。

当前 projection 本身就在 CUDA Core 上运行，square-sum 与 projection 会竞争：

- FP32 FMA pipeline；
- load/store pipeline；
- instruction issue bandwidth；
- register resources。

虽然只有第一个 CTA 执行 square-sum，但不存在“用 Tensor Core 空档隐藏 CUDA Core
norm”的异构执行优势。

### 6.3 大 batch 和高算术强度场景扩展性差

SIMT kernel 每个 output warp 为每个 batch 保存独立 accumulator。随着 Batch 增大：

- accumulator 寄存器线性增加；
- 每个 weight pair 后的 FMA 数量线性增加；
- x load 指令增加；
- warp shuffle reduction 数量增加；
- CUDA Core computation 逐渐成为瓶颈。

项目此前 B=16 测试已经显示 SIMT 路径明显落后于 PyTorch Tensor Core GEMM。因此本
实现只适合小 batch，无法替代文章面向大 M 的 Tensor Core fusion。

### 6.4 不支持训练和 backward

文章不仅讨论 forward，还给出：

- Multi-CTA norm backward；
- forward/backward fusion regrouping；
- FlashNormAttention backward recomputation；
- mean/variance/rstd 等小型 forward state 的保存策略。

当前层：

- fused weight 是初始化时生成的 buffer；
- 不保留源 RMSNorm/Linear parameter；
- 没有 backward kernel；
- 不适合权重持续更新。

因此功能范围远窄于文章的训练系统方案。

### 6.5 只覆盖 RMSNorm pre-norm + Linear

文章覆盖：

- RMSNorm；
- LayerNorm；
- pre-norm；
- post-norm；
- GEMM；
- attention；
- residual；
- forward/backward。

当前项目仅覆盖：

```text
RMSNorm(x) → bias-free Linear
```

而且只支持 `[Batch,1,K]`。

### 6.6 dtype 和硬件范围更窄

文章实验使用 BF16/B200，技术还涉及 Blackwell cluster、TMEM 和 TMA。

当前实现：

- 只支持 FP16；
- 只允许 compute capability 8.x；
- 没有 BF16 specialization；
- 没有 Hopper/Blackwell 路径；
- 没有跨架构 autotuning。

### 6.7 权重预融合引入额外舍入

当前计算：

```text
FP16 fused_weight = FP16(linear_weight * rmsnorm_weight)
```

因此 affine scale 在初始化时发生一次 FP16 舍入。文章 benchmark 通常关闭 affine，
Lazy Pre-Norm 本身没有这一步。

当前实现相对原生 `RMSNorm → Linear` 的误差来源包括：

- fused weight 的 FP16 舍入；
- projection reduction 顺序不同；
- numerator 在 scaling 前存为 FP16。

### 6.8 Scaling epilogue 仍有局部低效

当前 scaling kernel 中，每个输出线程都会为同一 batch 重复执行：

```text
rsqrtf(square_sum[b] / K + eps)
```

例如 B=1、N=4096 时会执行4096次相同的 `rsqrtf`。文章的单-kernel方案通常按 row/tile
计算一次 rstd，并在寄存器或 shared/TMEM 中复用。

这是当前实现中最直接、低风险的后续优化点之一。

### 6.9 缺少系统级 fusion

文章强调 megakernel 的价值不仅是减少 launch，而是减少整个模块的 HBM transfer。
FlashNormAttention 会联合消除 norm、attention、residual 之间的多次 materialization。

当前实现只融合一个 RMSNorm 和一个 Linear，无法消除更大模型 block 中的其他中间
tensor IO。

## 7. 对比表

| 维度 | 当前项目 | PyTorch 文章方案 |
|---|---|---|
| 核心恒等式 | 延迟 row-wise rstd | Lazy Pre-Norm，同一恒等式 |
| 主要场景 | 小 batch decode inference | 大规模 training GEMM/attention |
| 典型 M | 1～8 | 大 M，文章示例可达约256K |
| GPU | RTX 3090、A100 | B200 |
| dtype | FP16 | BF16 benchmark |
| Projection | CUDA Core SIMT GEMV | Tensor Core GEMM/attention |
| Square-sum | 仅第一个 CTA计算一次 | Lazy Pre-Norm 中不同 N CTA重复计算 |
| Norm overlap | 与同 CTA projection 部分并存 | 通过 warp specialization 与 Tensor Core重叠 |
| Scaling | 第二个 CUDA kernel | 同 kernel epilogue |
| Numerator HBM 中间值 | 有 | Lazy Pre-Norm 无 |
| RMS affine | 通过静态权重预融合支持 | Lazy Pre-Norm 明确较难支持 |
| LayerNorm | 不支持 | Multi-CTA/attention方案支持 |
| Post-norm | 不支持 | Multi-CTA Norm支持 |
| Attention fusion | 不支持 | FlashNormAttention支持 |
| Backward | 不支持 | 提供训练融合策略 |
| CTA cluster/DSMEM | 不使用 | Multi-CTA核心机制 |
| TMEM/TMA | 不使用 | Blackwell优化中使用 |
| Autotuning | 无 | Helion全面 autotuning，TLX手工调优 |
| 实现复杂度 | 较低 | 高，尤其multi-CTA与attention |
| 可移植到Ampere | 当前原生目标 | Blackwell特性部分不可直接移植 |

## 8. 公平解读性能结果

不能把文章中的“隐藏90% normalization latency”或“FlashNormAttention最高35%
speedup”直接与本项目的 speedup 数字比较，原因包括：

1. GPU 不同：B200 与 RTX 3090/A100。
2. dtype 不同：BF16 与 FP16。
3. workload 不同：大规模训练与单 token decode。
4. baseline 不同：文章部分使用 TorchInductor baseline，本项目使用 eager
   `nn.RMSNorm + nn.Linear`。
5. 指标不同：文章有时按被消除 norm latency 的百分比计算，而本项目直接报告端到端
   layer latency ratio。
6. 融合边界不同：单 GEMM、post-norm、attention megakernel 与单个 pre-norm Linear。

有意义的比较应在同一 GPU、dtype、shape、精度容限和 baseline 下重新实现并测试两种
策略：

- square-sum 每 CTA重复 + 单-kernel scaling；
- square-sum 单次计算 + 两-kernel scaling。

对于 Ampere B=1、K=N=4096，前者会产生约 `ceil(N/8)` 次重复 square reduction，
后者会产生一次额外 launch 和一次 numerator HBM round trip。哪一个更快不能仅凭
理论断言，需要通过 kernel benchmark 和 Nsight Compute 决定。

## 9. 可以借鉴到当前项目的优化

### 9.1 增加“真正 Lazy Pre-Norm”实验路径

可以为 B=1 增加单-kernel variant：

- 每个 projection CTA 自己累计 square-sum；
- projection accumulator 保留在寄存器；
- K-loop 结束后计算 rstd；
- 写回前直接 scaling。

该路径会重复 square-sum，但消除：

- 临时 square-sum allocation；
- 第二次 launch；
- numerator 的额外 HBM read/write。

应与当前“两 kernel、square-sum只算一次”的实现做 A/B benchmark，而不是直接替换。

### 9.2 Warp specialization 的适用性评估

文章通过独立 warp partition 让 norm CUDA Core 工作与 Tensor Core工作重叠。当前
SIMT projection 没有异构 pipe 可隐藏 norm，但仍可评估：

- 部分 warp 预取 weight/x；
- 部分 warp 做 projection；
- 一个 warp 做 square reduction；

不过对单 token GEMV，额外同步和 producer/consumer pipeline 可能超过收益，必须关注
固定 launch 和 barrier 成本。

### 9.3 优化 scaling epilogue

无论是否采用单 kernel，都应首先尝试：

- 每个 scale CTA 只计算一次 rstd；
- shared memory 广播 rstd；
- 使用 `half2` 对 numerator 做向量化 load/store；
- 对 B=1 将 batch division 从内循环消除；
- 复用或避免临时 square-sum allocator。

这比引入 CTA cluster 风险更低，更符合当前 Ampere小 batch场景。

### 9.4 精度与 recomputation 思路

文章在 backward 中通过保存很小的 reduction state、重算较大的 norm activation 来减少
IO。虽然当前不支持 backward，但类似原则仍适用：

- 只保存 FP32 square-sum/rstd 等小统计量；
- 不 materialize normalized x；
- 如果未来支持 backward，优先重算 normalized x，而不是 forward 保存完整 tensor。

### 9.5 不应直接照搬的技术

下列文章技术依赖 Blackwell 或大训练 workload，不适合直接加入当前 Ampere B=1路径：

- CTA cluster + DSMEM；
- TMEM 和 tcgen05 accumulate；
- TMA_REDUCE_ADD；
- 为大 M persistent GEMM 设计的2D swizzling；
- FlashNormAttention 级别的多模块 megakernel；
- 12个以上 specialized warps 的复杂 pipeline。

直接照搬可能增加寄存器、shared memory、barrier 和 launch约束，重复此前双缓冲版本
因资源压力反而变慢的问题。

## 10. 建议的后续路线

按照收益、风险和与当前场景的匹配程度，建议：

1. 优化现有 scaling kernel：一次 rstd/CTA + `half2` scaling。
2. 避免或复用 square-sum 临时 allocation。
3. 实现单-kernel、每 CTA重复 square-sum 的 Lazy Pre-Norm A/B variant。
4. 在 B=1、2、4、8 和 K/N=4096 上比较两种 reduction 协作策略。
5. 使用 Nsight Compute 判断瓶颈是 HBM、FP32 FMA、SFU、launch 还是 occupancy。
6. 只有在目标扩展到较大 batch 时，再恢复 Tensor Core GEMM fusion 研究。
7. 只有在目标硬件扩展到 Hopper/Blackwell 后，再考虑 cluster/DSMEM/TMA/TMEM。
8. 如果未来支持训练，重新设计 parameter ownership、fused-weight更新和 backward
   regrouping，而不是沿用当前固定 buffer。

## 11. 最终判断

当前实现不是 PyTorch 文章方案的简化复刻，而是同一数学思想在另一种性能区间内的
专用变体：

- 文章用大 M Tensor Core kernel 的计算余量隐藏 normalization；
- 本项目用小 M SIMT GEMV 避免 Tensor Core padding，并通过静态权重融合支持 affine；
- 文章接受跨 N CTA 的 square-sum重复计算，以换取单 kernel 和寄存器内 scaling；
- 本项目只计算一次 square-sum，以换取第二个 scaling kernel。

在当前 B≤8、SeqLen=1、Ampere inference 场景，本项目的专用性、affine支持和低实现
复杂度具有实际优势。但从融合完整度、HBM IO、训练支持、硬件流水、autotuning 和
适用范围看，文章方案明显更成熟、更全面。

最值得借鉴的不是直接移植 Blackwell multi-CTA 技术，而是对 Lazy Pre-Norm 的两种
权衡做实证比较：

```text
重复 reduction + 单 kernel + register scaling
vs.
单次 reduction + 两 kernel + HBM scaling
```

这将直接回答当前实现还能否通过消除第二次 launch 和 numerator HBM round trip 获得
进一步收益。
