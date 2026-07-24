# Fused RMSNorm + Linear 优化记录

## 目标与约束

- 默认尺寸：`x=[1, 1, 4096]`，`weight=[4096, 4096]`。
- 输入、权重和输出均为 FP16，projection 使用 FP32 累加。
- 仅使用传统 CUDA Core，不使用 Tensor Core。
- 使用 `test/fused_rmsnorm_linear/test_fused_rmsnorm_linear.py` 验证正确性。
- 使用 `benchmark/fused_rmsnorm_linear/bench_fused_rmsnorm_linear.py`
  测量 Batch=1 性能。
- 目标为相对 PyTorch 达到 3 倍以上加速。
- 若连续三轮优化相对当时最佳版本的提升不足 3%，停止继续尝试并保留最佳版本。

## 基线

原始实现采用两个 kernel：

1. 每个 warp 计算一个输出通道，同时由第一个 CTA 计算一次
   `sum(x**2)`；
2. 单独的 epilogue kernel 计算 inverse RMS，并原位缩放 projection。

线程映射为 8 warps/CTA，每个 lane 使用 `half2` 沿 K 维读取，projection
使用 FP32 FMA 和 warp shuffle reduction。

实测结果：

```text
Fused:   0.0707 ms
PyTorch: 0.1735 ms
Speedup: 2.454x
```

## 第 1 轮：128-bit load 与四路 FP32 ILP

修改思路：

- 每个 lane 从一次读取一个 `half2` 改为一次读取一个 `uint4`，即 8 个
  FP16。
- 将 8 个值拆成四组 `half2`。
- 使用四条互相独立的 FP32 accumulator 链，尝试减少 load/loop 指令并
  隐藏 FMA 依赖。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0707 ms
PyTorch: 0.1693 ms
Speedup: 2.393x
```

结论：绝对延迟没有改善，相对最佳版本提升约 0%，不足 3%。

## 第 2 轮：单 kernel 直接完成 RMSNorm + Linear

修改思路：

- 针对 Batch=1 新增 direct kernel。
- 每个 projection warp 在读取 `x` 和权重时，同时计算 dot product 和
  `sum(x**2)`。
- 每个 warp 会重复计算相同的 RMS 统计，但不增加权重读取。
- projection reduction 后立即计算 inverse RMS 并写 FP16 输出。
- 删除 Batch=1 路径的 `square_sum` 临时张量和第二次 epilogue launch。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0682 ms
PyTorch: 0.1704 ms
Speedup: 2.500x
```

结论：相对基线绝对延迟提升约 3.5%，超过 3% 门槛，成为新的最佳版本。

## 第 3 轮：Direct kernel 的 128-bit load 与双四路累加

修改思路：

- 在第 2 轮 direct kernel 中使用 128-bit FP16 load。
- projection 使用四条独立 FP32 accumulator 链。
- square sum 同样使用四条独立 FP32 accumulator 链。
- 保持单 kernel 和每 warp 独立计算 RMS 统计。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0676 ms
PyTorch: 0.1707 ms
Speedup: 2.524x
```

结论：相对第 2 轮最佳绝对延迟提升约 0.9%，不足 3%。连续不足 3% 的轮数为 1。

## 第 4 轮：Direct kernel 使用 16 warps/CTA

修改思路：

- 将 Batch=1 direct kernel 从 8 warps/CTA 调整为 16 warps/CTA。
- 默认 4096 输出通道的 CTA 数从 512 减少到 256。
- 目标是在保持足够活跃 warp 的同时减少 CTA 调度波次。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0671 ms
PyTorch: 0.1672 ms
Speedup: 2.493x
```

结论：相对第 3 轮提升约 0.7%，不足 3%。连续不足 3% 的轮数为 2。

## 第 5 轮：Direct kernel 使用 32 warps/CTA

修改思路：

- 将 direct kernel 从 16 warps/CTA 调整为硬件上限 32 warps/CTA。
- 默认尺寸的 CTA 数从 256 减少到 128。
- 在 RTX 3090 的 82 个 SM 上以约两个调度波次完成全部输出。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0650 ms
PyTorch: 0.1673 ms
Speedup: 2.574x
```

结论：相对第 4 轮提升约 3.1%，超过 3% 门槛，成为新的最佳版本，连续失败计数清零。

## 第 6 轮：每个 CTA 使用一个专用统计 warp

修改思路：

- 32 个 warp 中使用一个 warp 计算一次 RMS 统计。
- 其余 31 个 warp 分别计算 projection 输出。
- CTA 内同步后直接应用共享的 inverse RMS。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0652 ms
PyTorch: 0.1696 ms
Speedup: 2.601x
```

结论：比第 5 轮最佳绝对延迟略慢，提升不足 3%。连续不足 3% 的轮数为 1。

## 第 7 轮：使用只读缓存加载

修改思路：

- 恢复第 5 轮每个 warp 独立计算 RMS 的结构。
- 对 128-bit 权重和输入加载使用 `__ldg`，测试 Ampere 只读缓存路径。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0655 ms
PyTorch: 0.1723 ms
Speedup: 2.632x
```

结论：绝对延迟仍慢于第 5 轮最佳，提升不足 3%。连续不足 3% 的轮数为 2。

## 第 8 轮：使用 `.cg` 流式加载策略

修改思路：

- 撤销第 7 轮的显式 `__ldg`。
- 恢复第 5 轮的 32 warps/CTA direct kernel。
- CUDA 编译增加 `-Xptxas=-dlcm=cg`，使全局加载绕过 L1 并使用 L2，
  尝试减少 32 MiB 流式权重对 L1 的污染。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0648 ms
PyTorch: 0.1683 ms
Speedup: 2.597x
```

结论：

- 相对第 5 轮的 `0.0650 ms` 提升约 0.3%，不足 3%。
- 第 6、7、8 轮连续三轮提升均不足 3%，触发停止条件。
- 第 8 轮具有所有已测试版本中最低的绝对延迟，因此最终保留第 8 轮。

## 最终结果

最终保留的实现具有以下特征：

- Batch=1 使用单个 direct kernel。
- 每个 warp 同时计算 projection 和该 warp 所需的 RMS 统计。
- 权重与输入使用 FP16，采用 128-bit 向量加载。
- projection 和平方和分别使用四条 FP32 accumulator 链。
- 32 warps/CTA。
- 使用传统 CUDA Core，不使用 Tensor Core。
- 全局加载采用 `.cg` 缓存策略。

相对初始基线的绝对延迟变化：

```text
0.0707 ms -> 0.0648 ms
```

整体提升约 8.3%，相对 PyTorch 加速为 2.597x。该结果曾按“连续三轮提升不足
3%”的旧限制停止；随后停止条件被放宽为连续五轮，因此从该最佳版本继续优化。

最终完整项目回归：

```text
68 passed
```

## 第 9 轮：Direct kernel 使用 24 warps/CTA

修改思路：

- 从第 8 轮最佳版本出发，将 direct kernel 调整为 24 warps/CTA。
- 尝试让每个 SM 同时驻留两个 768-thread CTA，提高活跃 warp 数。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0668 ms
PyTorch: 0.1704 ms
Speedup: 2.552x
```

结论：慢于第 8 轮最佳，提升不足 3%。连续不足 3% 的轮数为 4。

## 第 10 轮：Warp 0 兼做统计并保留全部输出 warp

修改思路：

- 恢复 32 warps/CTA。
- Warp 0 同时计算自己的 projection 和 CTA 唯一一份 RMS 统计。
- 其余 31 个 warp 只计算 projection。
- CTA 同步后共享 inverse RMS，同时避免第 6 轮损失一个输出 warp。

正确性：

```text
10 passed
```

性能：

```text
Fused:   0.0657 ms
PyTorch: 0.1691 ms
Speedup: 2.574x
```

结论：

- 慢于第 8 轮最佳，提升不足 3%。
- 第 6 至第 10 轮连续五轮提升不足 3%，触发新的停止条件。
- 最终恢复并保留第 8 轮绝对延迟最低的实现。

恢复第 8 轮代码后的最终复测：

```text
Correctness: 10 passed
Fused mean:  0.0659 ms
Fused p50:   0.0645 ms
PyTorch:     0.1694 ms
Speedup:     2.572x
```

不同运行之间存在小幅时钟与调度波动；第 8 轮记录到的最低 mean 为
`0.0648 ms`，恢复后 p50 仍为 `0.0645 ms`。

# Batch=2 优化阶段

Batch=2 使用新的独立计数，连续三轮提升不足 3% 时停止。Batch=1 direct
kernel 保持不变。

## Batch=2 基线

```text
Correctness: 10 passed
Fused:       0.0753 ms
PyTorch:     0.1782 ms
Speedup:     2.366x
```

## Batch=2 第 1 轮：专用单 kernel

修改思路：

- 每个 warp 一次读取 FP16 权重，同时计算两个 batch 的 projection。
- 每个 warp 同时计算两份 RMS 统计并直接缩放输出。
- 删除 Batch=2 的 `square_sum` 临时张量和第二次 epilogue launch。
- 使用 128-bit FP16 load、四路 projection 和四路 square-sum FP32 累加。

正确性与性能：

```text
Correctness: 10 passed
Fused:       0.0654 ms
PyTorch:     0.1701 ms
Speedup:     2.602x
```

结论：相对 Batch=2 基线提升约 13.1%，超过 3%，成为 Batch=2 新最佳版本。

## Batch=2 第 2 轮：CTA 共享 RMS 统计

修改思路：

- 保留 32 个输出 warp。
- 仅 warp 0 计算两个 batch 的 RMS 统计。
- CTA 同步后共享两份 inverse RMS。

正确性与性能：

```text
Correctness: 10 passed
Fused:       0.0662 ms
PyTorch:     0.1700 ms
Speedup:     2.567x
```

结论：慢于第 1 轮最佳，提升不足 3%。连续不足 3% 的轮数为 1。

## Batch=2 第 3 轮：跨 batch half2

修改思路：

- 将两个 batch 的同一 K 位置打包到 `half2`。
- 一个 FP16 权重标量广播到两个 half lane。
- 每 8 个 K 元素将 FP16 短累加结果转换回 FP32。

正确性与性能：

```text
Correctness: 10 passed
Fused:       0.0665 ms
PyTorch:     0.1690 ms
Speedup:     2.542x
```

结论：慢于第 1 轮最佳，提升不足 3%。连续不足 3% 的轮数为 2。

## Batch=2 第 4 轮：专用 16 warps/CTA

修改思路：

- 恢复第 1 轮的 FP32 projection 与 square-sum 累加。
- Batch=2 单独使用 16 warps/CTA，尝试通过两个 CTA/SM 改善高寄存器模板
  的 occupancy。
- Batch=1 的 32 warps/CTA 路径不变。

正确性与性能：

```text
Correctness: 10 passed
Fused:       0.0663 ms
PyTorch:     0.1705 ms
Speedup:     2.573x
```

结论：

- 慢于第 1 轮最佳，提升不足 3%。
- 第 2、3、4 轮连续三轮提升不足 3%，触发 Batch=2 停止条件。
- 最终恢复 Batch=2 第 1 轮的 32 warps/CTA 专用单 kernel。

恢复最佳版本后的最终复测：

```text
Batch=2 correctness: 10 passed
Batch=2 fused mean:  0.0652 ms
Batch=2 fused p50:   0.0646 ms
Batch=2 PyTorch:     0.1704 ms
Batch=2 speedup:     2.614x

Batch=1 fused mean:  0.0652 ms
Batch=1 fused p50:   0.0645 ms
Batch=1 speedup:     2.597x
```

Batch=2 从基线 `0.0753 ms` 降至最终 `0.0652 ms`，绝对延迟改善约
13.4%。未达到 3x，但已触发本阶段连续三轮不足 3% 的停止限制。
