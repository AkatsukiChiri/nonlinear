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

# Fused Add + RMSNorm + Linear：Batch=1 优化阶段

本阶段使用
`benchmark/fused_rmsnorm_linear/bench_fused_add_rms_linear.py --batch 1
--iterations 100`。停止规则为：以三次修改窗口起点的最佳延迟为基准，三次修改
完成后的累计提升不足 5% 时停止，并恢复窗口内最佳版本。

## 基线

```text
Correctness max abs error: 0.000976562
Fused mean:               0.0717 ms
Fused p50:                0.0695 ms
PyTorch mean:             0.1805 ms
Speedup:                  2.518x
```

当前 Batch=1 使用 32 warps/CTA。每个输出 warp 同时计算 projection 和
`sum((x + x_res)^2)`，并由 block 0 的 warp 0 写回 FP16 residual。

## 第 1 轮：16 warps/CTA

修改思路：

- 将 Batch=1 从 32 warps/CTA 改成 16 warps/CTA。
- 尝试允许每个 SM 同时驻留两个 CTA，并减小最后一波 CTA 不均衡。

结果：

```text
Fused mean:   0.0770 ms
Fused p50:    0.0755 ms
PyTorch mean: 0.1825 ms
Speedup:      2.370x
```

相对基线退化约 7.4%。说明增加 CTA 数量和重复输入工作造成的代价大于驻留率收益。

## 第 2 轮：CTA 共享 inverse RMS

修改思路：

- 恢复 32 warps/CTA。
- 每个 CTA 仅由 warp 0 计算一次 residual平方和。
- 通过 shared memory向该CTA的32个 projection warp广播 inverse RMS。
- projection warp不再维护平方和累加流。

结果：

```text
Fused mean:   0.0750 ms
Fused p50:    0.0727 ms
PyTorch mean: 0.1772 ms
Speedup:      2.361x
```

相对基线退化约 4.6%。CTA同步等待和 projection累加链的 ILP下降抵消了减少平方
计算的收益。

## 第 3 轮：CTA 共享 FP16 residual

修改思路：

- 在第2轮基础上，由 warp 0将 `x + x_res` 写入动态 shared memory。
- projection warp复用 shared residual，避免重复读取两个输入并执行 FP16 add。
- block 0 同时将同一 residual写回最终输出张量。

结果：

```text
Fused mean:   0.0782 ms
Fused p50:    0.0735 ms
PyTorch mean: 0.1815 ms
Speedup:      2.321x
```

相对基线退化约 9.1%。shared memory阶段和 CTA barrier进一步延长了权重流式读取
开始前的等待。

## 停止结论

三次修改完成后，相对窗口起点 `0.0717 ms` 的累计提升为 0%，不足 5%，触发停止
规则。最终恢复原始 32 warps/CTA 单 kernel版本。该窗口未达到 3x目标，最佳实测
仍为基线的 2.518x。

恢复最佳版本后的最终复测：

```text
Correctness:               10 passed
Correctness max abs error: 0.000976562
Fused mean:                0.0721 ms
Fused p50:                 0.0696 ms
PyTorch mean:              0.1811 ms
Speedup:                   2.511x
```

# Fused Add Batch=1：激进优化重置窗口

用户重置停止计数后，以相同默认形状和 200 次迭代重新建立窗口。

## 新基线

```text
Fused mean:   0.0703 ms
Fused p50:    0.0694 ms
PyTorch mean: 0.1807 ms
Speedup:      2.571x
```

## 第 1 轮：packed FP16乘积、FP32累加

使用 CUDA Core `half2` 同时计算两个 projection乘积和平方，再将乘积转换成
FP32累加，减少循环中的 FP16解包和标量FMA指令。

首次结果为 `0.0700 ms`，但最终重复复测为：

```text
Fused mean:   0.0714 ms
Fused p50:    0.0707 ms
PyTorch mean: 0.1815 ms
Speedup:      2.542x
```

正确性通过，最大绝对误差 `0.000976562`。重复结果表明首次小幅改善属于测量
波动，因此未保留。

## 第 2 轮：局部 half2 FMA累加

每个 lane的短局部点积先以 `half2 FMA` 累加，warp归约前再转换成 FP32，进一步
减少转换指令。

```text
Correctness:  10 passed
Max abs error: 0.00195312
Fused mean:   0.0708 ms
Fused p50:    0.0695 ms
PyTorch mean: 0.1799 ms
Speedup:      2.543x
```

性能未改善且误差扩大到 `0.00195312`，未保留。

## 第 3 轮：Batch=1专用两阶段路径

第一 kernel只计算并写回 FP16 residual和 inverse RMS；第二 kernel执行不含平方和
的纯 projection。目标是彻底删除每个输出 warp重复的 add和平方计算。

```text
Correctness:  10 passed
Max abs error: 0.000976562
Fused mean:   0.0786 ms
Fused p50:    0.0777 ms
PyTorch mean: 0.1801 ms
Speedup:      2.291x
```

额外 launch和4字节临时张量分配的代价超过删除冗余计算的收益。

## 本窗口结论

三次修改相对新基线的累计稳定提升不足 5%，再次触发停止条件。最终恢复新基线的
32 warps/CTA、单 kernel、FP32 FMA实现。恢复后的验证为 `10 passed`，均值
`0.0710 ms`、p50 `0.0695 ms`；未达到 3x目标。

# Fused Add：Batch=2/4/8 分路径优化

默认形状均为 `K=N=4096`、FP16、RTX 3090，每组200次计时。

## 优化前基线

```text
Batch 2: 0.0716 ms，Add + Fused RMS 0.0764 ms
Batch 4: 0.0917 ms，Add + Fused RMS 0.0810 ms
Batch 8: 0.1330 ms，Add + Fused RMS 0.1000 ms
```

原direct模板的寄存器数据：

```text
Batch 4:  75 registers/thread
Batch 8: 144 registers/thread
```

B8同时保存四路projection和四路平方和的8个batch累加器，导致occupancy过低。

## Batch 4：projection/statistics + scale

采用与原 `fused_rmsnorm_linear` B4模板相同的两阶段结构：

- 权重只读取一次并跨4个batch复用。
- 仅block 0计算FP16 residual和平方和。
- projection warp不再保存平方和累加流。
- 第二个轻量kernel完成inverse RMS缩放。

结果约为 `0.079～0.081 ms`，相对原0.0917 ms提升约11%～14%，并达到或略快于
外部add加fused-rms路径。

## Batch 8：128-bit add + 原B8 fused-rms模板

实验表明在projection warp内即时add无法同时满足低寄存器与低指令数：

- 去掉重复平方和后约 `0.1237 ms`。
- 两路累加降至64寄存器后约 `0.1120 ms`。
- pair级half2流式load虽降寄存器，但退化到 `0.1826 ms`。
- 先add+inverse RMS再直接projection约 `0.1093 ms`，前置单CTA统计无法与
  projection重叠。

最终采用B8专用三阶段：

1. 128-bit FP16 residual add。
2. 直接调用原B8 `fused_rmsnorm_linear` projection/statistics模板。
3. scale kernel。

最终约 `0.099～0.100 ms`，相对原0.1330 ms提升约25%，与外部
`Add + FusedRMSNormLinear`持平或略快，同时保留单一融合层接口和内部buffer管理。

## Batch 2

B2继续使用32 warps/CTA direct单kernel，约 `0.072 ms`，稳定快于外部add加
fused-rms路径，未作结构性修改。
