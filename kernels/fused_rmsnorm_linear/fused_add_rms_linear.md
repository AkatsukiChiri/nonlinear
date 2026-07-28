# Fused Add + RMSNorm + Linear

## 计算语义

该推理 kernel 接收两个连续 FP16 张量：

```text
x:       [Batch, 1, HiddenDim]
x_res:   [Batch, 1, HiddenDim]
```

并返回：

```text
linear_output: [Batch, 1, OutDim]
residual_sum:  [Batch, 1, HiddenDim]
```

对应计算为：

\[
z = \operatorname{FP16}(x + x_{\mathrm{res}})
\]

\[
y = \frac{z(\gamma W)^\mathsf{T}}
{\sqrt{\operatorname{mean}(z^2)+\epsilon}} + b.
\]

`gamma * W` 在构造 `FusedAddRMSNormLinear` 时提前计算并以 FP16
`fused_weight` 保存。Linear 没有 bias 时不执行最后的加法。

## Kernel 设计

实现位于 `fused_add_rms_linear.cu`，与原有 RMSNorm + Linear kernel
相互独立。

- 使用 CUDA Core，不使用 Tensor Core。
- FP16 输入、权重、bias、residual 和输出，FP32 projection 与平方和累加。
- residual add 首先舍入到 FP16，与 PyTorch `x + x_res` 的结果一致。
- add 结果直接用于 projection 和平方和，不产生供 normalization 读取的中间张量。
- `blockIdx.x == 0 && warp == 0` 负责将 residual 写回一次；其他 warp
  只在寄存器中生成 add 结果，避免多个 warp 重复写同一地址。
- HiddenDim 是 8 的倍数时使用 128-bit load，并以四条独立累加流减少依赖链。
- Batch=1、2 使用 32 warps/CTA 的单 kernel direct 路径。
- Batch=4 使用两阶段专用路径：第一个 kernel 复用一次权重读取完成所有
  batch projection，仅首个 CTA 计算 residual 和 RMS统计；第二个 kernel
  完成缩放。
- Batch=8 直接复用原 `fused_rmsnorm_linear` 的高性能模板：先用128-bit
  residual-add kernel物化 FP16 sum，再执行原B8 projection/statistics和
  scale。这样避免B8 direct模板的144寄存器压力。
- 其他 Batch 使用编译期 Batch模板和8 warps/CTA。
- 同一个输出 warp 加载一次权重元素，并在寄存器中复用于该 warp负责的所有 batch。

## 测试和 benchmark

正确性测试：

```bash
conda run -n nonl --no-capture-output pytest \
  test/fused_rmsnorm_linear/test_fused_add_rms_linear.py
```

默认 LLM decode 形状 benchmark：

```bash
conda run -n nonl --no-capture-output python \
  benchmark/fused_rmsnorm_linear/bench_fused_add_rms_linear.py
```

benchmark 比较以下三条完整路径：

1. `FusedAddRMSNormLinear`
2. PyTorch add 加现有 `FusedRMSNormLinear`
3. PyTorch add、`nn.RMSNorm` 和 `nn.Linear`

所有路径都会返回或保留 FP16 residual，因此比较包含相同的必要输出语义。
