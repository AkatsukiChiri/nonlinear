# Fused LayerNorm + Linear CUDA 实现

## 目标

该实现优化固定权重推理中的 decode 阶段：

```text
x:      [Batch, 1, HiddenDim] FP16
weight: [OutDim, HiddenDim]   FP16
output: [Batch, 1, OutDim]    FP16
```

当前限制为：

- `SeqLen == 1`；
- `Batch` 在 1 到 8 之间；
- GPU 架构为 Ampere；
- LayerNorm 和 Linear 参数固定，不支持训练时更新；
- 输入、权重和输出为 FP16。

## 参数折叠

设 PyTorch Linear 权重为 $W\in\mathbb{R}^{O\times H}$，LayerNorm 参数为
$\gamma,\beta\in\mathbb{R}^{H}$。构造推理层时预计算：

$$
\widetilde W=W\odot\gamma,
$$

$$
c_1=\operatorname{sum}(\widetilde W,\dim=-1),
$$

$$
c_2=W\beta+b.
$$

存储格式为：

```text
fused_weight: [OutDim, HiddenDim] FP16
weight_sum:   [OutDim]            FP32
fused_bias:   [OutDim]            FP32
```

`weight_sum` 从最终量化到 FP16 的 `fused_weight` 计算，确保均值修正与
projection kernel 实际使用的权重一致。

## 两阶段计算

### 第一阶段：Projection 与统计量

第一个 kernel 同时计算：

$$
P=x\widetilde W^T,
$$

$$
s_1=\sum_k x_k,
$$

$$
s_2=\sum_k x_k^2.
$$

Projection 使用 FP32 accumulator，并写入 FP32 临时张量。LayerNorm 的均值修正
可能造成明显的数值消减，因此不在完成修正前将 projection 转换为 FP16。

线程布局参考 `fused_rmsnorm_linear`：

- 每个 CTA 包含 8 个 warp；
- 每个 warp 计算一个输出通道；
- HiddenDim 为偶数时使用 `half2` load；
- HiddenDim 为奇数时使用标量 FP16 load；
- warp shuffle 完成 K 维 FP32 reduction；
- 仅第一个 CTA 额外计算每个 batch 的 $s_1$ 和 $s_2$。

Batch 1 到 8 使用独立模板实例，使 batch 循环可以在编译期展开。

### 第二阶段：LayerNorm Epilogue

第二个 kernel 对每个 `[Batch, OutDim]` 元素计算：

$$
\mu=s_1/H,
$$

$$
\sigma^2=\max(s_2/H-\mu^2,0),
$$

$$
r=\operatorname{rsqrt}(\sigma^2+\epsilon),
$$

$$
y=(P-\mu c_1)r+c_2.
$$

上述运算均使用 FP32，最后一次写回时转换为 FP16。

## Python 推理层

`layers/fused_layernorm_linear/FusedLayerNormLinear` 接收冻结的
`torch.nn.LayerNorm` 和 `torch.nn.Linear`，并将三个预计算结果注册为 buffer。
原始模块不会保留在 fused layer 中，因此：

- buffer 会进入 `state_dict`；
- buffer 会随模块移动设备；
- `named_parameters()` 为空；
- 原始参数更新不会自动反映到 fused layer。

## 验证

正确性测试：

```bash
conda run -n nonl --no-capture-output \
  pytest -s -q test/fused_layernorm_linear/test_fused_layernorm_linear.py
```

默认 benchmark：

```bash
conda run -n nonl --no-capture-output \
  python benchmark/fused_layernorm_linear/bench_fused_layernorm_linear.py
```

默认尺寸为：

```text
x:      [1, 1, 4096]
weight: [4096, 4096]
```
