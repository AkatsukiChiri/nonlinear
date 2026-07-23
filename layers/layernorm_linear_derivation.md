# LayerNorm + Linear 的分解推导

## 1. 问题定义

设输入为

$$
X\in\mathbb{R}^{N\times H},
$$

其中 $N$ 表示展平后的 Batch 和 SeqLen，$H$ 表示 HiddenDim。Linear
的权重采用 PyTorch 语义，记为

$$
W\in\mathbb{R}^{H\times O},\qquad b\in\mathbb{R}^{O}.
$$

这里将每个 token 的输入写成行向量 $x\in\mathbb{R}^{1\times H}$。
LayerNorm 在长度为 $H$ 的最后一维上进行归一化。

## 2. 不带仿射参数的 LayerNorm

先考虑 `elementwise_affine=False` 的 LayerNorm：

$$
\operatorname{LN}(x)
=
\frac{x-\mu\mathbf{1}}{\sqrt{\sigma^2+\epsilon}},
$$

其中

$$
\mu=\frac{1}{H}\sum_{i=1}^{H}x_i,
$$

$$
m_2=\frac{1}{H}\sum_{i=1}^{H}x_i^2,
$$

$$
\sigma^2=m_2-\mu^2,
$$

$\mathbf{1}\in\mathbb{R}^{1\times H}$ 是全 1 行向量。

LayerNorm 后接 Linear 的结果为

$$
z=\operatorname{LN}(x)W+b.
$$

将 LayerNorm 展开：

$$
z
=
\frac{x-\mu\mathbf{1}}{\sqrt{\sigma^2+\epsilon}}W+b.
$$

由于同一个 token 的 $\mu$ 和 $\sigma^2$ 都是标量，可以利用矩阵乘法
的线性性质：

$$
\boxed{
z
=
\left(xW-\mu(\mathbf{1}W)\right)
\frac{1}{\sqrt{m_2-\mu^2+\epsilon}}
+b
}
$$

因此，第一阶段可以同时得到以下四个组件：

1. $xW$；
2. $\sum_i x_i$；
3. $\sum_i x_i^2$；
4. $\mathbf{1}W$。

第二阶段只需要标量归约结果和逐元素计算：

$$
\mu=\frac{\sum_i x_i}{H},
$$

$$
r=\operatorname{rsqrt}
\left(
\frac{\sum_i x_i^2}{H}-\mu^2+\epsilon
\right),
$$

$$
\boxed{z=(xW-\mu(\mathbf{1}W))\,r+b.}
$$

这里的 $r$ 对一个 token 是标量，在输出维度 $O$ 上广播。

对于整个输入 $X$，若

$$
s_1=\operatorname{sum}(X,\mathrm{dim}=-1,\mathrm{keepdim}=\mathrm{True}),
$$

$$
s_2=\operatorname{sum}(X^2,\mathrm{dim}=-1,\mathrm{keepdim}=\mathrm{True}),
$$

则可写成类似 PyTorch 的形式：

```python
mean = s1 / hidden_dim
variance = s2 / hidden_dim - mean.square()
inverse_std = torch.rsqrt(variance + eps)

output = (x @ weight - mean * (torch.ones_like(x) @ weight))
output = output * inverse_std + bias
```

注意，`torch.ones_like(x) @ weight` 的每一行都相同，等价于预先计算并
广播：

```python
weight_column_sum = weight.sum(dim=0)
output = (x @ weight - mean * weight_column_sum) * inverse_std + bias
```

因此没有必要真的构造与 $X$ 同形状的全 1 张量，也没有必要为每个
token 重复计算 $\mathbf{1}W$。

## 3. 带仿射参数的标准 LayerNorm

PyTorch 默认的 LayerNorm 还包含逐隐藏维度的缩放参数
$\gamma\in\mathbb{R}^{H}$ 和偏移参数
$\beta\in\mathbb{R}^{H}$：

$$
\operatorname{LN}_{\gamma,\beta}(x)
=
\left(
\frac{x-\mu\mathbf{1}}{\sqrt{\sigma^2+\epsilon}}
\right)\odot\gamma+\beta.
$$

令

$$
\widetilde{W}=\operatorname{diag}(\gamma)W,
$$

则

$$
\begin{aligned}
z
&=\operatorname{LN}_{\gamma,\beta}(x)W+b\\
&=
\frac{x\widetilde{W}-\mu(\mathbf{1}\widetilde{W})}
{\sqrt{m_2-\mu^2+\epsilon}}
+\beta W+b.
\end{aligned}
$$

最终得到

$$
\boxed{
z
=
\left(x\widetilde{W}-\mu(\mathbf{1}\widetilde{W})\right)r
+\beta W+b.
}
$$

因为

$$
\widetilde{W}_{i,j}=\gamma_iW_{i,j},
$$

所以不需要显式构造对角矩阵，只需按行缩放 Linear 权重：

```python
scaled_weight = gamma[:, None] * weight
scaled_weight_column_sum = scaled_weight.sum(dim=0)
constant = beta @ weight + bias

output = (
    x @ scaled_weight - mean * scaled_weight_column_sum
) * inverse_std + constant
```

若 $\gamma$ 不是常数，仅有原始的 $xW$ 和 $\mathbf{1}W$ 不足以恢复
带仿射参数的结果；此时必须使用
$x\operatorname{diag}(\gamma)W$ 和
$\mathbf{1}\operatorname{diag}(\gamma)W$，或者采用等价计算。

## 4. 两阶段计算总结

### 阶段一：GEMM 与统计量归约

对每个 token 同时计算：

$$
P=x\widetilde{W},\qquad
s_1=\sum_i x_i,\qquad
s_2=\sum_i x_i^2.
$$

以下仅依赖模型参数，可提前计算：

$$
c_1=\mathbf{1}\widetilde{W}
=\operatorname{sum}(\widetilde{W},\mathrm{dim}=0),
$$

$$
c_2=\beta W+b.
$$

不带 LayerNorm 仿射参数时，取
$\widetilde{W}=W$、$c_2=b$。

### 阶段二：逐 token、逐输出元素恢复结果

$$
\mu=s_1/H,
$$

$$
r=\operatorname{rsqrt}(s_2/H-\mu^2+\epsilon),
$$

$$
\boxed{z=(P-\mu c_1)r+c_2.}
$$

该阶段不再需要沿 HiddenDim 做 GEMM，只包含标量计算、广播和逐元素
运算。

## 5. 数值实现注意事项

- 对 FP16/BF16 输入，建议用 FP32 累加 $s_1$ 和 $s_2$，并用 FP32
  计算均值、方差和倒标准差。
- 公式 $E[x^2]-E[x]^2$ 可能因浮点舍入产生很小的负数。实现中可在
  加 $\epsilon$ 前对方差做 `clamp_min(0)`，但应确认这与目标框架的
  LayerNorm 数值语义一致。
- PyTorch LayerNorm 使用总体方差，即分母为 $H$，不是 $H-1$。
- 该分解在实数算术下严格成立；改变归约顺序以及融合 GEMM 后，浮点
  结果可能与 eager 实现存在小量误差。
- `Linear` 若无 bias，则令 $b=0$；LayerNorm 若无 $\beta$，则令
  $\beta=0$。
