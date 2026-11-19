"""Compare decode RMSNorm + W4 projection across all Llama decoder blocks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

from benchmark.fused_rmsnorm_linear_w4a16_marlin.baseline import (
    make_native_marlin,
)
from layers.fused_rmsnorm_linear_w4a16_marlin.layer import FusedRMSNormLinearW4A16Marlin
from layers.fused_rmsnorm_linear_w4a16_marlin.packing import (
    _unpack_awq,
    quantize_weight_to_awq_groupwise,
)


@torch.inference_mode()
def main() -> None:
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_awq_marlin_linear
    from vllm.scalar_type import scalar_types

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    torch.manual_seed(42)

    def read(name: str) -> torch.Tensor:
        with safe_open(args.model / index[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).cuda()

    groups = (
        ("qkv", "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("gate_up", "post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
    )
    eps = config["rms_norm_eps"]
    count = args.blocks or config["num_hidden_layers"]
    with args.csv.open("x", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(("model", "block", "projection", "batch", "k", "n", "stripe_n",
                         "pytorch_us", "vllm_us", "fused_us", "cosine"))
        for block in range(count):
            for projection, norm_name, names in groups:
                prefix = f"model.layers.{block}."
                gamma = read(f"{prefix}{norm_name}.weight").half()
                weight = torch.cat([read(f"{prefix}{name}.weight") for name in names])
                qweight, qzeros, scales = quantize_weight_to_awq_groupwise(weight)
                del weight
                k, n = qweight.shape[0], scales.shape[1]
                dense = (
                    (_unpack_awq(qweight, n).view(-1, 128, n).half()
                     - _unpack_awq(qzeros, n).half().unsqueeze(1))
                    * scales.unsqueeze(1)
                ).view(k, n).T.contiguous()
                native = make_native_marlin(qweight, qzeros, scales, 128)
                stripe = 64
                fused = FusedRMSNormLinearW4A16Marlin(
                    qweight, qzeros, scales, gamma, eps, stripe_n=stripe
                )
                norm = RMSNorm(k, eps=eps, dtype=torch.float16).cuda()
                norm.weight.copy_(gamma)
                for batch in range(1, 9):
                    x = torch.randn(batch, k, device="cuda", dtype=torch.float16)
                    xf = x.unsqueeze(1)

                    def pytorch() -> torch.Tensor:
                        fp32 = x.float()
                        normalized = (fp32 * torch.rsqrt(fp32.square().mean(-1, True) + eps)).half()
                        return torch.nn.functional.linear(normalized * gamma, dense)

                    def vllm() -> torch.Tensor:
                        return apply_awq_marlin_linear(
                            input=norm(x), weight=native[0], weight_scale=native[1],
                            weight_zp=native[2], g_idx=native[3],
                            g_idx_sort_indices=native[4], workspace=native[5],
                            quant_type=scalar_types.uint4, output_size_per_partition=n,
                            input_size_per_partition=k,
                        )

                    def fused_forward() -> torch.Tensor:
                        return fused(xf).squeeze(1)

                    paths = (pytorch, vllm, fused_forward)
                    reference = pytorch()
                    cosine = min(
                        torch.nn.functional.cosine_similarity(
                            reference.float(), path().float(), dim=-1
                        ).min().item() for path in paths[1:]
                    )
                    if cosine < 0.999 or not torch.isfinite(reference).all().item():
                        raise RuntimeError(f"Numerical check failed for {block=} {projection=} {batch=}")
                    for path in paths:
                        for _ in range(10):
                            path()
                    torch.cuda.synchronize()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    timings: list[list[float]] = [[], [], []]
                    for sample in range(args.samples):
                        for offset in range(3):
                            idx = (sample + offset) % 3
                            start.record()
                            paths[idx]()
                            end.record()
                            end.synchronize()
                            timings[idx].append(start.elapsed_time(end) * 1000)
                    writer.writerow((args.model.name, block, projection, batch, k, n, stripe,
                                     *(statistics.median(t) for t in timings), cosine))
                report.flush()
                print(f"{args.model.name} block {block + 1}/{count} {projection} done", flush=True)


if __name__ == "__main__":
    main()
