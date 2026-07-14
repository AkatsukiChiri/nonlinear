import torch

from layers.fused_square_linear import fused_square_linear_fp16 

DEFAULT_EPS = 1e-6
DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 1
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_OUT_DIM = 4096

def fused_rmsnorm_linear(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    fused_weight = rmsnorm_weight * linear_weight
    vec_one = torch.ones(
        1, 
        fused_weight.shape[1], 
        dtype=fused_weight.dtype,
        device=fused_weight.device)
    fused_weight = torch.cat([fused_weight, vec_one], dim=0)
    
    fused_out = fused_square_linear_fp16(x, fused_weight)

    if x.shape[0]==1 and x.shape[1]==1:
        x_square_sum = fused_out[1, -1]
        variance = x_square_sum / x.shape[-1]
        x_rms = torch.sqrt(variance + eps)

        linear_output = fused_out[0,0:-1]

        return linear_output / x_rms

