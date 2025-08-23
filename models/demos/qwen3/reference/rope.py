# SPDX-FileCopyrightText: © 2023
# SPDX-License-Identifier: MIT

from typing import Tuple
import torch

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig


def precompute_freqs_cis(config: Qwen3MoeConfig):
    theta = config.rope_theta
    dim = config.head_dim
    max_seq_len = config.max_seq_len

    with torch.device("cpu"), torch.no_grad():
        indices = torch.div(torch.arange(start=0, end=dim, step=2, dtype=torch.int64).to(dtype=torch.float32)[: (dim // 2)], dim)
        freqs = torch.reciprocal(torch.pow(theta, indices)).to(dtype=torch.float32)
        t = torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.int64).to(dtype=torch.float32)
        freqs = torch.outer(t, freqs).to(dtype=torch.float32)
        freqs_cis = torch.polar(abs=torch.ones_like(input=freqs, dtype=torch.float32), angle=torch.neg(freqs))
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = xq.dtype

    if dtype == torch.bfloat16:
        xq = xq.to(dtype=torch.float16)
        xk = xk.to(dtype=torch.float16)

    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(torch.mul(xq_, freqs_cis)).flatten(3)
    xk_out = torch.view_as_real(torch.mul(xk_, freqs_cis)).flatten(3)
    return xq_out.to(dtype=dtype), xk_out.to(dtype=dtype)
