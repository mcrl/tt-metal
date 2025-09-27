from typing import Tuple
import torch
import ttnn
from models.demos.qwen3.utils.profiler import profile_trace


def precompute_freqs_cis(config) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    theta = config.rope_theta
    dim = config.head_dim
    max_seq_len = config.max_seq_len

    with torch.device("cpu"), torch.no_grad():
        indices = torch.div(
            torch.arange(start=0, end=dim, step=2, dtype=torch.int64).to(dtype=torch.float32)[: (dim // 2)], dim
        )
        freqs = torch.reciprocal(torch.pow(theta, indices)).to(dtype=torch.float32)
        t = torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.int64).to(dtype=torch.float32)
        freqs = torch.outer(t, freqs).to(dtype=torch.float32)
        freqs_cis = torch.polar(abs=torch.ones_like(input=freqs, dtype=torch.float32), angle=torch.neg(freqs))

    return freqs_cis.real, freqs_cis.imag


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


@profile_trace("apply_rotary_emb", level=4)
def apply_rotary_emb(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cis: Tuple[ttnn.Tensor, ttnn.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    def rotate(x: ttnn.Tensor):
        batch_size, seq_len, num_heads, head_dim = x.shape

        cos, sin = freqs_cis

        cos = ttnn.reshape(
            ttnn.repeat(ttnn.reshape(cos, [1, seq_len, 1, head_dim // 2]), [batch_size, 1, num_heads, 1]),
            [batch_size, seq_len, num_heads, head_dim // 2, 1],
        )
        sin = ttnn.reshape(
            ttnn.repeat(ttnn.reshape(sin, [1, seq_len, 1, head_dim // 2]), [batch_size, 1, num_heads, 1]),
            [batch_size, seq_len, num_heads, head_dim // 2, 1],
        )

        x_ = ttnn.reshape(x, (batch_size, seq_len, num_heads, head_dim // 2, 2))

        even = ttnn.slice(x_, (0, 0, 0, 0, 0), (batch_size, seq_len, num_heads, head_dim // 2, 1))
        odd = ttnn.slice(x_, (0, 0, 0, 0, 1), (batch_size, seq_len, num_heads, head_dim // 2, 2))

        real = ttnn.subtract(ttnn.multiply(even, cos), ttnn.multiply(odd, sin))
        imag = ttnn.add(ttnn.multiply(odd, cos), ttnn.multiply(even, sin))

        y = ttnn.concat([real, imag], dim=-1)

        result = ttnn.reshape(y, (batch_size, seq_len, num_heads, head_dim))

        return result

    yq = ttnn.to_layout(rotate(xq), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    yk = ttnn.to_layout(rotate(xk), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    return yq, yk


def precompute_freqs_cis_v2(config) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    theta = config.rope_theta
    dim = config.head_dim
    max_seq_len = config.max_seq_len

    with torch.device("cpu"), torch.no_grad():
        indices = torch.div(
            torch.arange(start=0, end=dim, step=2, dtype=torch.int64).to(dtype=torch.float32)[: (dim // 2)], dim
        )
        freqs = torch.reciprocal(torch.pow(theta, indices)).to(dtype=torch.float32)
        t = torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.int64).to(dtype=torch.float32)
        freqs = torch.outer(t, freqs).to(dtype=torch.float32)
        freqs_cis = torch.polar(abs=torch.ones_like(input=freqs, dtype=torch.float32), angle=freqs)

        cos = freqs_cis.real
        sin = freqs_cis.imag

        cos_interleaved = torch.stack([cos, cos], dim=-1).flatten(start_dim=-2)
        sin_interleaved = torch.stack([sin, sin], dim=-1).flatten(start_dim=-2)

    return cos_interleaved, sin_interleaved


def apply_rotary_emb_v2(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cis: Tuple[ttnn.Tensor, ttnn.Tensor],
    trans_mat: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    batch_size, seq_len, num_heads, head_dim = xq.shape

    cos, sin = freqs_cis
    cos_full = ttnn.reshape(cos, [1, 1, seq_len, head_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
    sin_full = ttnn.reshape(sin, [1, 1, seq_len, head_dim], memory_config=ttnn.L1_MEMORY_CONFIG)

    xq_bnsh = ttnn.permute(xq, dims=(0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
    xk_bnsh = ttnn.permute(xk, dims=(0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

    yq_bnsh = ttnn.experimental.rotary_embedding_llama(xq_bnsh, cos_full, sin_full, trans_mat, is_decode_mode=False)
    yk_bnsh = ttnn.experimental.rotary_embedding_llama(xk_bnsh, cos_full, sin_full, trans_mat, is_decode_mode=False)

    return yq_bnsh, yk_bnsh
