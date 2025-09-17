from typing import Tuple
import torch
import ttnn
from models.demos.qwen3.utils.timer import start_timer, stop_timer
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


@profile_trace("apply_rotary_emb", level=3)
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
            torch.arange(start=0, end=dim, step=1, dtype=torch.int64).to(dtype=torch.float32)[:(dim)], dim
        )
        freqs = torch.reciprocal(torch.pow(theta, indices)).to(dtype=torch.float32)
        t = torch.arange(start=0, end=max_seq_len, step=1, dtype=torch.int64).to(dtype=torch.float32)
        freqs = torch.outer(t, freqs).to(dtype=torch.float32)
        freqs_cis = torch.polar(abs=torch.ones_like(input=freqs, dtype=torch.float32), angle=torch.neg(freqs))

    return freqs_cis.real, freqs_cis.imag


def apply_rotary_emb_v2(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cis: Tuple[ttnn.Tensor, ttnn.Tensor],
    trans_mat: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:

    batch_size, seq_len, num_heads, head_dim = xq.shape

    cos, sin = freqs_cis

    # if cos.shape[0] != seq_len:
    #     cos = ttnn.slice(cos, (0, 0), (seq_len, cos.shape[1]))
    #     sin = ttnn.slice(sin, (0, 0), (seq_len, sin.shape[1]))

    # cos_4d = ttnn.reshape(cos, [1, 1, seq_len, head_dim // 2])
    # sin_4d = ttnn.reshape(sin, [1, 1, seq_len, head_dim // 2])
    # cos_full = ttnn.concat([cos_4d, cos_4d], dim=-1)
    # sin_full = ttnn.concat([sin_4d, sin_4d], dim=-1)
    cos_full, sin_full = ttnn.reshape(cos, [1, 1, seq_len, head_dim]), ttnn.reshape(sin, [1, 1, seq_len, head_dim])

    xq_bnsh = ttnn.permute(xq, (0, 2, 1, 3))
    xk_bnsh = ttnn.permute(xk, (0, 2, 1, 3))

    xq_bnsh = ttnn.to_layout(xq_bnsh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    xk_bnsh = ttnn.to_layout(xk_bnsh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    yq_bnsh = ttnn.experimental.rotary_embedding_llama(xq_bnsh, cos_full, sin_full, trans_mat, is_decode_mode=False)
    yk_bnsh = ttnn.experimental.rotary_embedding_llama(xk_bnsh, cos_full, sin_full, trans_mat, is_decode_mode=False)

    yq = ttnn.to_layout(ttnn.permute(yq_bnsh, (0, 2, 1, 3)), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    yk = ttnn.to_layout(ttnn.permute(yk_bnsh, (0, 2, 1, 3)), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    return yq, yk
