# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch

import ttnn


def precompute_cossin(
    *,
    theta: float,
    head_dim: int,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin caches for TT usage.

    Returns tensors shaped [max_seq_len, head_dim] in torch.float32 by default.

    Notes:
    - Matches reference Qwen3 cis definition that uses angle = -freqs; to align,
      we emit sin with a negative sign.
    """

    with torch.device("cpu"), torch.no_grad():
        half_dim = head_dim // 2
        assert head_dim % 2 == 0, "head_dim must be even for rotary embedding"

        indices = torch.arange(0, half_dim, dtype=torch.float32) / head_dim
        freqs = torch.pow(theta, -indices)  # [half_dim]
        t = torch.arange(0, max_seq_len, dtype=torch.float32)  # [max_seq_len]
        angles = torch.outer(t, freqs).to(dtype)  # [max_seq_len, half_dim]

        # Duplicate half-dim by concatenation: [t1, .., t_half, t1, .., t_half]
        angles_cat = torch.cat([angles, angles], dim=-1)  # [max_seq_len, head_dim]
        cos = torch.cos(angles_cat)
        sin = torch.sin(angles_cat)

    return cos.to(dtype), sin.to(dtype)


def apply_rotary_ttnn(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Apply RoPE to a TTNN tensor using precomputed cos/sin caches.

    Args:
        x: Input tensor shaped [1, heads, seq_len, head_dim] (TILE layout).
        cos, sin: Caches shaped [seq_len, head_dim // 2].
        memory_config: Optional output memory config (defaults to x.memory_config()).

    Returns:
        Rotated tensor with the same shape as input.
    """

    output_memcfg = memory_config or x.memory_config()
    return ttnn.experimental.rotary_embedding(x, cos, sin, None, memory_config=output_memcfg)
