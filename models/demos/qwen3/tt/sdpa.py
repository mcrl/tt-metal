import ttnn
from typing import Optional, Union
from models.demos.qwen3.common.configuration_qwen3_moe import InferenceMode
from models.demos.qwen3.utils.profiler import profile_trace, Profiler


def _explicit_pad(
    x: ttnn.Tensor,
    value: Union[float, int],
    *,
    use_multicore: Optional[bool] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    unpadded_shape = x.shape
    padded_shape = ttnn.pad_to_tile_shape(unpadded_shape=unpadded_shape)
    padding = [(0, y - x) for x, y in zip(unpadded_shape, padded_shape)]
    return ttnn.pad(
        x,
        padding=padding,
        value=value,
        **{k: v for k, v in {"use_multicore": use_multicore, "memory_config": memory_config}.items() if v is not None},
    )


def sdpa_forward_prefill(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> ttnn.Tensor:
    batch_size, num_attention_heads, sequence_length, head_dim = query.shape

    with Profiler().trace_with_timer("padding", level=4, args={"class": "sdpa_forward_prefill"}):
        query = ttnn.to_memory_config(_explicit_pad(query, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(_explicit_pad(key, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(_explicit_pad(value, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with Profiler().trace_with_timer("attention", level=4, args={"class": "sdpa_forward_prefill"}):
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=True,
            scale=scaling,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # with Profiler().trace_with_timer("to_layout", level=4, args={"class": "sdpa_forward_prefill"}):
    #     attn_output = ttnn.to_layout(
    #         ttnn.permute(attn_output, dims=(0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG),
    #         layout=ttnn.TILE_LAYOUT,
    #         memory_config=ttnn.L1_MEMORY_CONFIG,
    #     )

    with Profiler().trace_with_timer("slicing", level=4, args={"class": "sdpa_forward_prefill"}):
        start_index = (0, 0, 0, 0)
        # end_index = (batch_size, sequence_length, num_attention_heads, head_dim)
        end_index = (batch_size, num_attention_heads, sequence_length, head_dim)
        attn_output = ttnn.slice(
            attn_output,
            slice_start=start_index,
            slice_end=end_index,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return attn_output


def sdpa_forward_decode(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    cur_pos: list,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> ttnn.Tensor:
    """Q: [S=1, B, n, H], KV: [B, n, S, H]"""

    with Profiler().trace_with_timer("padding", level=4, args={"class": "sdpa_forward_decode"}):
        key = _explicit_pad(key, 0.0)

    with Profiler().trace_with_timer("to_layout", level=4, args={"class": "sdpa_forward_decode"}):
        query = ttnn.to_memory_config(
            ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        key = ttnn.to_memory_config(ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(
            ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    """ Input Q: [S=1, B, n, H], KV: [B, n, S=1, H]"""
    with Profiler().trace_with_timer("attention", level=4, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key,
            value,
            is_causal=True,
            cur_pos=cur_pos,
            scale=scaling,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
            ),
        )
    """ Output O: [S=1, B, n, H]"""

    return attn_output


def sdpa_forward(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    cur_pos: Optional[list] = None,
    mode: InferenceMode = InferenceMode.PREFILL,
) -> ttnn.Tensor:
    if mode == InferenceMode.PREFILL:
        return sdpa_forward_prefill(query, key, value, dropout, scaling)
    elif mode == InferenceMode.DECODE:
        return sdpa_forward_decode(query, key, value, cur_pos, dropout, scaling)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
