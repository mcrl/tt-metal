import ttnn
import torch
from typing import Optional, Union
from models.demos.qwen3.common.configuration_qwen3_moe import InferenceMode
from models.demos.qwen3.utils.profiler import profile_trace, Profiler


def _explicit_pad(
        x: ttnn.Tensor, value: Union[float, int],
        *, use_multicore: Optional[bool] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor:
    unpadded_shape = x.shape
    padded_shape = ttnn.pad_to_tile_shape(unpadded_shape=unpadded_shape)
    padding = [(0, y - x) for x, y in zip(unpadded_shape, padded_shape)]
    return ttnn.pad(x, padding=padding, value=value, **
                    {k: v for k, v in {"use_multicore": use_multicore, "memory_config": memory_config}.items() if v is not None})


def sdpa_forward_prefill(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_attention_heads, sequence_length, head_dim = query.shape

    with Profiler().trace_with_timer("padding", level=3, args={"class": "sdpa_forward_prefill"}):
        query = ttnn.to_memory_config(_explicit_pad(query, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(_explicit_pad(key, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(_explicit_pad(value, 0.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with Profiler().trace_with_timer("attention", level=3, args={"class": "sdpa_forward_prefill"}):
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=True,
            scale=scaling,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_prefill"}):
        attn_output = ttnn.to_layout(ttnn.permute(attn_output, dims=(0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG),
                                     layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    with Profiler().trace_with_timer("slicing", level=3, args={"class": "sdpa_forward_prefill"}):
        start_index = (0, 0, 0, 0)
        end_index = (batch_size, sequence_length, num_attention_heads, head_dim)
        attn_output = ttnn.slice(attn_output, slice_start=start_index, slice_end=end_index, memory_config=ttnn.L1_MEMORY_CONFIG)

    return attn_output


def sdpa_forward_decode(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:
    with Profiler().trace_with_timer("permute", level=3):
        query = ttnn.permute(query, dims=(2, 0, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

    with Profiler().trace_with_timer("padding", level=3, args={"class": "sdpa_forward_decode"}):
        key = _explicit_pad(key, 0.0)

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_decode"}):
        query = ttnn.to_memory_config(ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with Profiler().trace_with_timer("attention", level=3, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=False,
            scale=scaling,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
            ),
            # memory_config=ttnn.L1_MEMORY_CONFIG,  # Incorrect Results!
        )

    with Profiler().trace_with_timer("permute", level=3):
        attn_output = ttnn.permute(attn_output, dims=(1, 0, 2, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.to_layout(attn_output, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    return attn_output


def sdpa_forward(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    mode: InferenceMode = InferenceMode.PREFILL,
) -> torch.Tensor:
    if mode == InferenceMode.PREFILL:
        return sdpa_forward_prefill(query, key, value, attention_mask, dropout, scaling)
    elif mode == InferenceMode.DECODE:
        return sdpa_forward_decode(query, key, value, attention_mask, dropout, scaling)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
