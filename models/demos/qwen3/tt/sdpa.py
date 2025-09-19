import ttnn
import torch
from typing import Optional
from models.demos.qwen3.common.configuration_qwen3_moe import InferenceMode
from models.demos.qwen3.utils.profiler import profile_trace, Profiler

PAD_MULTIPLE = 32


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def repeat_kv_dim2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seq_len, num_key_value_heads, n_rep, head_dim
    )  # [B, S, H, 1, D] -> [B, S, H, N, D]
    return hidden_states.reshape(batch, seq_len, n_rep * num_key_value_heads, head_dim)


def sdpa_forward_prefill(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:

    query_shape = query.shape
    key_shape = key.shape
    value_shape = value.shape

    padded_query_shape = (
        query_shape[0],
        query_shape[1],
        ((query_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((query_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    padded_key_shape = (
        key_shape[0],
        key_shape[1],
        ((key_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((key_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    padded_value_shape = (
        value_shape[0],
        value_shape[1],
        ((value_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((value_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )

    with Profiler().trace_with_timer("padding", level=3, args={"class": "sdpa_forward_prefill"}):
        query = ttnn.pad(
            query,
            [(0, 0), (0, 0), (0, padded_query_shape[2] - query_shape[2]), (0, padded_query_shape[3] - query_shape[3])],
            0.0,
        )
        value = ttnn.pad(
            value,
            [(0, 0), (0, 0), (0, padded_value_shape[2] - value_shape[2]), (0, padded_value_shape[3] - value_shape[3])],
            0.0,
        )
        key = ttnn.pad(
            key, [(0, 0), (0, 0), (0, padded_key_shape[2] - key_shape[2]), (0, padded_key_shape[3] - key_shape[3])], 0.0
        )

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_prefill"}):
        query = ttnn.to_memory_config(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
        )

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_prefill"}):
        attn_output = ttnn.to_layout(ttnn.permute(attn_output, dims=(0, 2, 1, 3)), layout=ttnn.ROW_MAJOR_LAYOUT)

    with Profiler().trace_with_timer("slicing", level=3, args={"class": "sdpa_forward_prefill"}):
        attn_output = ttnn.slice(
            attn_output, [0, 0, 0, 0], [query_shape[0], query_shape[2], query_shape[1], value_shape[3]]
        )

    return attn_output


def sdpa_forward_decode(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:

    query_shape = query.shape
    key_shape = key.shape
    value_shape = value.shape

    padded_query_shape = (
        query_shape[0],
        query_shape[1],
        ((query_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((query_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    padded_key_shape = (
        key_shape[0],
        key_shape[1],
        ((key_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((key_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    padded_value_shape = (
        value_shape[0],
        value_shape[1],
        ((value_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((value_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    padded_attention_mask_shape = (
        attention_mask.shape[0],
        attention_mask.shape[1],
        ((attention_mask.shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((attention_mask.shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )

    with Profiler().trace_with_timer("padding", level=3, args={"class": "sdpa_forward_decode"}):
        query = ttnn.pad(
            query,
            [(0, 0), (0, 0), (0, padded_query_shape[2] - query_shape[2]), (0, padded_query_shape[3] - query_shape[3])],
            0.0,
        )
        value = ttnn.pad(
            value,
            [(0, 0), (0, 0), (0, padded_value_shape[2] - value_shape[2]), (0, padded_value_shape[3] - value_shape[3])],
            0.0,
        )
        key = ttnn.pad(
            key, [(0, 0), (0, 0), (0, padded_key_shape[2] - key_shape[2]), (0, padded_key_shape[3] - key_shape[3])], 0.0
        )

        attention_mask = ttnn.pad(
            attention_mask,
            [
                (0, 0),
                (0, 0),
                (0, padded_attention_mask_shape[2] - attention_mask.shape[2]),
                (0, padded_attention_mask_shape[3] - attention_mask.shape[3]),
            ],
            0.0,
        )

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_decode"}):
        query = ttnn.to_memory_config(query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention_mask = ttnn.to_memory_config(attention_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with Profiler().trace_with_timer("attention", level=3, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            scale=scaling,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
            ),
        )

    with Profiler().trace_with_timer("to_layout", level=3, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.to_layout(ttnn.permute(attn_output, dims=(0, 2, 1, 3)), layout=ttnn.ROW_MAJOR_LAYOUT)

    with Profiler().trace_with_timer("slicing", level=3, args={"class": "sdpa_forward_decode"}):
        attn_output = ttnn.slice(
            attn_output, [0, 0, 0, 0], [query_shape[0], query_shape[2], query_shape[1], value_shape[3]]
        )

    return attn_output


def sdpa_forward_decode_v2(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:

    attn_output = ttnn.transformer.scaled_dot_product_attention(
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
    )

    attn_output = ttnn.permute(attn_output, dims=(0, 2, 1, 3))

    attn_output = ttnn.to_layout(attn_output, layout=ttnn.TILE_LAYOUT)

    return attn_output


def sdpa_forward_decode_v3(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
) -> torch.Tensor:

    query = ttnn.permute(query, dims=(2, 0, 1, 3))

    # print(f"{query.shape=}, {query.dtype=}, {query.layout=}, {query.tile=}")
    # print(f"{key.shape=}, {key.dtype=}, {key.layout=}, {key.tile=}")
    # print(f"{value.shape=}, {value.dtype=}, {value.layout=}, {value.tile=}")

    key_shape = key.shape
    padded_key_shape = (
        key_shape[0],
        key_shape[1],
        ((key_shape[2] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
        ((key_shape[3] + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE,
    )
    key = ttnn.pad(key, [(0, 0), (0, 0), (0, padded_key_shape[2] - key_shape[2]), (0, 0)], 0.0)

    value = ttnn.to_layout(value, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    key = ttnn.to_layout(key, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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
    )

    attn_output = ttnn.permute(attn_output, dims=(1, 0, 2, 3))

    attn_output = ttnn.to_layout(attn_output, layout=ttnn.TILE_LAYOUT)

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
        # return sdpa_forward_decode(query, key, value, attention_mask, dropout, scaling)
        # return sdpa_forward_decode_v2(query, key, value, attention_mask, dropout, scaling)
        return sdpa_forward_decode_v3(query, key, value, attention_mask, dropout, scaling)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
