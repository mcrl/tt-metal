from typing import Optional

import torch
import ttnn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def sdpa_forward(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attention_mask: Optional[ttnn.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    mesh_device: ttnn.Device = None,
) -> torch.Tensor:

    query_shape = query.shape
    key_shape = key.shape
    value_shape = value.shape
    # padded_query_shape = (query_shape[0], query_shape[1], ((query_shape[2] + 31) // 32) * 32, ((query_shape[3] + 31) // 32) * 32)
    # padded_key_shape = (key_shape[0], key_shape[1], ((key_shape[2] + 31) // 32) * 32, ((key_shape[3] + 31) // 32) * 32)
    # padded_value_shape = (value_shape[0], value_shape[1], ((value_shape[2] + 31) // 32) * 32, ((value_shape[3] + 31) // 32) * 32)
    # query = ttnn.pad(query, [(0, 0), (0, 0), (0, padded_query_shape[2] - query_shape[2]), (0, padded_query_shape[3] - query_shape[3])], 0.0)
    # value = ttnn.pad(value, [(0, 0), (0, 0), (0, padded_value_shape[2] - value_shape[2]), (0, padded_value_shape[3] - value_shape[3])], 0.0)
    # key = ttnn.pad(key, [(0, 0), (0, 0), (0, padded_key_shape[2] - key_shape[2]), (0, padded_key_shape[3] - key_shape[3])], 0.0)
    # query = ttnn.to_layout(query, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # value = ttnn.to_layout(value, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # key = ttnn.to_layout(key, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    attn_output = ttnn.transformer.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=False,
        scale=scaling
    )
    attn_output = ttnn.permute(attn_output, dims=(0, 2, 1, 3))
    # attn_output = ttnn.to_torch(attn_output, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # attn_output = attn_output[:, :query_shape[2], :, :value_shape[3]]

    return attn_output
