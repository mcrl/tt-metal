# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.qwen3.utils.tensor_info import print_tensor_info
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.rope import (
    apply_rotary_emb as ttnn_apply_rotary_emb,
    precompute_freqs_cis as ttnn_precompute_freqs_cis,
    precompute_freqs_cis_v2 as ttnn_precompute_freqs_cis_v2,
    apply_rotary_emb_v2 as ttnn_apply_rotary_emb_v2,
)
from models.demos.qwen3.reference.rope import (
    apply_rotary_emb as ref_apply_rotary_emb,
    precompute_freqs_cis as ref_precompute_freqs_cis,
)
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.tt_transformers.tt.common import get_rot_transformation_mat

from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding, apply_rotary_pos_emb

def reshape_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_half1, x_half2 = x.chunk(2, dim=-1)
    stacked = torch.stack([x_half1, x_half2], dim=-1)
    return stacked.flatten(start_dim=-2)

def reshape_from_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    bsz, seqlen, num_heads, head_dim = x.shape
    unflattened = x.reshape([bsz, seqlen, num_heads, -1, 2])
    x_half1 = unflattened[..., 0]
    x_half2 = unflattened[..., 1]
    return ttnn.concat([x_half1, x_half2], dim=-1)

@pytest.mark.parametrize(
    "batch_size,seq_len,heads,head_dim",
    [
        # (1, 32, 8, 128),
        # (1, 64, 16, 128),
        # (4, 64, 32, 128),
        # (8, 64, 64, 128),
        (8, 64, 32, 128),
        (8, 128, 32, 128),
        (8, 512, 32, 128),
    ],
)
def test_rope(batch_size, seq_len, heads, head_dim, mesh_device):
    torch.manual_seed(0)

    # [B S n H] Q K tensors
    q = torch.randn(batch_size, seq_len, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, heads, head_dim, dtype=torch.bfloat16)
    q_tt = ttnn.from_torch(
        reshape_to_interleaved(q),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_tt = ttnn.from_torch(
        reshape_to_interleaved(k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    ref_position_embeddings = ref_precompute_freqs_cis(Qwen3MoeConfig(head_dim=head_dim, max_seq_len=seq_len))
    ttnn_position_embeddings = ttnn_precompute_freqs_cis_v2(Qwen3MoeConfig(head_dim=head_dim, max_seq_len=seq_len))
    cos_tt = ttnn.from_torch(
        ttnn_position_embeddings[0],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        ttnn_position_embeddings[1],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    ref_out = ref_apply_rotary_emb(q, k, ref_position_embeddings)

    config = AutoConfig.from_pretrained("/shared/models/Qwen3-30B-A3B/")
    ref_rope = Qwen3MoeRotaryEmbedding(config=config)

    start_pos = 0
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.int64).unsqueeze(0)
    ref_cos, ref_sin = ref_rope(q, position_ids)
    ref_out_2 = apply_rotary_pos_emb(q, k, ref_cos, ref_sin, unsqueeze_dim=2)

    trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
    trans_mat_tt = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = ttnn_apply_rotary_emb_v2(q_tt, k_tt, (cos_tt, sin_tt), trans_mat_tt)

    q_tt_out = ttnn.to_layout(tt_out[0], ttnn.ROW_MAJOR_LAYOUT)
    k_tt_out = ttnn.to_layout(tt_out[1], ttnn.ROW_MAJOR_LAYOUT)

    q_tt_out = ttnn.permute(q_tt_out, (0, 2, 1, 3))
    k_tt_out = ttnn.permute(k_tt_out, (0, 2, 1, 3))

    q_tt_out = reshape_from_interleaved(q_tt_out)
    k_tt_out = reshape_from_interleaved(k_tt_out)

    q_tt_out = ttnn.to_torch(q_tt_out, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    k_tt_out = ttnn.to_torch(k_tt_out, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    compare_tensor_pcc(ref_out[0], q_tt_out)
    compare_tensor_pcc(ref_out[1], k_tt_out)

    compare_tensor_pcc(ref_out[0], ref_out_2[0])
    compare_tensor_pcc(ref_out[1], ref_out_2[1])


if __name__ == "__main__":
    pytest.main([__file__])
