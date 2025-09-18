# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
import tt_lock

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


@pytest.mark.parametrize(
    "batch_size,seq_len,heads,head_dim",
    [
        # (1, 32, 8, 128),
        # (1, 64, 16, 128),
        # (4, 64, 32, 128),
        # (8, 64, 64, 128),
        (8, 64, 64, 128),
        (8, 128, 64, 128),
        (8, 512, 64, 128),
    ],
)
def test_rope(batch_size, seq_len, heads, head_dim, mesh_device):
    torch.manual_seed(0)

    # [B S n H] Q K tensors
    q = torch.randn(batch_size, seq_len, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, heads, head_dim, dtype=torch.bfloat16)
    q_tt = ttnn.from_torch(
        q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_tt = ttnn.from_torch(
        k,
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

    trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
    trans_mat_tt = ttnn.from_torch(
        trans_mat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = ttnn_apply_rotary_emb_v2(q_tt, k_tt, (cos_tt, sin_tt), trans_mat_tt)
    q_tt_out = ttnn.to_torch(tt_out[0], dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    k_tt_out = ttnn.to_torch(tt_out[1], dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    compare_tensor_pcc(ref_out[0], q_tt_out)
    compare_tensor_pcc(ref_out[1], k_tt_out)


if __name__ == "__main__":
    pytest.main([__file__])
