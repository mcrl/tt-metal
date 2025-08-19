# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.qwen3.reference.qwen3_moe.rope_helpers import apply_rotary_emb as ref_apply_rotary
from models.demos.qwen3.reference.qwen3_moe.rope_helpers import precompute_freqs_cis
from models.demos.qwen3.reference.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.rope import apply_rotary_ttnn, precompute_cossin
from models.demos.qwen3.utils.test_utils import assert_hidden_dim_pcc


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len,heads,head_dim", [(32, 4, 128), (64, 8, 128)])
def test_rope_forward(seq_len, heads, head_dim, hf_config, mesh_device):
    torch.manual_seed(0)

    # Prepare reference config for cis computation
    ref_cfg = Qwen3MoeConfig(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        rope_theta=hf_config.rope_theta,
        max_seq_len=hf_config.max_seq_len,
    )

    # Reference cis and application
    freqs_cis = precompute_freqs_cis(ref_cfg)  # [max_seq_len, head_dim//2]

    # Create random q,k in reference shape [B, S, H, D]
    B = 1
    q_ref = torch.randn(B, seq_len, heads, head_dim, dtype=torch.float32)
    k_ref = torch.randn(B, seq_len, heads, head_dim, dtype=torch.float32)
    q_rot_ref, k_rot_ref = ref_apply_rotary(q_ref, k_ref, freqs_cis[:seq_len])

    # TTNN path: build cos/sin caches compatible with ttnn.experimental.rotary_embedding
    cos, sin = precompute_cossin(
        theta=hf_config.rope_theta,
        head_dim=head_dim,
        max_seq_len=seq_len,
    )

    # Match TT shapes [1, heads, S, D]
    q_tt = q_ref.permute(0, 2, 1, 3).contiguous()
    k_tt = k_ref.permute(0, 2, 1, 3).contiguous()
    tt_q = ttnn.from_torch(
        q_tt,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_k = ttnn.from_torch(
        k_tt,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    q_rot_tt = apply_rotary_ttnn(tt_q, cos_tt, sin_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_rot_tt = apply_rotary_ttnn(tt_k, cos_tt, sin_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_rot_torch = ttnn.to_torch(
        q_rot_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), dims=(0, -1)),
    )
    k_rot_torch = ttnn.to_torch(
        k_rot_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), dims=(0, -1)),
    )

    # Bring TT output to reference layout [B, S, H, D]
    q_rot_torch = q_rot_torch.permute(0, 2, 1, 3).contiguous()
    k_rot_torch = k_rot_torch.permute(0, 2, 1, 3).contiguous()

    # Replicated tensors may concatenate device axis in dim 0; select first batch
    if q_rot_torch.shape[0] != 1:
        q_rot_torch = q_rot_torch[:1]
    if k_rot_torch.shape[0] != 1:
        k_rot_torch = k_rot_torch[:1]

    # If device concat changed head or dim sizes, slice back
    if q_rot_torch.shape[2] != heads:
        q_rot_torch = q_rot_torch[:, :, :heads, :]
    if k_rot_torch.shape[2] != heads:
        k_rot_torch = k_rot_torch[:, :, :heads, :]
    if q_rot_torch.shape[-1] != head_dim:
        q_rot_torch = q_rot_torch[..., :head_dim]
    if k_rot_torch.shape[-1] != head_dim:
        k_rot_torch = k_rot_torch[..., :head_dim]

    # No additional reordering; TT op expects interleaved even/odd already

    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(q_rot_tt)
    ttnn.deallocate(k_rot_tt)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)

    # Compare PCC over hidden dim for both q and k
    # Wrap to 4D [1,1,S,H*D] to reuse assert_hidden_dim_pcc helper
    ref_cmp_q = q_rot_ref.reshape(1, 1, seq_len, heads * head_dim)
    tt_cmp_q = q_rot_torch.reshape(1, 1, seq_len, heads * head_dim)
    assert_hidden_dim_pcc(tt_cmp_q, ref_cmp_q, pcc_required=0.98)

    ref_cmp_k = k_rot_ref.reshape(1, 1, seq_len, heads * head_dim)
    tt_cmp_k = k_rot_torch.reshape(1, 1, seq_len, heads * head_dim)
    assert_hidden_dim_pcc(tt_cmp_k, ref_cmp_k, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
