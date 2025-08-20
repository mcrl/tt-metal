# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import math
import pytest
import torch

from loguru import logger

import ttnn
from models.demos.qwen3.reference.sdpa import sdpa_forward as ref_sdpa_forward
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward
from models.demos.qwen3.utils.test_utils import assert_tensor_pcc


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,head_dim",
    [
        (1, 8, 16, 64),
        (4, 8, 6, 128),
        (4, 8, 4, 128),
        (4, 8, 1, 128),
        (4, 8, 69, 128),
        (4, 64, 64, 64),
        (1, 88, 77, 43),
    ],
)
def test_sdpa_tt_matches_reference(mesh_device, batch_size, num_heads, seq_len, head_dim):
    torch.manual_seed(0)

    assert num_heads % mesh_device.get_num_devices() == 0, "Number of heads must be divisible by number of devices"

    # Inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    scale = 1.0 / math.sqrt(head_dim)

    ref_out = ref_sdpa_forward(q, k, v, attention_mask=None, dropout=0.0, scaling=scale)

    # TTNN inputs
    # mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)  # Shard along heads
    tt_q = ttnn.from_torch(
        q,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_k = ttnn.from_torch(
        k,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_v = ttnn.from_torch(
        v,
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = tt_sdpa_forward(tt_q, tt_k, tt_v, attention_mask=None, dropout=0.0, scaling=scale, mesh_device=mesh_device)

    # Cleanup device tensors
    ttnn.deallocate(tt_q)
    ttnn.deallocate(tt_k)
    ttnn.deallocate(tt_v)

    logger.info(f"TT output shape: {tt_out.shape}")
    logger.info(f"Reference output shape: {ref_out.shape}")

    # We want concat along heads because the sdpa_forward has transpose (1, 2) inside it
    tt_out = ttnn.to_torch(tt_out, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))

    # Validate PCC across hidden dim
    assert_tensor_pcc(
        tt_out,
        ref_out,
        pcc_required=0.98,
    )


if __name__ == "__main__":
    pytest.main([__file__])
