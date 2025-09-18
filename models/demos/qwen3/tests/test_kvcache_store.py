# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
import tt_lock

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc


@pytest.mark.parametrize(
    "batch_size,seq_len,max_seq_len,heads,head_dim",
    [
        (8, 64, 512, 64, 128),
        (8, 1, 512, 64, 128),
    ],
)
def test_kvcache_store_prefill(batch_size, seq_len, max_seq_len, heads, head_dim, mesh_device):
    torch.manual_seed(0)

    k = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.bfloat16)
    k_tt = ttnn.from_torch(
        k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    kcache = torch.zeros(batch_size, heads, max_seq_len, head_dim, dtype=torch.bfloat16)
    kcache_tt = ttnn.from_torch(
        kcache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    kcache[:, :, :seq_len, :] = k

    ### FIX HERE ###
    for b in range(batch_size):
        ttnn.kv_cache.fill_cache_for_user_(kcache_tt, k_tt[b: b + 1], b)

    kcache_tt_cpu = ttnn.to_torch(kcache_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    compare_tensor_pcc(kcache_tt_cpu, kcache, assert_mode=True)


@pytest.mark.parametrize(
    "batch_size,offset,max_seq_len,heads,head_dim",
    [
        (8, 64, 512, 64, 128),
        (8, 1, 512, 64, 128),
    ],
)
def test_kvcache_store_decode(batch_size, offset, max_seq_len, heads, head_dim, mesh_device):
    torch.manual_seed(0)

    heads_per_device = heads // mesh_device.get_num_devices()

    k = torch.randn(batch_size, heads, 1, head_dim, dtype=torch.bfloat16)
    k_tt = ttnn.from_torch(
        k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    kcache = torch.zeros(batch_size, heads, max_seq_len, head_dim, dtype=torch.bfloat16)
    kcache_tt = ttnn.from_torch(
        kcache,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    kcache[:, :, offset:offset+1, :] = k

    # Batched across users in one call at the given offset
    token_1nBh = ttnn.permute(k_tt, dims=(2, 1, 0, 3))
    ttnn.kv_cache.update_cache_for_token_(kcache_tt, token_1nBh, offset, 0)

    kcache_tt_cpu = ttnn.to_torch(kcache_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    compare_tensor_pcc(kcache_tt_cpu, kcache, assert_mode=True)


if __name__ == "__main__":
    pytest.main([__file__])
