# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
import tt_lock

from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.demos.qwen3.tt.ccl_1d import CCL1D

import tracy

def all_reduce(input_tensor_mesh, mesh_device, ccl):
    reduced = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor_mesh,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )
    out_mesh = ttnn.experimental.all_gather_async(
        reduced,
        3,
        cluster_axis=1,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_links=1,
    )
    ttnn.synchronize_device(mesh_device)
    return out_mesh

@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        [1, 1, 4096, 2048], # attention prefill [32, 128, 2048]
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_allreduce_ring_sum(per_chip_output_shape, mesh_device):
    torch.manual_seed(0)

    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires >=2 devices for AllReduce")

    num_devices = mesh_device.get_num_devices()

    input_tensor = torch.randn(per_chip_output_shape)

    ccl = TT_CCL(mesh_device)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tracy.signpost("Warmup")
    for _ in range(5):
        out_mesh = all_reduce(input_tensor_mesh, mesh_device, ccl)

    tracy.signpost("Run")
    out_mesh = all_reduce(input_tensor_mesh, mesh_device, ccl)