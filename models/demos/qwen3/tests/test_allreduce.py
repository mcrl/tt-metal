# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.demos.qwen3.tt.ccl_1d import CCL1D


@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        # [1, 1, 32, 1024],
        [1, 1, 64, 512],
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

    ccl = CCL1D(mesh_device)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_mesh = ttnn.experimental.all_reduce_async(
        input_tensor_mesh,
        math_op=ttnn.ReduceType.Sum,
        from_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
        to_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
        gather_multi_device_global_semaphore=ccl.get_semaphore(0),
        num_links=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)

    answer = input_tensor * num_devices

    # Validate each device tensor
    for i, dev_tensor in enumerate(ttnn.get_device_tensors(out_mesh)):
        out_cpu = ttnn.to_torch(dev_tensor)
        compare_tensor_pcc(out_cpu, answer, assert_mode=True)
