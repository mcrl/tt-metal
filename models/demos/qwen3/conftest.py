# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest
from loguru import logger
from transformers import AutoConfig

import ttnn

# from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.demos.qwen3.utils.device import create_mesh_device


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """
    Pytest fixture to set up a device mesh for Qwen3 tests.
    Many Qwen3 submodules operate on a single row of devices,
    so we are happy to run those on a TG or T3K. Others need
    the full Galaxy mesh in (rows=4, cols=8) format.

    If a galaxy is available, it returns a mesh of 4x8 devices.
    If a t3k is available, it returns a mesh of 1x8 devices.
    If no galaxy or t3k is available, it returns a mesh of 1x1 devices.

    Yields:
        mesh_device: Initialized device mesh object.
    """
    import ttnn

    device_ids = ttnn.get_device_ids()
    request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids]

    if len(device_ids) == 32:  # If running on Galaxy system
        default_mesh_shape = ttnn.MeshShape(4, 8)
    else:
        default_mesh_shape = ttnn.MeshShape(4, 2)

    updated_device_params = get_updated_device_params(device_params)

    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params.setdefault("mesh_shape", default_mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2, 8))

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
    yield mesh_device

    # Cleanup
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)

    # Reset fabric config if it was set
    if device_params and "fabric_config" in device_params:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    del mesh_device
