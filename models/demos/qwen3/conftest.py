# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest
from loguru import logger
from transformers import AutoConfig

import ttnn

# from models.demos.qwen3.tt.ccl_1d import CCL1D
from tests.scripts.common import get_updated_device_params

# Global cache for mesh device to avoid recreation
_mesh_device_cache = {}


@pytest.fixture(scope="module")
def cached_mesh_device():
    """
    Module-scoped fixture that caches mesh device for reuse across tests in the same module.
    This avoids recreating the device for every test.
    """
    import ttnn

    cache_key = "default"

    if cache_key not in _mesh_device_cache:
        device_ids = ttnn.get_device_ids()

        if len(device_ids) == 32:  # If running on Galaxy system
            default_mesh_shape = ttnn.MeshShape(4, 8)
        else:
            default_mesh_shape = ttnn.MeshShape(1, len(device_ids))

        # Use default device params
        device_params = {}
        updated_device_params = get_updated_device_params(device_params)

        fabric_config = updated_device_params.pop("fabric_config", None)
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

        updated_device_params.setdefault("mesh_shape", default_mesh_shape)
        mesh_device = ttnn.open_mesh_device(**updated_device_params)

        logger.debug(f"Created cached mesh device with {mesh_device.get_num_devices()} devices and shape {mesh_device.shape}")
        _mesh_device_cache[cache_key] = (mesh_device, fabric_config)

    mesh_device, fabric_config = _mesh_device_cache[cache_key]
    yield mesh_device

    # Cleanup happens only when pytest finishes


@pytest.fixture(scope="function")
def mesh_device(request, device_params, cached_mesh_device):
    """
    Function-scoped fixture that uses cached mesh device when possible.
    Falls back to creating a new device only when device_params are custom.
    """
    import ttnn

    # Check if device_params are non-default (e.g., from parametrize)
    if device_params:
        # Custom params specified, need to create a new device
        device_ids = ttnn.get_device_ids()
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids]

        if len(device_ids) == 32:  # If running on Galaxy system
            default_mesh_shape = ttnn.MeshShape(4, 8)
        else:
            default_mesh_shape = ttnn.MeshShape(1, len(device_ids))

        updated_device_params = get_updated_device_params(device_params)

        fabric_config = updated_device_params.pop("fabric_config", None)
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

        updated_device_params.setdefault("mesh_shape", default_mesh_shape)
        mesh_device = ttnn.open_mesh_device(**updated_device_params)

        logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
        yield mesh_device

        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)

        ttnn.close_mesh_device(mesh_device)
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        del mesh_device
    else:
        # No custom params, use cached device
        device_ids = ttnn.get_device_ids()
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids]
        logger.debug(f"Using cached mesh device with {cached_mesh_device.get_num_devices()} devices")
        yield cached_mesh_device
