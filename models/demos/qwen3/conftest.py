import os
from pathlib import Path
import warnings

import pytest
from loguru import logger
from transformers import AutoConfig

import ttnn

from models.demos.qwen3.utils.device import create_mesh_device

# Silence noisy deprecation warnings from third-party packages (Pydantic V2 migration)
try:
    from pydantic.warnings import PydanticDeprecatedSince20  # type: ignore
except Exception:  # pragma: no cover
    PydanticDeprecatedSince20 = None  # type: ignore

if PydanticDeprecatedSince20 is not None:  # type: ignore
    warnings.simplefilter("ignore", PydanticDeprecatedSince20)  # type: ignore


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

    mesh_device = create_mesh_device(device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)

    if device_params and "fabric_config" in device_params:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    del mesh_device
