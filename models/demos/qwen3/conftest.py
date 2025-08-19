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


@pytest.fixture(scope="session")
def model_path():
    return Path(os.getenv("QWEN3_HF_MODEL", "/shared/models/Qwen3-30B-A3B"))


@pytest.fixture(scope="session")
def hf_config(model_path):
    """Load Qwen3 config for testing"""
    # Try to load from the actual config file first
    config_path = model_path / "qwen3_moe" / "config-Qwen3-30B-A3B.json"
    print(f"Looking for config file at: {config_path}")
    print(f"Config path exists: {config_path.exists()}")

    if config_path.exists():
        import json

        with open(config_path, "r") as f:
            config_data = json.load(f)
        print(f"Loaded config data: {config_data}")
        from models.demos.qwen3.reference.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        config = Qwen3MoeConfig.from_dict(config_data)
        print(f"Created config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
    else:
        # Fallback to default values
        print("Config file not found, using default values")
        from models.demos.qwen3.reference.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        config = Qwen3MoeConfig()
        print(f"Default config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
    return config


@pytest.fixture
def mesh_row(mesh_device):
    """
    Qwen3 runs many modules on a single 8-device row of a Galaxy system.
    This can be emulated on a T3K or by selecting a single submesh of a TG.
    For Galaxy+ systems (32+ devices), creates a submesh with shape (1, 8)
    and returns the first row. Otherwise, returns the original mesh_device.
    """
    if ttnn.get_num_devices() >= 32:
        rows = mesh_device.create_submeshes(ttnn.MeshShape(1, 8))
        yield rows[0]
    else:
        yield mesh_device


@pytest.fixture
def ccl(mesh_device):
    """
    Fixture to create a CCL1D instance for testing.
    This is used to test distributed operations in Qwen3 modules.
    """
    from models.demos.qwen3.tt.ccl_1d import CCL1D

    return CCL1D(mesh_device)
