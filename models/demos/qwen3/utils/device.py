from typing import Optional, Dict
from loguru import logger
from tests.scripts.common import get_updated_device_params
import ttnn


def create_mesh_device(device_params: Optional[Dict] = None):
    params = dict(device_params or {})

    fabric_config = params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params = get_updated_device_params(params)
    device_ids = ttnn.get_device_ids()

    # Default mesh shape: Galaxy (32) -> 4x8; otherwise 4 x num_devices // 4
    default_mesh_shape = ttnn.MeshShape(4, 8) if len(device_ids) == 32 else ttnn.MeshShape(4, len(device_ids) // 4)

    updated_device_params.setdefault("mesh_shape", default_mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)
    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")

    return mesh_device
