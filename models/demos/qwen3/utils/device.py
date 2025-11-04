from typing import Optional, Dict
from loguru import logger
from tests.scripts.common import get_updated_device_params
import ttnn

# Flag to track if we've already printed the device info
_printed_device_info = False


def create_mesh_device(device_params: Optional[Dict] = None):
    """Create mesh device with appropriate mesh shape based on available devices.

    Mesh shape selection:
    - Galaxy (32 devices): 4x8
    - T3000 (8+ devices): 1x8
    - Single/Few devices: 1x{num_devices}

    Args:
        device_params: Optional device parameters (trace_region_size, fabric_config, etc.)

    Returns:
        Initialized mesh device
    """
    global _printed_device_info

    params = dict(device_params or {})

    fabric_config = params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params = get_updated_device_params(params)
    device_ids = ttnn.get_device_ids()

    # Determine default mesh shape based on number of devices
    if len(device_ids) == 32:  # Galaxy system
        default_mesh_shape = ttnn.MeshShape(4, 8)
    elif len(device_ids) >= 8:  # T3000 or similar
        default_mesh_shape = ttnn.MeshShape(2, 4)
        # default_mesh_shape = ttnn.MeshShape(1, 8)
    else:  # Single device or smaller configurations
        default_mesh_shape = ttnn.MeshShape(1, len(device_ids))

    updated_device_params.setdefault("mesh_shape", default_mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)

    # Print device info only once (first call)
    if not _printed_device_info:
        _printed_device_info = True
        mesh_shape = mesh_device.shape
        num_devices = mesh_device.get_num_devices()

        print(f"=" * 60)
        print(f"Mesh Device Configuration:")
        print(f"  Available Devices: {len(device_ids)}")
        print(f"  Mesh Shape: {mesh_shape[0]}x{mesh_shape[1]} ({num_devices} devices)")
        print(f"  Fabric Config: {fabric_config if fabric_config else 'DISABLED'}")
        if "trace_region_size" in params:
            trace_mb = params["trace_region_size"] // (1024 * 1024)
            print(f"  Trace Region Size: {trace_mb}MB")
        print(f"=" * 60)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")

    return mesh_device
