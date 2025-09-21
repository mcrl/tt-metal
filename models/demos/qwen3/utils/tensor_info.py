# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional


def print_tensor_info(tensor: ttnn.Tensor, name: Optional[str] = None) -> None:
    """
    Utility function to print detailed information about a ttnn Tensor.

    Args:
      tensor: ttnn.Tensor object to inspect
      name: Optional name/identifier for the tensor
    """
    header = f"[{name}] " if name else "[Tensor] "
    info_parts = []

    # Basic shape information
    info_parts.append(f"shape={tensor.shape}")

    # Check for padded shape if available
    if hasattr(tensor, 'padded_shape'):
        info_parts.append(f"padded={tensor.padded_shape}")

    # Data type
    if hasattr(tensor, 'dtype'):
        info_parts.append(f"dtype={tensor.dtype}")

    # Layout information
    if hasattr(tensor, 'layout'):
        info_parts.append(f"layout={tensor.layout}")
    elif hasattr(tensor, 'get_layout'):
        info_parts.append(f"layout={tensor.get_layout()}")

    # Volume (total elements)
    if hasattr(tensor, 'volume'):
        info_parts.append(f"volume={tensor.volume()}")

    # Memory configuration
    if hasattr(tensor, 'memory_config'):
        try:
            mem_config = tensor.memory_config()
            mem_str = f"mem={mem_config}"

            # Add memory details inline if available
            details = []
            if hasattr(mem_config, 'memory_layout'):
                details.append(f"layout:{mem_config.memory_layout}")
            if hasattr(mem_config, 'buffer_type'):
                details.append(f"buffer:{mem_config.buffer_type}")

            if details:
                mem_str = f"mem=({', '.join(details)})"

            info_parts.append(mem_str)
        except:
            pass

    # Device information
    if hasattr(tensor, 'device'):
        try:
            device = tensor.device()
            if device:
                info_parts.append(f"device={device}")
        except:
            pass

    # Check if tensor is sharded
    if hasattr(tensor, 'is_sharded'):
        try:
            is_sharded = tensor.is_sharded()
            info_parts.append(f"sharded={is_sharded}")
        except:
            pass

    # Storage type
    if hasattr(tensor, 'storage_type'):
        try:
            storage = tensor.storage_type()
            info_parts.append(f"storage={storage}")
        except:
            pass

    print(header + ", ".join(info_parts))


def get_tensor_info_dict(tensor: ttnn.Tensor) -> dict:
    """
    Get tensor information as a dictionary for programmatic use.

    Args:
      tensor: ttnn.Tensor object to inspect

    Returns:
      Dictionary containing tensor information
    """
    info = {}

    # Basic shape information
    info['shape'] = tensor.shape

    # Padded shape
    if hasattr(tensor, 'padded_shape'):
        info['padded_shape'] = tensor.padded_shape

    # Data type
    if hasattr(tensor, 'dtype'):
        info['dtype'] = str(tensor.dtype)

    # Layout
    if hasattr(tensor, 'layout'):
        info['layout'] = str(tensor.layout)
    elif hasattr(tensor, 'get_layout'):
        info['layout'] = str(tensor.get_layout())

    # Volume
    if hasattr(tensor, 'volume'):
        try:
            info['volume'] = tensor.volume()
        except:
            pass

    # Memory configuration
    if hasattr(tensor, 'memory_config'):
        try:
            mem_config = tensor.memory_config()
            info['memory_config'] = str(mem_config)

            if hasattr(mem_config, 'memory_layout'):
                info['memory_layout'] = str(mem_config.memory_layout)

            if hasattr(mem_config, 'buffer_type'):
                info['buffer_type'] = str(mem_config.buffer_type)
        except:
            pass

    # Device
    if hasattr(tensor, 'device'):
        try:
            device = tensor.device()
            if device:
                info['device'] = str(device)
        except:
            pass

    # Sharding
    if hasattr(tensor, 'is_sharded'):
        try:
            info['is_sharded'] = tensor.is_sharded()
        except:
            pass

    # Storage type
    if hasattr(tensor, 'storage_type'):
        try:
            info['storage_type'] = str(tensor.storage_type())
        except:
            pass

    return info
