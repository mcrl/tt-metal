from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache


__all__ = ["ttnn_model_cache_path", "get_model_path", "get_model_name", "get_model_short_name", "get_model_cache_prefix"]

# Flag to track if we've already printed the model info
_printed_model_info = False


def _cache_root() -> Path:
    """Return the base directory for TTNN model cache.

    Uses ${HOME}/.cache/ttnn-weights.
    """
    return Path.home() / ".cache" / ("ttnn-weights" + get_model_base_name())


def get_model_name() -> str:
    """Get the model name from QWEN3_MODEL environment variable.

    Returns:
        Model name (e.g., "Qwen3-30B-A3B" or "Qwen3-235B-A22B")
        Default: "Qwen3-30B-A3B" if QWEN3_MODEL is not set

    Raises:
        ValueError: If QWEN3_MODEL is set but invalid
    """
    model_name = os.getenv("QWEN3_MODEL", "Qwen3-30B-A3B")

    valid_models = ["Qwen3-30B-A3B", "Qwen3-235B-A22B", "qwen3-235b", "gpt-oss-120b", "gpt-oss-20b", "DeepSeek-R1"]
    if model_name not in valid_models:
        raise ValueError(
            f"Invalid QWEN3_MODEL: {model_name}. "
            f"Must be one of: {', '.join(valid_models)}"
        )

    return model_name


def get_model_base_name() -> str:
    """Get the base model name without size or suffix.

    Returns:
        Base model name (e.g., "qwen3" or "gpt-oss" or "DeepSeek-R1")
    """
    model_name = get_model_name().lower()
    if "qwen3" in model_name:
        return ""
    elif "gpt-oss" in model_name:
        return "-gpt-oss"
    elif "deepseek-r1" in model_name:
        return "-deepseek"
    else:
        raise ValueError(f"Unknown base model in: {model_name}")


def get_model_short_name() -> str:
    """Get short model name for cache prefix.

    Returns:
        Short name (e.g., "30b" or "235b")
    """
    model_name = get_model_name().lower()
    # Extract model size: "Qwen3-30B-A3B" -> "30b"
    if "20b" in model_name:
        return "20b"
    elif "30b" in model_name:
        return "30b"
    elif "120b" in model_name:
        return "120b"
    elif "235b" in model_name:
        return "235b"
    elif "r1" in model_name:
        return "671b"
    else:
        raise ValueError(f"Unknown model size in: {model_name}")


def get_model_cache_prefix(mesh_device) -> str:
    """Get model cache prefix including model name and mesh shape.

    Args:
        mesh_device: TT mesh device

    Returns:
        Cache prefix string (e.g., "30b_4x8_" or "30b_1x8_")

    Example:
        >>> prefix = get_model_cache_prefix(mesh_device)
        >>> cache_path = ttnn_model_cache_path(f"{prefix}decoder_0_qkv_proj")
    """
    model_short = get_model_short_name()
    mesh_shape = mesh_device.shape
    return f"{model_short}_{mesh_shape[0]}x{mesh_shape[1]}_"


def get_model_path() -> str:
    """Get the full model path from environment variables.

    Constructs path from QWEN3_MODEL_DIR (default: /shared/models)
    and QWEN3_MODEL.

    Returns:
        Full model path (e.g., "/shared/models/Qwen3-30B-A3B")

    Raises:
        ValueError: If QWEN3_MODEL is not set or invalid
    """
    global _printed_model_info

    model_dir = os.getenv("QWEN3_MODEL_DIR", "/shared/models")
    model_name = get_model_name()
    model_path = os.path.join(model_dir, model_name)

    # Print model info only once (first call)
    if not _printed_model_info:
        _printed_model_info = True
        is_default_model = os.getenv("QWEN3_MODEL") is None
        is_default_dir = os.getenv("QWEN3_MODEL_DIR") is None

        print(f"=" * 60)
        print(f"Qwen3 Model Configuration:")
        print(f"  Model: {model_name}" + (" (default)" if is_default_model else ""))
        print(f"  Model Directory: {model_dir}" + (" (default)" if is_default_dir else ""))
        print(f"  Full Path: {model_path}")
        print(f"  Cache Prefix: {get_model_short_name()}")
        print(f"=" * 60)

    return model_path


def ttnn_model_cache_path(name: str) -> str:
    """Return absolute cache file path for a given model `name`.

    Ensures parent directories exist. `name` may include subdirectories
    (e.g. "qwen3/7b/weights.bin").
    """
    base = _cache_root()
    path = (base / name).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
