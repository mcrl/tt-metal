from __future__ import annotations

from pathlib import Path


__all__ = ["ttnn_model_cache_path"]


def _cache_root() -> Path:
    """Return the base directory for TTNN model cache.

    Uses ${HOME}/.cache/ttnn-weights.
    """
    return Path.home() / ".cache" / "ttnn-weights"


def ttnn_model_cache_path(name: str) -> str:
    """Return absolute cache file path for a given model `name`.

    Ensures parent directories exist. `name` may include subdirectories
    (e.g. "qwen3/7b/weights.bin").
    """
    base = _cache_root()
    path = (base / name).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
