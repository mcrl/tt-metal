"""
Memory-efficient loader that uses safetensors mmap for lazy weight loading.

This keeps models in meta state and loads weights on-demand during setup_tt(),
using memory-mapped files for zero-copy access.
"""

from typing import Dict, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from safetensors import safe_open
from models.demos.qwen3.utils.profiler import profile_trace
import gc
import psutil
import os


def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    except:
        return 0.0


class LazyWeightLoader:
    """
    Lazy weight loader using safetensors memory-mapped files.
    
    This keeps file handles open with mmap, allowing zero-copy access to weights
    without loading everything into RAM.
    """
    
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_paths = sorted(self.ckpt_dir.glob("*.safetensors"))
        
        if not self.ckpt_paths:
            raise ValueError(f"No .safetensors files found in {ckpt_dir}")
        
        # Keep mmap'd file handles open for zero-copy access
        self.mmap_files = {}  # path -> file handle
        
        # Build a map from parameter name to (checkpoint file, original key)
        self.param_to_file = {}
        self.param_to_key = {}  # Store original key with "model." prefix if present
        self.param_metadata = {}  # Store shape/dtype info
        
        for ckpt_path in self.ckpt_paths:
            # Open with mmap for zero-copy access
            f = safe_open(ckpt_path, framework="pt", device="cpu")
            self.mmap_files[str(ckpt_path)] = f
            
            for key in f.keys():
                clean_key = key[len("model."):] if key.startswith("model.") else key
                self.param_to_file[clean_key] = str(ckpt_path)
                self.param_to_key[clean_key] = key
                
                # Store metadata without loading the full tensor
                tensor_slice = f.get_slice(key)
                self.param_metadata[clean_key] = {
                    'shape': tensor_slice.get_shape(),
                    'dtype': str(tensor_slice.get_dtype())
                }
        
        print(f"✓ Initialized lazy loader: {len(self.param_to_file)} parameters across {len(self.ckpt_paths)} files")
        print(f"✓ Using memory-mapped access - minimal RAM usage until weights are loaded")
    
    def get_parameter(self, param_name: str, load_to_ram: bool = True) -> torch.Tensor:
        """
        Get a parameter using memory-mapped access.
        
        Args:
            param_name: Name of the parameter to load
            load_to_ram: If True, clone tensor to RAM. If False, return mmap'd view.
                        Note: mmap tensors must be cloned before device upload.
        
        Returns:
            Tensor loaded from disk (mmap'd or copied to RAM)
        """
        if param_name not in self.param_to_file:
            # Provide helpful error message with suggestions
            available_params = [p for p in self.param_to_file.keys() if param_name.split('.')[-1] in p]
            error_msg = f"Parameter '{param_name}' not found in checkpoints"
            if available_params:
                error_msg += f"\nDid you mean one of these?\n  " + "\n  ".join(available_params[:5])
            raise ValueError(error_msg)
        
        ckpt_path = self.param_to_file[param_name]
        original_key = self.param_to_key[param_name]
        
        try:
            # Access via mmap
            f = self.mmap_files[ckpt_path]
            tensor = f.get_tensor(original_key)
            
            if load_to_ram:
                # Clone to break mmap reference and load into RAM
                # This is necessary before uploading to device
                tensor = tensor.clone()
            
            # Convert to bfloat16 if needed
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(dtype=torch.bfloat16)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Error loading parameter '{param_name}' from {ckpt_path}: {e}")
    
    def get_parameter_metadata(self, param_name: str) -> Dict[str, Any]:
        """Get parameter shape and dtype without loading the data."""
        if param_name not in self.param_metadata:
            raise ValueError(f"Parameter '{param_name}' not found")
        return self.param_metadata[param_name]
    
    def has_parameter(self, param_name: str) -> bool:
        """Check if a parameter exists."""
        return param_name in self.param_to_file
    
    def list_parameters(self, prefix: str = "") -> list[str]:
        """List all parameter names, optionally filtered by prefix."""
        if prefix:
            return [name for name in self.param_to_file.keys() if name.startswith(prefix)]
        return list(self.param_to_file.keys())
    
    def close(self):
        """Close all mmap'd file handles."""
        self.mmap_files.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global loader instance
_global_lazy_loader: Optional[LazyWeightLoader] = None


def init_lazy_loader(ckpt_dir: str) -> LazyWeightLoader:
    """Initialize the global lazy loader."""
    global _global_lazy_loader
    _global_lazy_loader = LazyWeightLoader(ckpt_dir)
    return _global_lazy_loader


def get_lazy_loader() -> Optional[LazyWeightLoader]:
    """Get the global lazy loader instance."""
    return _global_lazy_loader


def is_meta_tensor(tensor: torch.Tensor) -> bool:
    """Safely check if a tensor is on meta device."""
    try:
        return tensor.device.type == 'meta'
    except:
        return False


def load_parameter_for_module(module: nn.Module, param_name: str, full_param_path: str):
    """
    Load a parameter from the lazy loader and assign it to a module.
    
    Args:
        module: The module to assign the parameter to
        param_name: The parameter name within the module (e.g., "weight", "bias")
        full_param_path: The full path to the parameter in checkpoints (e.g., "layers.0.self_attn.q_proj.weight")
    """
    loader = get_lazy_loader()
    if loader is None:
        raise RuntimeError("Lazy loader not initialized. Call init_lazy_loader() first.")
    
    try:
        # Load the weight
        print(f"  Loading parameter: {full_param_path}")
        weight = loader.get_parameter(full_param_path, load_to_ram=True)
        print(f"    ✓ Loaded {full_param_path}: shape={weight.shape}, dtype={weight.dtype}")
        
        # Assign to module
        module._parameters[param_name] = nn.Parameter(weight, requires_grad=False)
    except Exception as e:
        print(f"    ✗ Failed to load {full_param_path}: {e}")
        raise


def load_parameters_for_module(module: nn.Module, param_prefix: str):
    """
    Load all parameters for a module from the lazy loader.
    
    Args:
        module: The module to load parameters for
        param_prefix: Prefix for parameter names (e.g., "layers.0.self_attn")
    """
    loader = get_lazy_loader()
    if loader is None:
        raise RuntimeError("Lazy loader not initialized. Call init_lazy_loader() first.")
    
    # Recursively load parameters
    def _load_params(mod: nn.Module, prefix: str):
        for param_name, parameter in mod._parameters.items():
            if parameter is None:
                continue
            if not parameter.is_meta:
                continue
            
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            
            if loader.has_parameter(full_name):
                weight = loader.get_parameter(full_name, load_to_ram=True)
                mod._parameters[param_name] = nn.Parameter(weight, requires_grad=False)
        
        for child_name, child in mod._modules.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _load_params(child, child_prefix)
    
    _load_params(module, param_prefix)


def clear_module_weights(module: nn.Module):
    """
    Clear CPU weights from a module after uploading to device.
    Converts parameters back to meta tensors to free RAM.
    """
    import gc
    
    def _recurse(mod: nn.Module):
        for name, parameter in mod._parameters.items():
            if parameter is None:
                continue
            
            # Skip if already meta
            if is_meta_tensor(parameter):
                continue
            
            # Replace with meta tensor to free memory
            with torch.no_grad():
                meta_param = nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype, device='meta'),
                    requires_grad=False
                )
                mod._parameters[name] = meta_param
        
        for child in mod._modules.values():
            _recurse(child)
    
    _recurse(module)
    
    # Force garbage collection to immediately free memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Backward compatibility with existing code
@profile_trace("materialize-meta", level=1)
def materialize(model: nn.Module) -> None:
    """
    Materialize model parameters as meta tensors only (no RAM allocation).
    
    This replaces the old materialize() which allocated CPU RAM.
    Now we keep everything in meta state until setup_tt() is called.
    """
    seen_param_map = dict()

    def _recurse(m: nn.Module) -> None:
        for name, parameter in m._parameters.items():
            if parameter is None:
                continue
            if not parameter.is_meta:
                continue

            parameter.grad = None
            key = id(parameter)
            if key in seen_param_map:
                m._parameters[name] = seen_param_map[key]
            else:
                # Keep as meta tensor - no RAM allocation!
                new_parameter = nn.Parameter(
                    torch.empty_like(parameter, device=torch.device("meta")), 
                    requires_grad=False
                )
                seen_param_map[key] = new_parameter
                m._parameters[name] = new_parameter

        for child in m.children():
            _recurse(child)

    _recurse(model)
    print("✓ Model materialized as meta tensors (zero RAM usage)")


@profile_trace("load-with-lazy-loader", level=1)
def load(ckpt_dir: str, model: nn.Module, lazy: bool = False, io_workers: int = 4, blas_workers: int = 2) -> None:
    """
    Load model weights with optional lazy loading.
    
    Args:
        ckpt_dir: Directory containing .safetensors files
        model: Model with meta tensors
        lazy: If True, only initialize lazy loader without loading weights.
              If False, load all weights to RAM (old behavior).
        io_workers: Number of I/O worker threads (only used if lazy=False)
        blas_workers: Number of BLAS threads (only used if lazy=False)
    """
    if lazy:
        # Initialize lazy loader - no weights loaded yet
        init_lazy_loader(ckpt_dir)
        print("✓ Lazy loader initialized - weights will be loaded on-demand during setup_tt()")
    else:
        # Old behavior - load everything to RAM
        from models.demos.qwen3.common.loader import load as old_load
        old_load(ckpt_dir, model, io_workers, blas_workers)
