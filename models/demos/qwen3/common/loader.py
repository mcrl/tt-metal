from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.demos.qwen3.utils.profiler import profile_trace

def load_shard(ckpt_path: Path, state_dict: Dict[str, torch.Tensor]) -> None:
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            source: torch.Tensor = f.get_tensor(key)

            key = key[len("model."):] if key.startswith("model.") else key
            target: torch.Tensor = state_dict[key]

            assert source.shape == target.shape
            with torch.no_grad():
                target.copy_(source.to(dtype=torch.float16))


def load(ckpt_dir: str, model: nn.Module, io_workers: int = 4, blas_workers: int = 2) -> None:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.safetensors"))
    with torch.no_grad():
        state_dict = dict(model.named_parameters())

    num_threads = torch.get_num_threads()
    torch.set_num_threads(blas_workers)

    with torch.no_grad():
        if io_workers == 1:
            for ckpt_path in ckpt_paths:
                load_shard(ckpt_path, state_dict)
        else:
            with ThreadPoolExecutor(max_workers=io_workers) as ex:
                futures = [ex.submit(load_shard, ckpt_path, state_dict) for ckpt_path in ckpt_paths]
                for fut in as_completed(futures):
                    _ = fut.result()

    torch.set_num_threads(num_threads)

@profile_trace("load-model", level=0)
def materialize(model: nn.Module) -> None:
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
                new_parameter = nn.Parameter(torch.empty_like(parameter, device=torch.device("cpu")), requires_grad=False)
                seen_param_map[key] = new_parameter
                m._parameters[name] = new_parameter

        for child in m.children():
            _recurse(child)

    _recurse(model)
