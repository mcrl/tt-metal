import torch
from torch import nn
import ttnn
from pathlib import Path
from models.demos.qwen3.common.configuration_qwen3_moe import InferenceMode


def reshape_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_half1, x_half2 = x.chunk(2, dim=-1)
    stacked = torch.stack([x_half1, x_half2], dim=-1)
    return stacked.flatten(start_dim=-2)


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, mesh_device: ttnn.Device = None, interleaved: bool = False):
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.interleaved = interleaved
        self.is_tt_setup = False

    def setup_tt(self):
        if self.is_tt_setup:
            return

        if self.interleaved:
            self.weight_tensor = ttnn.as_tensor(
                reshape_to_interleaved(self.weight),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                # cache_file_name=Path.home() / ".cache/weights" / f"rmsnorm_{id(self)}_weight",
            )
        else:
            self.weight_tensor = ttnn.as_tensor(
                self.weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                # cache_file_name=Path.home() / ".cache/weights" / f"rmsnorm_{id(self)}_weight",
            )

        self.is_tt_setup = True

    def forward(self, hidden_states: ttnn.Tensor, mode: InferenceMode) -> ttnn.Tensor:
        if mode == InferenceMode.DECODE:
            mem_cfg = hidden_states.memory_config()
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG, dtype=hidden_states.dtype)
        
        hidden_states = ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.weight_tensor)

        if mode == InferenceMode.DECODE:
            hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg, dtype=hidden_states.dtype)

        return hidden_states


__all__ = ["Qwen3MoeRMSNorm"]
