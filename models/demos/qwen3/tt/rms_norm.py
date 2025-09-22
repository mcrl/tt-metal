import torch
from torch import nn
import ttnn
from pathlib import Path


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, mesh_device: ttnn.Device = None):
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def setup_tt(self):
        self.weight_tt = ttnn.as_tensor(
            self.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=Path.home() / ".cache/weights" / f"rmsnorm_{id(self)}_weight",
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            hidden_states, epsilon=self.epsilon, weight=self.weight_tt, memory_config=ttnn.L1_MEMORY_CONFIG
        )


__all__ = ["Qwen3MoeRMSNorm"]
