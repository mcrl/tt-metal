import torch
from torch import nn
import ttnn


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, mesh_device: ttnn.Device = None):
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = torch.mul(hidden_states, torch.rsqrt(variance + self.variance_epsilon))
        return torch.mul(self.weight, hidden_states.to(input_dtype))


__all__ = ["Qwen3MoeRMSNorm"]
