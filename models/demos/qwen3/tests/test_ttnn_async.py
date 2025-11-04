import ttnn
import time
import torch
from models.demos.qwen3.utils.device import create_mesh_device


def test_ttnn_async(mesh_device):
    ttnn.synchronize_device(mesh_device)
    N = 8192

    a = torch.randn((N, N), dtype=torch.bfloat16)
    b = torch.randn((N, N), dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

    for i in range(10):
        start_time = time.perf_counter()
        output = ttnn.linear(a_tt, b_tt)
        api_return_time = time.perf_counter()
        print(f"API return time: {(api_return_time - start_time) * 1000:.2f} ms")
        ttnn.synchronize_device(mesh_device)
        end_time = time.perf_counter()
        print(f"End time: {(end_time - start_time) * 1000:.2f} ms")


if __name__ == "__main__":
    mesh_device = create_mesh_device()
    test_ttnn_async(mesh_device)
