import ttnn
import pytest
import torch
import time
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc

def reference_moe_bmm(input_tensor, weights, num_routed_tokens):
    """
    Reference PyTorch implementation of moe_bmm.

    For each expert e, computes:
        output[e, :, :] = input[e, :, :] @ weights[e, :, :]

    Only the first num_routed_tokens[e] rows produce non-zero results.

    Args:
        input_tensor: (E/D, T, H_in) tensor
        weights: (E/D, H_in, H_out) tensor
        num_routed_tokens: (E/D,) 1D or (E/D, 1) 2D tensor with token counts per expert

    Returns:
        output: (E/D, T, H_out) tensor
    """
    num_experts, max_tokens, h_in = input_tensor.shape
    h_out = weights.shape[2]

    output = torch.zeros(num_experts, max_tokens, h_out, dtype=input_tensor.dtype, device=input_tensor.device)

    for e in range(num_experts):
        if num_routed_tokens.ndim == 2:
            num_tokens = num_routed_tokens[e, 0].item()
        else:
            num_tokens = num_routed_tokens[e].item()
        if num_tokens > 0:
            active_input = input_tensor[e, :num_tokens, :]  # (num_tokens, H_in)
            expert_weight = weights[e, :, :]  # (H_in, H_out)
            
            output[e, :num_tokens, :] = active_input @ expert_weight
    
    return output


def pad_to_tile_size(size, tile_size=32):
    """Pad size to nearest multiple of tile_size."""
    return ((size + tile_size - 1) // tile_size) * tile_size


@pytest.mark.parametrize("config", [
    {"num_experts": 128, "max_tokens": 2048, "h_in": 2048, "h_out": 768},
    {"num_experts": 128, "max_tokens": 64, "h_in": 2048, "h_out": 768},
    {"num_experts": 32, "max_tokens": 2048, "h_in": 4096, "h_out": 1536},
    {"num_experts": 32, "max_tokens": 1024, "h_in": 4096, "h_out": 1536},
])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_moe_bmm(mesh_device, config):
    """
    Test moe_bmm operation against PyTorch reference.
    Validates correctness for batched matrix multiplication per expert.
    """
    torch.manual_seed(42)
    
    num_experts = config["num_experts"]
    max_tokens = config["max_tokens"]
    h_in = config["h_in"]
    h_out = config["h_out"]
    
    max_tokens_padded = pad_to_tile_size(max_tokens)
    h_in_padded = pad_to_tile_size(h_in)
    h_out_padded = pad_to_tile_size(h_out)
    
    print(f"\nTest config: experts={num_experts}, tokens={max_tokens} (padded={max_tokens_padded}), "
          f"h_in={h_in} (padded={h_in_padded}), h_out={h_out} (padded={h_out_padded})")
    
    input_torch = torch.randn(num_experts, max_tokens_padded, h_in_padded, dtype=torch.bfloat16)
    weights_torch = torch.randn(num_experts, h_in_padded, h_out_padded, dtype=torch.bfloat16)
    
    num_routed = torch.zeros(num_experts, 1, dtype=torch.int32)
    for e in range(num_experts):
        num_routed[e, 0] = torch.randint(1, max_tokens + 1, (1,)).item()

    num_routed_2d = num_routed.clone()
    
    print(f"Token counts per expert: {num_routed.squeeze().tolist()}")

    reference_output = reference_moe_bmm(input_torch, weights_torch, num_routed_2d)

    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices

    input_tt = ttnn.from_torch(
        input_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    weights_tt = ttnn.from_torch(
        weights_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    num_routed_1d = num_routed.squeeze()

    num_routed_tt = ttnn.from_torch(
        num_routed_1d,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    output_tt = ttnn.experimental.moe_bmm(
        input_tt,
        weights_tt,
        num_routed_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_torch = ttnn.to_torch(
        output_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    print(f"Output shape (TT): {output_torch.shape}")
    print(f"Output shape (Ref): {reference_output.shape}")
    
    assert output_torch.shape == reference_output.shape, \
        f"Shape mismatch: {output_torch.shape} vs {reference_output.shape}"
    
    expected_shape = (num_experts, max_tokens_padded, h_out_padded)
    assert output_torch.shape == expected_shape, \
        f"Shape mismatch: {output_torch.shape} vs expected {expected_shape}"
    
    assert output_tt.layout == ttnn.TILE_LAYOUT, \
        f"Layout should be TILE, got {output_tt.layout}"
    assert output_tt.dtype == ttnn.bfloat8_b, \
        f"Dtype should be bfloat8_b, got {output_tt.dtype}"
    
    max_diff_overall = 0.0
    mean_diff_overall = 0.0
    num_comparisons = 0
    
    for e in range(num_experts):
        num_tokens = num_routed[e, 0].item()
        if num_tokens > 0:
            ref = reference_output[e, :num_tokens, :]
            actual = output_torch[e, :num_tokens, :]

            print(f"Expert {e} (tokens={num_tokens}):") 
            compare_tensor_pcc(ref, actual, assert_mode=True)
    print("Acrruaacy Test passed!")


@pytest.mark.parametrize("config", [
    {"num_experts": 32, "max_tokens": 1024, "h_in": 2880, "h_out": 2880},
])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_moe_bmm_perf(mesh_device, config):
    """
    Test moe_bmm operation against PyTorch reference.
    Validates correctness for batched matrix multiplication per expert.
    """
    torch.manual_seed(42)
    
    num_experts = config["num_experts"]
    max_tokens = config["max_tokens"]
    h_in = config["h_in"]
    h_out = config["h_out"]
    
    max_tokens_padded = pad_to_tile_size(max_tokens)
    h_in_padded = pad_to_tile_size(h_in)
    h_out_padded = pad_to_tile_size(h_out)
    
    print(f"\nTest config: experts={num_experts}, tokens={max_tokens} (padded={max_tokens_padded}), "
          f"h_in={h_in} (padded={h_in_padded}), h_out={h_out} (padded={h_out_padded})")
    
    input_torch = torch.randn(num_experts, max_tokens_padded, h_in_padded, dtype=torch.bfloat16)
    weights_torch = torch.randn(num_experts, h_in_padded, h_out_padded, dtype=torch.bfloat16)
    
    num_routed = torch.zeros(num_experts, 1, dtype=torch.int32)
    for e in range(num_experts):
        num_routed[e, 0] = max_tokens

    num_routed_2d = num_routed.clone()
    
    print(f"Token counts per expert: {num_routed.squeeze().tolist()}")

    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices

    input_tt = ttnn.from_torch(
        input_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    weights_tt = ttnn.from_torch(
        weights_torch,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    num_routed_1d = num_routed.squeeze()

    num_routed_tt = ttnn.from_torch(
        num_routed_1d,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    output_tt = ttnn.experimental.moe_bmm(
        input_tt,
        weights_tt,
        num_routed_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    
    assert output_tt.layout == ttnn.TILE_LAYOUT, \
        f"Layout should be TILE, got {output_tt.layout}"
    assert output_tt.dtype == ttnn.bfloat8_b, \
        f"Dtype should be bfloat8_b, got {output_tt.dtype}"

    print("Running performance test...")

    OPS_PER_ITERATION = 0
    for e in range(num_experts):
        num_tokens = num_routed[e, 0].item()
        if num_tokens > 0:
            OPS_PER_ITERATION += 2 * num_tokens * h_in * h_out
    print(f"FLOPs per iteration: {OPS_PER_ITERATION}")

    num_iters = 10
    start_time = time.time()
    for i in range(num_iters):
        output_tt = ttnn.experimental.moe_bmm(
            input_tt,
            weights_tt,
            num_routed_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    flops = OPS_PER_ITERATION * num_iters / elapsed_time
    print(f"Average time per iteration: {elapsed_time / num_iters:.6f} seconds")
    print(f"FLOPS: {flops / 1e12:.2f} TFLOPS")
    print(f"FLOPS per chip: {flops / num_devices / 1e12:.2f} TFLOPS")


if __name__ == "__main__":
    print("To run tests, use: pytest test_moe_bmm.py -v")

