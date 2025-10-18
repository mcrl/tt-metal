# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc

def reference_moe_bmm(input_tensor, weights, num_routed_tokens):
    """
    Reference PyTorch implementation of moe_bmm.
    
    For each expert e, computes:
        output[e, :, :] = input[e, :, :] @ weights[e, :, :]
    
    Only the first num_routed_tokens[e, 0] rows produce non-zero results.
    
    Args:
        input_tensor: (E/D, T, H_in) tensor
        weights: (E/D, H_in, H_out) tensor
        num_routed_tokens: (E/D, 1) tensor with token counts per expert
    
    Returns:
        output: (E/D, T, H_out) tensor
    """
    num_experts, max_tokens, h_in = input_tensor.shape
    h_out = weights.shape[2]
    
    output = torch.zeros(num_experts, max_tokens, h_out, dtype=input_tensor.dtype, device=input_tensor.device)
    
    for e in range(num_experts):
        num_tokens = num_routed_tokens[e, 0].item()
        if num_tokens > 0:
            # Only compute for active tokens
            active_input = input_tensor[e, :num_tokens, :]  # (num_tokens, H_in)
            expert_weight = weights[e, :, :]  # (H_in, H_out)
            
            # Matmul: (num_tokens, H_in) @ (H_in, H_out) -> (num_tokens, H_out)
            output[e, :num_tokens, :] = active_input @ expert_weight
    
    return output


def pad_to_tile_size(size, tile_size=32):
    """Pad size to nearest multiple of tile_size."""
    return ((size + tile_size - 1) // tile_size) * tile_size


@pytest.mark.parametrize("config", [
    {"num_experts": 8, "max_tokens": 16, "h_in": 4, "h_out": 8},
    {"num_experts": 16, "max_tokens": 256, "h_in": 4, "h_out": 8},
    {"num_experts": 32, "max_tokens": 1024, "h_in": 2048, "h_out": 768},
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
    
    # Pad dimensions to tile size (32x32)
    max_tokens_padded = pad_to_tile_size(max_tokens)
    h_in_padded = pad_to_tile_size(h_in)
    h_out_padded = pad_to_tile_size(h_out)
    
    print(f"\nTest config: experts={num_experts}, tokens={max_tokens} (padded={max_tokens_padded}), "
          f"h_in={h_in} (padded={h_in_padded}), h_out={h_out} (padded={h_out_padded})")
    
    # Create random inputs (padded for TILE layout)
    input_torch = torch.randn(num_experts, max_tokens_padded, h_in_padded, dtype=torch.bfloat16)
    weights_torch = torch.randn(num_experts, h_in_padded, h_out_padded, dtype=torch.bfloat16)
    
    # Create varied num_routed_tokens (some experts with fewer tokens)
    num_routed = torch.zeros(num_experts, 1, dtype=torch.int32)
    for e in range(num_experts):
        # Random token count between 1 and max_tokens
        num_routed[e, 0] = torch.randint(1, max_tokens + 1, (1,)).item()
    
    print(f"Token counts per expert: {num_routed.squeeze().tolist()}")
    
    # Calculate num_tiled_tokens: sum of ceiling(num_routed[e] / 32) for all experts
    total_token_tiles = 0
    for e in range(num_experts):
        num_tokens = num_routed[e, 0].item()
        num_tiles = (num_tokens + 31) // 32  # Ceiling division
        total_token_tiles += num_tiles
    
    print(f"Total token tiles: {total_token_tiles}")
    
    # Create num_tiled_tokens tensor (1, 1)
    num_tiled_tokens = torch.tensor([[total_token_tiles]], dtype=torch.int32)
    
    # Compute reference output (on non-padded dimensions conceptually)
    reference_output = reference_moe_bmm(input_torch, weights_torch, num_routed)
    
    # Convert to TTNN tensors
    # Input: (E/D, T, H_in) TILE_LAYOUT, sharded across devices
    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices
    
    # Shard input across devices by expert dimension
    input_tt = ttnn.from_torch(
        input_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    # Shard weights across devices by expert dimension
    weights_tt = ttnn.from_torch(
        weights_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    # Shard num_routed_tokens across devices
    num_routed_tt = ttnn.from_torch(
        num_routed,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    # Create num_tiled_tokens tensor (1, 1) - replicated on all devices
    num_tiled_tokens_tt = ttnn.from_torch(
        num_tiled_tokens,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    
    # Run moe_bmm operation
    output_tt = ttnn.experimental.moe_bmm(
        input_tt,
        weights_tt,
        num_routed_tt,
        num_tiled_tokens_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    
    # Convert back to torch
    output_torch = ttnn.to_torch(
        output_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    print(output_torch[0])
    print(reference_output[0])
    
    # Verify shapes
    print(f"Output shape (TT): {output_torch.shape}")
    print(f"Output shape (Ref): {reference_output.shape}")
    
    assert output_torch.shape == reference_output.shape, \
        f"Shape mismatch: {output_torch.shape} vs {reference_output.shape}"
    
    expected_shape = (num_experts, max_tokens_padded, h_out_padded)
    assert output_torch.shape == expected_shape, \
        f"Shape mismatch: {output_torch.shape} vs expected {expected_shape}"
    
    # Verify layout and dtype
    assert output_tt.layout == ttnn.TILE_LAYOUT, \
        f"Layout should be TILE, got {output_tt.layout}"
    assert output_tt.dtype == ttnn.bfloat16, \
        f"Dtype should be bfloat16, got {output_tt.dtype}"
    
    # Compare outputs per expert (only active tokens)
    max_diff_overall = 0.0
    mean_diff_overall = 0.0
    num_comparisons = 0
    
    for e in range(num_experts):
        num_tokens = num_routed[e, 0].item()
        if num_tokens > 0:
            # Compare only active tokens
            ref = reference_output[e, :num_tokens, :]
            actual = output_torch[e, :num_tokens, :]

            print(f"Expert {e} (tokens={num_tokens}):") 
            compare_tensor_pcc(ref, actual)
            
    print("\n✓ Test passed!")


if __name__ == "__main__":
    # For standalone testing without pytest
    print("To run tests, use: pytest test_moe_bmm.py -v")

