# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch


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
    {"num_experts": 8, "max_tokens": 256, "h_in": 4, "h_out": 8},
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
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    # Run moe_bmm operation
    output_tt = ttnn.experimental.moe_bmm(
        input_tt,
        weights_tt,
        num_routed_tt,
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
            
            diff = torch.abs(ref - actual)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            max_diff_overall = max(max_diff_overall, max_diff)
            mean_diff_overall += mean_diff * num_tokens
            num_comparisons += num_tokens
            
            print(f"Expert {e} (tokens={num_tokens}): max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            
            # Assert reasonable accuracy for bfloat16
            # Allow higher tolerance due to accumulation in matmul
            assert max_diff < 1.0, f"Expert {e}: max diff {max_diff} too large"
    
    mean_diff_overall /= num_comparisons
    print(f"\nOverall: max_diff={max_diff_overall:.6f}, mean_diff={mean_diff_overall:.6f}")
    
    print("\n✓ Test passed!")


@pytest.mark.parametrize("num_experts,max_tokens,h_in,h_out", [
    (1, 32, 128, 256),  # Single expert
    (2, 64, 256, 512),  # Two experts
    (4, 64, 512, 1024),  # Four experts
])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_moe_bmm_varying_token_counts(mesh_device, num_experts, max_tokens, h_in, h_out):
    """
    Test moe_bmm with varying token counts per expert to verify
    correct handling of num_routed_tokens.
    """
    torch.manual_seed(123)
    
    # Pad to tile size
    max_tokens_padded = pad_to_tile_size(max_tokens)
    h_in_padded = pad_to_tile_size(h_in)
    h_out_padded = pad_to_tile_size(h_out)
    
    print(f"\nTest varying tokens: experts={num_experts}, max_tokens={max_tokens_padded}, "
          f"h_in={h_in_padded}, h_out={h_out_padded}")
    
    # Create inputs
    input_torch = torch.randn(num_experts, max_tokens_padded, h_in_padded, dtype=torch.bfloat16)
    weights_torch = torch.randn(num_experts, h_in_padded, h_out_padded, dtype=torch.bfloat16)
    
    # Create specific token counts: [max, max/2, max/4, ...]
    num_routed = torch.zeros(num_experts, 1, dtype=torch.int32)
    for e in range(num_experts):
        count = max(1, max_tokens // (2 ** e))
        num_routed[e, 0] = count
    
    print(f"Token counts: {num_routed.squeeze().tolist()}")
    
    # Reference computation
    reference_output = reference_moe_bmm(input_torch, weights_torch, num_routed)
    
    # Convert to TTNN
    num_devices = mesh_device.get_num_devices()
    
    input_tt = ttnn.from_torch(
        input_torch, device=mesh_device, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    weights_tt = ttnn.from_torch(
        weights_torch, device=mesh_device, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    num_routed_tt = ttnn.from_torch(
        num_routed, device=mesh_device, dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    
    # Run operation
    output_tt = ttnn.experimental.moe_bmm(input_tt, weights_tt, num_routed_tt)
    output_torch = ttnn.to_torch(output_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    
    # Verify per expert
    for e in range(num_experts):
        num_tokens = num_routed[e, 0].item()
        ref = reference_output[e, :num_tokens, :]
        actual = output_torch[e, :num_tokens, :]
        
        diff = torch.abs(ref - actual)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Expert {e} (tokens={num_tokens}): max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1.0, f"Expert {e}: max diff {max_diff} too large"
        
        # Verify zeros after active tokens
        if num_tokens < max_tokens:
            inactive = output_torch[e, num_tokens:max_tokens, :]
            assert torch.abs(inactive).max() < 0.01, f"Expert {e}: inactive region not zeroed"
    
    print("✓ Test passed!")


if __name__ == "__main__":
    # For standalone testing without pytest
    print("To run tests, use: pytest test_moe_bmm.py -v")

