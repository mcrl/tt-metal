# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the moe_down_projection operation.

This operation performs batched matrix multiplication for MoE down projection (Step 4):
- Takes combined activations from Step 3 (after SiLU and elementwise multiply)
- Each expert processes its assigned tokens with down_proj weights
- Performs batched matmul: (T_e × H') @ (H' × H) = T_e × H
- Multiplies each result by corresponding routing weight
- Accumulates results to final output tensor (T × H)

Key differences from expert_projection (Steps 1&2):
- Input is combined activations (H' dimension) not hidden states (H dimension)
- Weights are H' × H (down projection) not H × H' (up/gate projection)
- Results are multiplied by routing weights before accumulation
- Output is accumulated to final T × H tensor (not separate per-expert outputs)

IMPLEMENTATION STATUS:
✅ IMPLEMENTED - Kernel fixed and tested
   - C++ operation layer with validation (supports both 2D and 3D device_expert_mapping)
   - Program factory with circular buffer setup
   - Dataflow kernel with output initialization, routing weight application, and accumulation
   - Python bindings (ttnn.moe_down_projection)
   - Build system integration

KERNEL FIXES (2025-10-07):
1. Fixed NaN issue by initializing output tensor to zeros before processing
2. Fixed multi-device support: validation now accepts both (1, E/D) and (1, 1, E/D) shapes
3. Fixed mapping page size calculation for different tensor ranks
4. Corrected accumulation logic: routing_weight * (activation @ weights) + previous_output
5. Optimized inner loop to read weight rows once per expert_dim (previously read per output element)

TEST STATUS:
✅ config0: PASS (num_tokens=8, hidden_dim=128, expert_dim=64) - ~8.6s
⏱️  config1: TIMEOUT (num_tokens=128, hidden_dim=256, expert_dim=128) - needs further optimization
⏱️  config2: TIMEOUT (num_tokens=256, hidden_dim=2048, expert_dim=768) - needs further optimization

PERFORMANCE NOTES:
- Current implementation reads weight rows one at a time (expert_dim reads per token)
- For config1: 128 weight reads per token, each 512 bytes = 64KB per token
- For config2: 768 weight reads per token, each 4KB = 3MB per token
- Further optimization needed: batch weight reads, use compute kernels, or multi-core distribution
"""

import ttnn
import pytest
import torch
import tt_lock


def reference_moe_down_projection(
    combined_activations,     # (T * K, H') actual valid data in pre-allocated tensor
    actual_size,               # Actual number of valid rows in combined_activations
    routed_tokens,            # (E, max_tokens)
    num_routed_tokens,        # (E,)
    routed_token_weights,     # (E, max_tokens) routing weights
    expert_weights,           # (E/D, H', H) per device (sharded)
    device_expert_mapping,    # (E/D,) global expert indices for this device
    num_tokens,               # T - total number of tokens
    hidden_dim,               # H - hidden dimension
):
    """
    Reference implementation for moe_down_projection (Step 4).

    Args:
        combined_activations: (T * K, H') combined gate & up projections after SiLU
        actual_size: Number of valid rows in combined_activations
        routed_tokens: (E, max_tokens) token indices per expert
        num_routed_tokens: (E,) count of tokens per expert
        routed_token_weights: (E, max_tokens) routing weights per token-expert pair
        expert_weights: (E/D, H', H) down projection weight matrices (sharded per device)
        device_expert_mapping: (E/D,) global expert indices assigned to this device
        num_tokens: T - total number of tokens
        hidden_dim: H - output hidden dimension

    Returns:
        output: (T, H) final MoE output with accumulated expert contributions
    """
    experts_per_device, H_prime, H = expert_weights.shape

    # Pre-allocate output tensor (will accumulate results)
    output = torch.zeros(num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Track read position in combined activations
    read_pos = 0

    # Process experts assigned to this device (expert parallelism)
    for local_expert_idx in range(experts_per_device):
        # Use device-expert mapping to get global expert index
        global_expert_idx = device_expert_mapping[local_expert_idx].item()
        count = num_routed_tokens[global_expert_idx].item()

        if count == 0:
            continue

        # Get token indices and routing weights for this expert
        token_indices = routed_tokens[global_expert_idx, :count].long()
        routing_weights = routed_token_weights[global_expert_idx, :count]

        # Validate token indices
        assert torch.all(token_indices >= 0) and torch.all(token_indices < num_tokens), \
            f"Invalid token indices for expert {global_expert_idx}: {token_indices}"

        # Get expert's combined activations from Step 3
        expert_activations = combined_activations[read_pos:read_pos + count]  # (T_e, H')
        read_pos += count

        # Get expert down_proj weights (use LOCAL index for weights!)
        weights = expert_weights[local_expert_idx]  # (H', H)

        # Perform matmul
        expert_output = expert_activations @ weights  # (T_e, H)

        # Multiply by routing weights
        # routing_weights shape: (T_e,), need to broadcast to (T_e, 1)
        weighted_output = expert_output * routing_weights.unsqueeze(1)  # (T_e, H)

        # ACCUMULATE to output tensor at appropriate token positions
        # Multiple experts contribute to the same token, so we accumulate
        for i, token_idx in enumerate(token_indices):
            output[token_idx] += weighted_output[i]

    # Verify we consumed all valid activations
    assert read_pos == actual_size, \
        f"Did not consume all activations: read {read_pos}, expected {actual_size}"

    return output


@pytest.mark.parametrize("config", [
    {"num_tokens": 8, "top_k": 2, "num_experts": 8, "hidden_dim": 128, "expert_dim": 64},
    {"num_tokens": 128, "top_k": 4, "num_experts": 16, "hidden_dim": 256, "expert_dim": 128},
    {"num_tokens": 256, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # Realistic Qwen3-30B-A3B configurations:
    # {"num_tokens": 1024, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # {"num_tokens": 4096, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
])
def test_moe_down_projection(mesh_device, config):
    """
    Test moe_down_projection operation (Step 4) with various configurations.
    """
    import random

    num_tokens = config["num_tokens"]
    top_k = config["top_k"]
    num_experts = config["num_experts"]
    hidden_dim = config["hidden_dim"]
    expert_dim = config["expert_dim"]

    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

    random.seed(42)
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()

    # Ensure experts can be evenly distributed
    if num_experts % num_devices != 0:
        pytest.skip(f"num_experts ({num_experts}) must be divisible by num_devices ({num_devices})")

    experts_per_device = num_experts // num_devices

    # Create simulated combined activations from Step 3
    # In real use, this would come from SiLU(gate_proj) * up_proj
    max_output_size = top_k * num_tokens
    combined_activations_np = torch.randn(max_output_size, expert_dim, dtype=torch.bfloat16)

    # Create routing information
    selected_experts_np = torch.zeros((num_tokens, top_k), dtype=torch.int32)
    for t in range(num_tokens):
        experts = torch.randperm(num_experts)[:top_k]
        selected_experts_np[t] = experts

    routing_weights_np = torch.rand(num_tokens, top_k, dtype=torch.bfloat16)
    routing_weights_np = routing_weights_np / routing_weights_np.sum(dim=1, keepdim=True)

    # Prepare routing tensors
    selected_experts = ttnn.from_torch(
        selected_experts_np,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    routing_weights = ttnn.from_torch(
        routing_weights_np,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Get routed tokens tensor
    num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, num_experts
    )

    # Convert routing info to torch for reference
    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_weights_torch = ttnn.to_torch(routed_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Extract first device data (all devices have identical copies)
    num_routed_np = num_routed_torch[0, :num_experts]
    routed_tokens_np = routed_tokens_torch[:num_experts]
    routed_weights_np = routed_weights_torch[:num_experts]

    # Create down projection expert weights (sharded across devices)
    # Shape: (E, H', H) - note H' comes first for down projection
    down_proj_weights_np = torch.randn(num_experts, expert_dim, hidden_dim, dtype=torch.bfloat16)

    # Upload combined activations (replicated for now, could be sharded in practice)
    combined_activations_tt = ttnn.from_torch(
        combined_activations_np,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Upload down projection weights (sharded by expert dimension)
    down_proj_weights_tt = ttnn.from_torch(
        down_proj_weights_np,  # (E, H', H)
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR required by operation
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),  # Shard along expert dimension
    )

    # Create device-expert mapping (uniform partitioning)
    device_expert_mappings = []
    for device_id in range(num_devices):
        mapping = torch.arange(
            device_id * experts_per_device,
            (device_id + 1) * experts_per_device,
            dtype=torch.int32
        )
        device_expert_mappings.append(mapping)

    # Stack and upload device-expert mappings (sharded)
    # Each device needs shape (1, E/D) for its mapping
    device_expert_mapping_np = torch.stack([m.unsqueeze(0) for m in device_expert_mappings], dim=0)  # (D, 1, E/D)
    device_expert_mapping_tt = ttnn.from_torch(
        device_expert_mapping_np,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Calculate actual size (sum of all T_e for experts on each device)
    actual_sizes_per_device = []
    for device_id in range(num_devices):
        device_mapping = device_expert_mappings[device_id]
        size = sum(num_routed_np[idx].item() for idx in device_mapping)
        actual_sizes_per_device.append(size)

    # Call the actual TTNN operation
    output_tt = ttnn.moe_down_projection(
        combined_activations_tt,
        routed_tokens,
        num_routed,
        routed_weights,  # Pass routing weights for multiplication
        down_proj_weights_tt,
        device_expert_mapping_tt,
        num_tokens,
        top_k,
    )

    print(f"\n✓ Test structure prepared: T={num_tokens}, K={top_k}, E={num_experts}, H={hidden_dim}, H'={expert_dim}")

    # Debug: Print input tensor information
    print(f"\nInput tensor shapes before operation:")
    print(f"  combined_activations: {combined_activations_tt.shape}")
    print(f"  routed_tokens: {routed_tokens.shape}")
    print(f"  num_routed: {num_routed.shape}")
    print(f"  routed_weights: {routed_weights.shape}")
    print(f"  down_proj_weights: {down_proj_weights_tt.shape}")
    print(f"  device_expert_mapping: {device_expert_mapping_tt.shape}")

    # Read back the TTNN output - each device has its partial output
    # The operation outputs per-device partial results that need allreduce (Step 5)
    output_torch = ttnn.to_torch(output_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Read back tensors for reference computation
    device_expert_mapping_readback = ttnn.to_torch(device_expert_mapping_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    down_proj_weights_readback = ttnn.to_torch(down_proj_weights_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Test reference implementation for each device
    all_device_outputs = []
    for device_id in range(num_devices):
        # Get device's expert mapping and weights
        # Shape of mapping is now (D, 1, E/D), so we need [device_id, 0]
        device_expert_mapping = device_expert_mapping_readback[device_id, 0] if device_expert_mapping_readback.dim() == 3 else device_expert_mapping_readback[device_id]
        device_expert_weights = down_proj_weights_readback[device_id * experts_per_device:(device_id + 1) * experts_per_device]

        # Simulate combined activations for this device (would come from Step 3)
        # In reality, each device would have its own activations from its experts
        actual_size = actual_sizes_per_device[device_id]
        device_combined_activations = combined_activations_np[:actual_size]

        ref_output = reference_moe_down_projection(
            device_combined_activations,
            actual_size,
            routed_tokens_np,
            num_routed_np,
            routed_weights_np,
            device_expert_weights,
            device_expert_mapping,
            num_tokens,
            hidden_dim,
        )

        all_device_outputs.append(ref_output)

        # Debug output
        print(f"\nDevice {device_id} debug:")
        print(f"  device_expert_mapping: {device_expert_mapping.tolist()}")
        print(f"  actual_size: {actual_size}")
        print(f"  output shape: {ref_output.shape}")
        print(f"  output norm: {ref_output.norm():.4f}")


    # Step 5: Simulate allreduce - sum outputs from all devices
    final_output = torch.zeros_like(all_device_outputs[0])
    for device_output in all_device_outputs:
        final_output += device_output

    print(f"\n✓ Final output after allreduce:")
    print(f"  Shape: {final_output.shape}")
    print(f"  Norm: {final_output.norm():.4f}")
    print(f"  Mean: {final_output.mean():.4f}")
    print(f"  Std: {final_output.std():.4f}")

    # Verify output properties
    assert final_output.shape == (num_tokens, hidden_dim), \
        f"Output shape mismatch: expected ({num_tokens}, {hidden_dim}), got {final_output.shape}"

    # Check that output is non-zero (experts contributed)
    assert final_output.norm() > 0, "Output is all zeros - no expert contributions"

    # Compare TTNN output with reference implementation
    print(f"\nTTNN Output Analysis:")
    print(f"  Raw output shape: {output_torch.shape}")
    print(f"  Raw output dtype: {output_torch.dtype}")

    # The output is concatenated from all devices: [num_devices, num_tokens, hidden_dim]
    # or [num_devices * num_tokens, hidden_dim] depending on concat
    if output_torch.dim() == 3:
        # Shape is [num_devices, num_tokens, hidden_dim]
        print(f"  Output is 3D: {output_torch.shape}")
        # Each device has partial results that need to be summed (allreduce)
        ttnn_combined = torch.zeros(num_tokens, hidden_dim, dtype=output_torch.dtype)
        for device_id in range(num_devices):
            device_output = output_torch[device_id]
            print(f"  Device {device_id} partial output norm: {device_output.norm():.4f}")
            ttnn_combined += device_output
    elif output_torch.dim() == 2 and output_torch.shape[0] == num_devices * num_tokens:
        # Shape is [num_devices * num_tokens, hidden_dim]
        print(f"  Output is 2D concatenated: {output_torch.shape}")
        # Reshape to [num_devices, num_tokens, hidden_dim]
        output_reshaped = output_torch.view(num_devices, num_tokens, hidden_dim)
        # Sum across devices (allreduce)
        ttnn_combined = torch.zeros(num_tokens, hidden_dim, dtype=output_torch.dtype)
        for device_id in range(num_devices):
            device_output = output_reshaped[device_id]
            print(f"  Device {device_id} partial output norm: {device_output.norm():.4f}")
            ttnn_combined += device_output
    else:
        # Single device or unexpected shape
        print(f"  Output shape suggests single device or unexpected format")
        ttnn_combined = output_torch

    print(f"\nTTNN combined output (after simulated allreduce):")
    print(f"  Shape: {ttnn_combined.shape}")
    print(f"  Norm: {ttnn_combined.norm():.4f}")
    print(f"  Reference norm: {final_output.norm():.4f}")

    # Compare combined result with reference
    if torch.allclose(ttnn_combined, final_output, atol=0.5, rtol=0.1):
        print(f"  ✓ TTNN output matches reference implementation")
    else:
        diff = (ttnn_combined - final_output).abs()
        print(f"  ⚠ TTNN output differs from reference")
        print(f"    Max difference: {diff.max():.4f}")
        print(f"    Mean difference: {diff.mean():.4f}")
        # Print some debug info
        print(f"    TTNN mean: {ttnn_combined.mean():.4f}, std: {ttnn_combined.std():.4f}")
        print(f"    Reference mean: {final_output.mean():.4f}, std: {final_output.std():.4f}")
        assert torch.allclose(ttnn_combined, final_output, atol=0.5, rtol=0.1), \
            "TTNN output does not match reference implementation"

    print(f"\n✓ Test passed: moe_down_projection implementation verified")
    print(f"  Each device processes {experts_per_device} experts")
    print(f"  Final output shape: ({num_tokens}, {hidden_dim})")