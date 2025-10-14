# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import tt_lock


def reference_projection_to_output(
    combined_activations,     # (E/D, T, H') - pre-allocated tensor with space for all tokens
    token_idx_map,            # (E/D, max_tokens) - mapping from expert-local to global token index
    num_routed_tokens,        # (E/D,) - count per local expert (as 1D array)
    routed_token_weights,     # (E/D, max_tokens) - routing weights
    expert_weights,           # (E/D, H', H) per device (sharded)
    num_tokens,               # T - total number of tokens
    hidden_dim,               # H - hidden dimension
):
    """
    Reference implementation for projection_to_output.

    Args:
        combined_activations: (E/D, T, H') combined gate & up projections after SiLU
        token_idx_map: (E/D, max_tokens) mapping from expert-local index to global token index
        num_routed_tokens: (E/D,) count of tokens per local expert
        routed_token_weights: (E/D, max_tokens) routing weights
        expert_weights: (E/D, H', H) down projection weight matrices (sharded per device)
        num_tokens: T - total number of tokens
        hidden_dim: H - output hidden dimension

    Returns:
        output: (T, H) final MoE output with accumulated expert contributions
    """
    experts_per_device, _, H_prime = combined_activations.shape
    _, _, H = expert_weights.shape

    # Pre-allocate output tensor (will accumulate results)
    output = torch.zeros(num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Process each local expert
    for local_expert_idx in range(experts_per_device):
        # Handle both 1D and 2D num_routed_tokens
        if num_routed_tokens.dim() > 1:
            count = num_routed_tokens[local_expert_idx, 0].item()
        else:
            count = num_routed_tokens[local_expert_idx].item()

        if count == 0:
            continue

        # Get routing weights for this expert
        routing_weights = routed_token_weights[local_expert_idx, :count]

        # Get token indices for this expert (using token_idx_map)
        token_indices = token_idx_map[local_expert_idx, :count].long()

        # Validate token indices
        assert torch.all(token_indices >= 0) and torch.all(token_indices < num_tokens), \
            f"Invalid token indices for local expert {local_expert_idx}: {token_indices}"

        # Get expert's combined activations from combined_activations tensor
        # Use token_idx_map to get the right rows
        expert_activations = combined_activations[local_expert_idx, token_indices]  # (count, H')

        # Get expert down_proj weights
        weights = expert_weights[local_expert_idx]  # (H', H)

        # Perform matmul
        expert_output = expert_activations @ weights  # (count, H)

        # Multiply by routing weights
        weighted_output = expert_output * routing_weights.unsqueeze(1)  # (count, H)

        # ACCUMULATE to output tensor at appropriate token positions
        for i, token_idx in enumerate(token_indices):
            output[token_idx] += weighted_output[i]

    return output


@pytest.mark.parametrize("config", [
    {"num_tokens": 8, "top_k": 2, "num_experts": 8, "hidden_dim": 128, "expert_dim": 128},
    # {"num_tokens": 8, "top_k": 2, "num_experts": 8, "hidden_dim": 256, "expert_dim": 128},
    # {"num_tokens": 128, "top_k": 4, "num_experts": 16, "hidden_dim": 256, "expert_dim": 256},
    # {"num_tokens": 32, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # {"num_tokens": 256, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # Realistic Qwen3-30B-A3B configurations:
    # {"num_tokens": 1024, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # {"num_tokens": 4096, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
])
def test_projection_to_output(mesh_device, config):
    """
    Test projection_to_output operation (Step 4) with various configurations.
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

    # Create routing information - this determines which experts get which tokens
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
    device_expert_mapping_np = torch.stack(device_expert_mappings, dim=0)  # (D, E/D)
    device_expert_mapping_tt = ttnn.from_torch(
        device_expert_mapping_np,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Get routed tokens tensor (now device-local, sharded by experts)
    # Returns 4 tensors: num_routed_tokens, routed_tokens, routed_token_weights, token_idx_map
    num_routed, routed_tokens, routed_weights, token_idx_map = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping_tt, num_experts
    )

    # Convert routing info to torch for reference (sharded outputs)
    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_weights_torch = ttnn.to_torch(routed_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Get global view - num_routed is 2D (E, 1), squeeze to 1D
    num_routed_np = num_routed_torch[:num_experts, 0]  # Extract first column, shape (E,)
    routed_tokens_np = routed_tokens_torch[:num_experts]  # Concat gives (E, max_tokens)
    routed_weights_np = routed_weights_torch[:num_experts]  # Concat gives (E, max_tokens)

    # Create down projection expert weights (sharded across devices)
    # Shape: (E, H', H) - note H' comes first for down projection
    down_proj_weights_np = torch.randn(num_experts, expert_dim, hidden_dim, dtype=torch.bfloat16)

    # Calculate actual size (sum of all T_e for experts on each device)
    actual_sizes_per_device = []
    for device_id in range(num_devices):
        device_mapping = device_expert_mappings[device_id]
        size = sum(num_routed_np[idx].item() for idx in device_mapping)
        actual_sizes_per_device.append(size)

    # Create simulated combined activations from Step 3 (output of gate * up projections)
    # Shape: (E/D, T, H') - each device has space for all tokens per expert
    # In real flow, this would come from projection_to_intermediate for gate and up, then element-wise multiply
    combined_activations_np = torch.randn(experts_per_device, num_tokens, expert_dim, dtype=torch.bfloat16)

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

    # Call the actual TTNN operation
    output_tt = ttnn.projection_to_output(
        combined_activations_tt,
        token_idx_map,
        routed_tokens,
        num_routed,
        routed_weights,  # Pass routing weights for multiplication
        down_proj_weights_tt,
        num_tokens,
        top_k,
    )

    print(f"\nTest config: T={num_tokens}, K={top_k}, E={num_experts}, H={hidden_dim}, H'={expert_dim}, devices={num_devices}")

    # Read back the TTNN output - each device has its partial output
    # The operation outputs per-device partial results that need allreduce (Step 5)
    output_torch = ttnn.to_torch(output_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Read back token_idx_map for reference
    token_idx_map_torch = ttnn.to_torch(token_idx_map, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    token_idx_map_np = token_idx_map_torch[:num_experts]  # (E, max_tokens)

    # Read back tensors for reference computation
    down_proj_weights_readback = ttnn.to_torch(down_proj_weights_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Test reference implementation for each device
    all_device_outputs = []
    for device_id in range(num_devices):
        # Get device's routing info and weights
        start_expert = device_id * experts_per_device
        end_expert = (device_id + 1) * experts_per_device
        
        device_num_routed = num_routed_np[start_expert:end_expert]
        device_token_idx_map = token_idx_map_np[start_expert:end_expert]
        device_routed_weights = routed_weights_np[start_expert:end_expert]
        device_expert_weights = down_proj_weights_readback[start_expert:end_expert]

        # Get combined activations for this device
        device_combined_activations = combined_activations_np

        ref_output = reference_projection_to_output(
            device_combined_activations,
            device_token_idx_map,
            device_num_routed,
            device_routed_weights,
            device_expert_weights,
            num_tokens,
            hidden_dim,
        )

        all_device_outputs.append(ref_output)

        print(output_torch[device_id][:, :5])
        print(ref_output[:, :5])

        # Print first few output values for this device
        print(f"Device {device_id} ref output[0, :8]: {ref_output[0, :8].tolist()}")

    print(output_torch.shape, ref_output.shape)
    return

    # Step 5: Simulate allreduce - sum outputs from all devices
    final_output = torch.zeros_like(all_device_outputs[0])
    for device_output in all_device_outputs:
        final_output += device_output

    # Verify output properties
    assert final_output.shape == (num_tokens, hidden_dim), \
        f"Output shape mismatch: expected ({num_tokens}, {hidden_dim}), got {final_output.shape}"
    assert final_output.norm() > 0, "Output is all zeros - no expert contributions"

    # Combine TTNN device outputs (simulate allreduce)
    # Output shape from ConcatMeshToTensor is (num_devices * experts_per_device, num_tokens, hidden_dim)
    output_reshaped = output_torch.view(num_devices, experts_per_device, num_tokens, hidden_dim)
    ttnn_combined = torch.zeros(num_tokens, hidden_dim, dtype=output_torch.dtype)
    for device_id in range(num_devices):
        # Sum over local experts for this device
        device_output = output_reshaped[device_id].sum(dim=0)  # Sum over experts_per_device dimension
        print(f"Device {device_id} ttnn output[0, :8]: {device_output[0, :8].tolist()}")
        ttnn_combined += device_output

    # Compare combined result with reference
    diff = (ttnn_combined - final_output).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    if torch.allclose(ttnn_combined, final_output, atol=0.5, rtol=0.1):
        print(f"✓ Test passed - max_diff: {max_diff:.4f}, mean_diff: {mean_diff:.4f}")
    else:
        print(f"✗ Test failed - max_diff: {max_diff:.4f}, mean_diff: {mean_diff:.4f}")
        print(f"  TTNN: mean={ttnn_combined.mean():.4f}, std={ttnn_combined.std():.4f}")
        print(f"  Reference: mean={final_output.mean():.4f}, std={final_output.std():.4f}")
        assert False, "TTNN output does not match reference implementation"
