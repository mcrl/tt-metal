# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the moe_down_projection operation.

This operation performs the down projection (step 4) in the MoE layer:
- Takes the activated intermediate states from step 3 (after SwiGLU)
- Projects them back to the model dimension using down_proj weights
- Accumulates outputs from multiple experts into the final output tensor
- Each token position receives contributions from K experts (weighted sum)

The operation handles:
- Expert-parallel computation (each device processes E/D experts)
- Token gathering based on routing information
- Weighted accumulation at output positions
- Proper handling of overlapping token positions (multiple experts per token)

IMPLEMENTATION STATUS:
⏳ TODO - Kernel implementation pending
   - C++ operation layer
   - Program factory with circular buffer setup
   - Dataflow kernel with token gathering and weighted accumulation
   - Python bindings (ttnn.moe_down_projection)
   - Build system integration
"""

import ttnn
import pytest
import torch
import tt_lock


def reference_moe_down_projection(
    intermediate_states,     # (T_d, H') concatenated expert outputs for this device
    routed_tokens,           # (E, T) token indices per expert
    num_routed_tokens,       # (E,) count of tokens per expert
    routed_weights,          # (E, T) routing weights per expert-token pair
    down_proj_weights,       # (E/D, H', H) per device (sharded)
    device_expert_mapping,   # (E/D,) global expert indices for this device
    num_tokens,              # Original number of tokens T
):
    """
    Reference implementation for moe_down_projection.

    Args:
        intermediate_states: (T_d, H') activated intermediate states from step 3
        routed_tokens: (E, T) token indices per expert
        num_routed_tokens: (E,) count of tokens per expert
        routed_weights: (E, T) routing weights for weighted accumulation
        down_proj_weights: (E/D, H', H) down projection weights (sharded)
        device_expert_mapping: (E/D,) global expert indices assigned to this device
        num_tokens: Original number of tokens (T)

    Returns:
        output: (T, H) device's contribution to final MoE output
    """
    T_d, H_prime = intermediate_states.shape
    experts_per_device, _, H = down_proj_weights.shape

    # Initialize output tensor for accumulation
    output = torch.zeros(num_tokens, H, dtype=torch.bfloat16)

    # Track read position in intermediate_states
    read_pos = 0

    # Process experts assigned to this device
    for local_expert_idx in range(experts_per_device):
        # Get global expert index
        global_expert_idx = device_expert_mapping[local_expert_idx].item()
        count = num_routed_tokens[global_expert_idx].item()

        if count == 0:
            continue

        # Get intermediate states for this expert
        expert_intermediate = intermediate_states[read_pos:read_pos + count]  # (T_e, H')
        read_pos += count

        # Get down projection weights (use LOCAL index)
        weights = down_proj_weights[local_expert_idx]  # (H', H)

        # Perform down projection: (T_e, H') @ (H', H) = (T_e, H)
        expert_output = expert_intermediate @ weights

        # Get token indices and routing weights for this expert
        token_indices = routed_tokens[global_expert_idx, :count].long()
        expert_weights = routed_weights[global_expert_idx, :count]  # (T_e,)

        # Accumulate weighted outputs to original token positions
        for i, token_idx in enumerate(token_indices):
            # Apply routing weight and accumulate
            output[token_idx] += expert_weights[i].unsqueeze(-1) * expert_output[i]

    return output


@pytest.mark.parametrize("config", [
    {"num_tokens": 8, "top_k": 2, "num_experts": 8, "model_dim": 128, "expert_dim": 64},
    {"num_tokens": 128, "top_k": 4, "num_experts": 16, "model_dim": 256, "expert_dim": 128},
    {"num_tokens": 256, "top_k": 4, "num_experts": 32, "model_dim": 512, "expert_dim": 256},
    {"num_tokens": 256, "top_k": 8, "num_experts": 128, "model_dim": 2048, "expert_dim": 768},
])
def test_moe_down_projection(mesh_device, config):
    """
    Test moe_down_projection operation with various configurations.

    Note: This test is currently skipped as the kernel implementation is pending.
    """
    pytest.skip("Kernel implementation pending - test framework ready")

    import random

    num_tokens = config["num_tokens"]
    top_k = config["top_k"]
    num_experts = config["num_experts"]
    model_dim = config["model_dim"]
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

    # Calculate total intermediate states size per device
    # Each device processes its assigned experts, T_d = sum of T_e for experts on device
    device_expert_mappings = []
    device_token_counts = []

    for device_id in range(num_devices):
        mapping = torch.arange(
            device_id * experts_per_device,
            (device_id + 1) * experts_per_device,
            dtype=torch.int32
        )
        device_expert_mappings.append(mapping)

        # Count tokens for this device
        T_d = sum(num_routed_np[mapping[i]].item() for i in range(experts_per_device))
        device_token_counts.append(T_d)

    # Create intermediate states per device (simulating output from step 3)
    intermediate_states_per_device = []
    for T_d in device_token_counts:
        intermediate_states_per_device.append(
            torch.randn(T_d, expert_dim, dtype=torch.bfloat16)
        )

    # Stack for upload (shape: D x max(T_d) x H', padded)
    max_T_d = max(device_token_counts)
    intermediate_states_stacked = torch.zeros(num_devices, max_T_d, expert_dim, dtype=torch.bfloat16)
    for device_id, states in enumerate(intermediate_states_per_device):
        intermediate_states_stacked[device_id, :states.shape[0]] = states

    # Upload intermediate states (sharded by device)
    intermediate_states_tt = ttnn.from_torch(
        intermediate_states_stacked,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Create down projection weights (sharded across devices)
    down_proj_weights_np = torch.randn(num_experts, expert_dim, model_dim, dtype=torch.bfloat16)

    # Upload down projection weights (sharded by expert dimension)
    down_proj_weights_tt = ttnn.from_torch(
        down_proj_weights_np,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Create device-expert mapping
    device_expert_mapping_np = torch.stack(device_expert_mappings, dim=0)
    device_expert_mapping_tt = ttnn.from_torch(
        device_expert_mapping_np,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Call the operation
    output_tt = ttnn.moe_down_projection(
        intermediate_states_tt,
        routed_tokens,
        num_routed,
        routed_weights,
        down_proj_weights_tt,
        device_expert_mapping_tt,
        num_tokens
    )

    # Get output from all devices
    output_torch = ttnn.to_torch(output_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Sum across devices to get final output
    final_output = output_torch.sum(dim=0) if num_devices > 1 else output_torch[0]

    # Compute reference for each device and sum
    all_device_outputs = []
    for device_id in range(num_devices):
        device_expert_mapping = device_expert_mapping_np[device_id]
        device_down_proj_weights = down_proj_weights_np[
            device_id * experts_per_device:(device_id + 1) * experts_per_device
        ]
        device_intermediate_states = intermediate_states_per_device[device_id]

        ref_output = reference_moe_down_projection(
            device_intermediate_states,
            routed_tokens_np,
            num_routed_np,
            routed_weights_np,
            device_down_proj_weights,
            device_expert_mapping,
            num_tokens,
        )
        all_device_outputs.append(ref_output)

    # Sum all device outputs for reference
    ref_final_output = sum(all_device_outputs)

    # Compare with reference
    diff = (final_output - ref_final_output).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    relative_error = (diff / (ref_final_output.abs() + 1e-6)).mean() * 100

    print(f"\nTest results:")
    print(f"  max_diff: {max_diff:.4f}")
    print(f"  mean_diff: {mean_diff:.4f}")
    print(f"  relative_error: {relative_error:.2f}%")

    assert torch.allclose(final_output, ref_final_output, atol=0.5, rtol=0.01), \
        f"Output mismatch:\n" \
        f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}\n" \
        f"Relative error: {relative_error:.2f}%\n" \
        f"Expected shape: {ref_final_output.shape}, Got shape: {final_output.shape}"

    print(f"✓ Test passed: T={num_tokens}, K={top_k}, E={num_experts}, "
          f"H={model_dim}, H'={expert_dim}, Devices={num_devices}")
