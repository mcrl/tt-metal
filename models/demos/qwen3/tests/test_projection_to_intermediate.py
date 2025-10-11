# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import tt_lock


def reference_projection_to_intermediate(
    hidden_states,           # (T, H)
    routed_tokens,           # (E, max_tokens) - concatenated from device-local (E/D) tensors
    num_routed_tokens,       # (E,) - concatenated from device-local (E/D) tensors
    expert_weights,          # (E/D, H, H') per device (sharded)
    device_expert_mapping,   # (E/D,) global expert indices for this device
    top_k,                   # Top-K experts per token
):
    """
    Reference implementation for projection_to_intermediate (for validation after gathering device-local outputs).

    Note: This reference operates on CONCATENATED routing tensors (E experts) for validation purposes.
    The actual TT operation receives SHARDED device-local tensors (E/D experts per device).

    Args:
        hidden_states: (T, H) input token embeddings (replicated)
        routed_tokens: (E, max_tokens) token indices per expert (concatenated from sharded device-local)
        num_routed_tokens: (E,) count of tokens per expert (concatenated from sharded device-local)
        expert_weights: (E/D, H, H') expert weight matrices (sharded per device)
        device_expert_mapping: (E/D,) global expert indices assigned to this device
        top_k: number of experts per token (K)

    Returns:
        output: (T * K, H') projection outputs (sparse, padded)
    """
    T, H = hidden_states.shape
    experts_per_device, _, H_prime = expert_weights.shape

    # Calculate maximum possible output size
    # According to plan: Pre-allocated size is K * T (conservative upper bound for all token-expert pairs)
    max_output_size = top_k * T

    # Pre-allocate output tensor
    output = torch.zeros(max_output_size, H_prime, dtype=torch.bfloat16)

    # Track write position
    write_pos = 0

    # Process experts assigned to this device (expert parallelism)
    for local_expert_idx in range(experts_per_device):
        # Use device-expert mapping to get global expert index
        global_expert_idx = device_expert_mapping[local_expert_idx].item()
        count = num_routed_tokens[global_expert_idx].item()

        if count == 0:
            continue

        # Gather tokens for this expert
        token_indices = routed_tokens[global_expert_idx, :count].long()

        # Validate token indices
        assert torch.all(token_indices >= 0) and torch.all(token_indices < T), \
            f"Invalid token indices for expert {global_expert_idx}: {token_indices}"

        expert_inputs = hidden_states[token_indices]  # (T_e, H)

        # Get expert weights (use LOCAL index for weights!)
        weights = expert_weights[local_expert_idx]  # (H, H')

        # Perform matmul
        expert_output = expert_inputs @ weights  # (T_e, H')

        # Write to output tensor
        output[write_pos:write_pos + count] = expert_output
        write_pos += count

    # Return full pre-allocated output tensor
    # The actual valid data is in output[:write_pos], rest is zero-padded
    return output, write_pos  # Return both output and actual size


@pytest.mark.parametrize("config", [
    {"num_tokens": 8, "top_k": 2, "num_experts": 8, "hidden_dim": 128, "expert_dim": 64},
    {"num_tokens": 128, "top_k": 4, "num_experts": 16, "hidden_dim": 256, "expert_dim": 128},
    {"num_tokens": 256, "top_k": 4, "num_experts": 32, "hidden_dim": 512, "expert_dim": 256},
    {"num_tokens": 256, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    {"num_tokens": 1024, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
    # {"num_tokens": 4096, "top_k": 8, "num_experts": 128, "hidden_dim": 2048, "expert_dim": 768},
])
def test_projection_to_intermediate(mesh_device, config):
    """
    Test projection_to_intermediate operation with various configurations including realistic Qwen3-30B-A3B cases.
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

    # Create input hidden states
    hidden_states_np = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)

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

    # Create device-expert mapping (uniform partitioning)
    # Each device gets a contiguous range of experts
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
    num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping_tt, num_experts
    )

    # Convert routing info to torch for reference
    # prepare_moe_routing_tensors outputs are SHARDED across devices (device-local):
    # After concat: num_routed shape is (D, E/D) where each row has different expert counts
    # After concat: routed_tokens shape is (D*(E/D), max_tokens) = (E, max_tokens)
    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Reshape to get global view (D, E/D) -> (E,)
    num_routed_np = num_routed_torch.flatten()[:num_experts]  # Flatten and take first E values
    routed_tokens_np = routed_tokens_torch[:num_experts]  # Concat already gives (E, max_tokens)

    # Create expert weights (sharded across devices)
    expert_weights_np = torch.randn(num_experts, hidden_dim, expert_dim, dtype=torch.bfloat16)

    # Upload hidden states (replicated)
    # Note: Using ROW_MAJOR_LAYOUT for initial implementation (simpler kernel)
    # Can optimize to TILE_LAYOUT later for better matmul performance
    hidden_states_tt = ttnn.from_torch(
        hidden_states_np,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Upload expert weights (sharded by expert dimension)
    # Shape: (E, H, H') -> each device gets (E/D, H, H')
    # Note: Using ROW_MAJOR_LAYOUT as required by current kernel implementation
    # Can optimize to TILE_LAYOUT later for better matmul performance
    expert_weights_tt = ttnn.from_torch(
        expert_weights_np,  # (E, H, H')
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),  # Shard along expert dimension (E)
    )

    # Create device-expert mapping (uniform partitioning)
    # Each device gets a contiguous range of experts
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
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),  # Each device gets its own mapping
    )

    # Call the operation
    output_tt = ttnn.projection_to_intermediate(
        hidden_states_tt,
        routed_tokens,
        num_routed,
        expert_weights_tt,
        device_expert_mapping_tt,
        top_k
    )

    # Verify output
    output_torch = ttnn.to_torch(output_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Read back device expert mapping to see actual device-to-expert assignment
    device_expert_mapping_readback = ttnn.to_torch(device_expert_mapping_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    expert_weights_readback = ttnn.to_torch(expert_weights_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Calculate output size per device
    max_output_size = top_k * num_tokens

    # Verify all device outputs
    for device_id in range(num_devices):
        # Get device's expert mapping and weights
        device_expert_mapping = device_expert_mapping_readback[device_id]
        device_expert_weights = expert_weights_readback[device_id * experts_per_device:(device_id + 1) * experts_per_device]

        ref_output, actual_output_size = reference_projection_to_intermediate(
            hidden_states_np,
            routed_tokens_np,
            num_routed_np,
            device_expert_weights,
            device_expert_mapping,
            top_k,
        )

        # Extract this device's output
        start_idx = device_id * max_output_size
        end_idx = (device_id + 1) * max_output_size
        device_output = output_torch[start_idx:end_idx]
        device_output_valid = device_output[:actual_output_size]
        ref_output_valid = ref_output[:actual_output_size]

        # Debug output
        print(f"\nDevice {device_id} debug:")
        print(f"  device_expert_mapping: {device_expert_mapping.tolist()}")
        print(f"  actual_output_size: {actual_output_size}")

        if actual_output_size > 0:
            # Compare with reference
            diff = (device_output_valid - ref_output_valid).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            relative_error = (diff / (ref_output_valid.abs() + 1e-6)).mean() * 100

            print(f"  ref_output shape: {ref_output_valid.shape}")
            print(f"  device_output shape: {device_output_valid.shape}")
            print(f"  max_diff: {max_diff:.4f}")
            print(f"  mean_diff: {mean_diff:.4f}")
            print(f"  relative_error: {relative_error:.2f}%")
            print(f"  ref_output[0][:10]: {ref_output_valid[0][:10]}")
            print(f"  device_output[0][:10]: {device_output_valid[0][:10]}")

            # Tolerance based on observed precision with FP32 accumulation in kernel
            # Our kernel uses FP32 accumulation then converts to bfloat16 at the end
            # Typical precision: max_diff ≤0.5, relative_error ≤1%
            assert torch.allclose(device_output_valid, ref_output_valid, atol=0.5, rtol=0.01), \
                f"Output mismatch for device {device_id}:\n" \
                f"Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}\n" \
                f"Relative error: {relative_error:.2f}%\n" \
                f"Expected shape: {ref_output_valid.shape}, Got shape: {device_output_valid.shape}"
        else:
            print(f"  No output for this device (no tokens routed to assigned experts)")

        print(f"✓ Device {device_id} output validated: output_size={actual_output_size}")

    print(
        f"✓ Test structure validated: T={num_tokens}, K={top_k}, E={num_experts}, H={hidden_dim}, H'={expert_dim}, Devices={num_devices}, E/D={experts_per_device}")
