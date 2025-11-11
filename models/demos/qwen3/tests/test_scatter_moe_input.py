# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch


def reference_scatter_moe_input(input_hidden_state, num_routed_tokens, routed_tokens):
    """
    Reference implementation for scatter_moe_input operation.

    Args:
        input_hidden_state: (T, H) tensor of input token embeddings
        num_routed_tokens: (E/D,) 1D tensor of token counts per expert
        routed_tokens: (E/D, T) tensor of token indices per expert

    Returns:
        output: (E/D, T, H) tensor with tokens scattered by expert assignment
    """
    T, H = input_hidden_state.shape
    # Only handle 1D (E/D) shape
    num_local_experts = num_routed_tokens.shape[0]
    _, max_tokens = routed_tokens.shape

    # Initialize output with zeros
    output = torch.zeros((num_local_experts, max_tokens, H), dtype=input_hidden_state.dtype)

    # For each local expert
    for expert_idx in range(num_local_experts):
        t_e = num_routed_tokens[expert_idx].item()

        # Gather assigned tokens
        for i in range(t_e):
            token_idx = routed_tokens[expert_idx, i].item()
            # Gather: output[e, i, :] = input[token_idx, :]
            output[expert_idx, i, :] = input_hidden_state[token_idx, :]

        # Remaining positions [t_e, T) are already zero-initialized

    return output


@pytest.mark.parametrize("num_tokens,top_k,num_experts,hidden_dim", [
    (32, 4, 8, 256),
    (32, 4, 32, 128),
    (64, 4, 32, 256),
    (128, 4, 32, 256),
    (128, 8, 32, 512),
    (256, 4, 128, 256),
    (1024, 8, 128, 2048),
])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_scatter_moe_input(mesh_device, num_tokens, top_k, num_experts, hidden_dim):
    """
    Test scatter_moe_input operation against PyTorch reference.
    Validates correctness, shape, layout, and dtype.
    """
    import random

    # Skip test if top_k > num_experts
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

    random.seed(42)
    torch.manual_seed(42)

    # Create test input
    input_hidden_state = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Create expert selection (no duplicate experts per token)
    selected_experts_np = torch.zeros((num_tokens, top_k), dtype=torch.int32)
    for t in range(num_tokens):
        experts = torch.randperm(num_experts)[:top_k]
        selected_experts_np[t] = experts

    routing_weights_np = torch.rand(num_tokens, top_k, dtype=torch.bfloat16)
    routing_weights_np = routing_weights_np / routing_weights_np.sum(dim=1, keepdim=True)

    # Create device-expert mapping (uniform partitioning)
    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices
    device_expert_mappings = []
    for device_id in range(num_devices):
        mapping = torch.arange(
            device_id * experts_per_device,
            (device_id + 1) * experts_per_device,
            dtype=torch.int32
        )
        device_expert_mappings.append(mapping)

    device_expert_mapping_np = torch.stack(device_expert_mappings, dim=0)

    # Prepare routing tensors using existing operation
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

    device_expert_mapping = ttnn.from_torch(
        device_expert_mapping_np,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Get routing tensors
    num_routed_tokens, routed_tokens, routed_token_weights, token_idx_map = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping, num_experts
    )

    # Convert input to ttnn tensor
    input_hidden_state_tt = ttnn.from_torch(
        input_hidden_state,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run scatter_moe_input operation
    output_tt = ttnn.scatter_moe_input(
        input_hidden_state_tt,
        num_routed_tokens,
        routed_tokens,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Convert outputs back to torch for comparison
    output_torch = ttnn.to_torch(
        output_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    # Extract per-device outputs for reference comparison
    num_routed_torch = ttnn.to_torch(
        num_routed_tokens,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    routed_tokens_torch = ttnn.to_torch(
        routed_tokens,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    # Compute reference output for all devices
    reference_output = reference_scatter_moe_input(
        input_hidden_state,
        num_routed_torch,
        routed_tokens_torch
    )

    # Compare outputs
    print(f"\nTest config: tokens={num_tokens}, top_k={top_k}, experts={num_experts}, hidden_dim={hidden_dim}")
    print(f"Output shape (TT): {output_torch.shape}")
    print(f"Output shape (Ref): {reference_output.shape}")

    # Verify shape dimensions
    num_devices = mesh_device.get_num_devices()
    experts_per_device = num_experts // num_devices
    expected_shape = (num_devices * experts_per_device, num_tokens, hidden_dim)

    assert output_torch.shape == reference_output.shape, \
        f"Shape mismatch: {output_torch.shape} vs {reference_output.shape}"
    assert output_torch.shape == expected_shape, \
        f"Shape mismatch: {output_torch.shape} vs expected {expected_shape}"

    # Verify layout and dtype
    assert output_tt.layout == ttnn.ROW_MAJOR_LAYOUT, \
        f"Layout should be ROW_MAJOR, got {output_tt.layout}"
    assert output_tt.dtype == ttnn.bfloat16, \
        f"Dtype should be bfloat16, got {output_tt.dtype}"

    # Verify correctness per expert
    total_experts = num_routed_torch.shape[0]
    for expert_idx in range(total_experts):
        t_e = num_routed_torch[expert_idx].item()  # num_routed_torch is 1D (E,)

        # Check assigned tokens (non-zero region)
        if t_e > 0:
            expert_output = output_torch[expert_idx, :t_e, :]
            expert_ref = reference_output[expert_idx, :t_e, :]
            expert_diff = torch.abs(expert_output - expert_ref).max().item()

            assert expert_diff < 0.01, \
                f"Expert {expert_idx}: Large difference in assigned tokens: {expert_diff}"

        # Note: Padding region [t_e, num_tokens) is not initialized and not checked

    print("Test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
