# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the prepare_moe_mapping_tensor operation.

This operation prepares a sparse mapping tensor for Mixture of Experts (MoE) layers
by scattering routing weights to their corresponding expert positions.

Key behaviors:
- Each token selects top_k unique experts (no duplicates)
- Routing weights are scattered to create a sparse (num_tokens, num_experts) tensor
- Unselected experts have weight 0.0
- The operation is fully functional with all tests passing
"""

import ttnn
import pytest


@pytest.mark.parametrize("num_tokens", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("top_k", [2, 3, 4])
@pytest.mark.parametrize("num_experts", [8, 16, 32])
def test_prepare_moe_mapping_tensor(mesh_device, num_tokens, top_k, num_experts):
    """
    Test prepare_moe_mapping_tensor on mesh_device.
    Ensures no duplicate expert selection per token.

    Each device in the mesh performs the same duplicate operation.
    """
    import torch
    import random

    # Skip test if top_k > num_experts (can't select more experts than available)
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

    random.seed(42)
    torch.manual_seed(42)

    # Create test data with no duplicate expert selection per token
    selected_experts_np = torch.zeros((num_tokens, top_k), dtype=torch.int32)
    for t in range(num_tokens):
        # Select top_k unique experts for each token
        experts = torch.randperm(num_experts)[:top_k]
        selected_experts_np[t] = experts

    routing_weights_np = torch.rand(num_tokens, top_k, dtype=torch.bfloat16)
    # Normalize weights per token
    routing_weights_np = routing_weights_np / routing_weights_np.sum(dim=1, keepdim=True)

    # Upload to mesh_device (duplicate on all devices)
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

    # Call the operation on mesh
    output = ttnn.prepare_moe_mapping_tensor(selected_experts, routing_weights, num_experts)

    # Verify output shape
    output_shape = output.shape
    assert output_shape[0] == num_tokens, f"Expected num_tokens={num_tokens}, got {output_shape[0]}"
    assert output_shape[1] >= num_experts, f"Expected num_experts>={num_experts}, got {output_shape[1]}"

    # Convert to PyTorch for verification (from all devices)
    output_torch_all = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Each device should have identical output (replicated)
    num_devices = mesh_device.get_num_devices()
    chunk_size = output_torch_all.shape[0] // num_devices

    # Verify all devices produced the same result
    for device_id in range(1, num_devices):
        first_device_output = output_torch_all[:chunk_size, :num_experts]
        current_device_output = output_torch_all[device_id * chunk_size:(device_id + 1) * chunk_size, :num_experts]
        assert torch.allclose(first_device_output, current_device_output, atol=1e-2), \
            f"Device {device_id} output differs from device 0"

    # Use output from first device for validation
    output_torch = output_torch_all[:num_tokens, :num_experts]

    # Build expected output
    expected = torch.zeros(num_tokens, num_experts, dtype=torch.bfloat16)
    for t in range(num_tokens):
        for k in range(top_k):
            expert_idx = selected_experts_np[t, k].item()
            # Since no duplicates, we can directly set the weight
            expected[t, expert_idx] = routing_weights_np[t, k]

    # Verify correctness
    assert torch.allclose(
        output_torch, expected, atol=1e-2
    ), f"Output mismatch:\nExpected:\n{expected}\nGot:\n{output_torch}"

    print(f"✓ Test passed: num_tokens={num_tokens}, top_k={top_k}, num_experts={num_experts}")
