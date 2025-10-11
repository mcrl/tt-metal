# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the prepare_moe_routing_tensors operation.

This operation converts sparse MoE expert selection into efficient routing tensors
for expert-parallel computation.

Key behaviors:
- Creates three tensors: num_routed_tokens, routed_tokens, routed_token_weights
- Each token selects top_k unique experts (no duplicates)
- Output tensors are padded to rectangular shape for efficient processing
"""

import ttnn
import pytest
import torch
import tt_lock


def reference_prepare_moe_routing_tensors(selected_experts, routing_weights, num_experts):
    """
    Reference implementation for prepare_moe_routing_tensors.

    Args:
        selected_experts: (T, K) tensor of expert indices
        routing_weights: (T, K) tensor of routing weights
        num_experts: total number of experts

    Returns:
        - num_routed_tokens: (E,) count of tokens per expert
        - routed_tokens: (E, max_tokens) token indices per expert
        - routed_token_weights: (E, max_tokens) weights per expert
    """
    num_tokens, top_k = selected_experts.shape
    max_tokens_per_expert = num_tokens

    # Initialize outputs
    num_routed_tokens = torch.zeros(num_experts, dtype=torch.int32)
    routed_tokens = torch.full((num_experts, max_tokens_per_expert), -1, dtype=torch.int32)
    routed_token_weights = torch.zeros((num_experts, max_tokens_per_expert), dtype=torch.bfloat16)

    # Build routing
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_idx = selected_experts[token_idx, k].item()
            weight = routing_weights[token_idx, k]

            if expert_idx < num_experts:
                count = num_routed_tokens[expert_idx].item()
                if count < max_tokens_per_expert:
                    routed_tokens[expert_idx, count] = token_idx
                    routed_token_weights[expert_idx, count] = weight
                    num_routed_tokens[expert_idx] = count + 1

    return num_routed_tokens, routed_tokens, routed_token_weights


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("num_experts", [8, 32, 128])
def test_prepare_moe_routing_tensors(mesh_device, num_tokens, top_k, num_experts):
    """
    Test prepare_moe_routing_tensors on mesh_device.
    """
    import random

    # Skip test if top_k > num_experts
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

    # Get reference output
    ref_num_routed, ref_routed_tokens, ref_routed_weights = reference_prepare_moe_routing_tensors(
        selected_experts_np, routing_weights_np, num_experts
    )

    # Upload to mesh_device
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

    # Call the operation
    num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, num_experts
    )

    # Verify output shapes
    assert num_routed.shape[0] == 1  # Single row tensor
    assert num_routed.shape[1] == num_experts  # Exact size (no padding)
    assert routed_tokens.shape[0] == num_experts  # Exact size (no padding)
    assert routed_tokens.shape[1] == num_tokens  # max_tokens_per_expert
    assert routed_weights.shape == routed_tokens.shape

    # Convert to PyTorch for verification
    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_weights_torch = ttnn.to_torch(routed_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Each device should have identical output (replicated)
    num_devices = mesh_device.get_num_devices()

    # Extract first device output
    num_routed_torch = num_routed_torch[0, :]  # First row, all values
    routed_tokens_torch = routed_tokens_torch[:num_experts]
    routed_weights_torch = routed_weights_torch[:num_experts]

    # Verify num_routed_tokens
    assert torch.equal(num_routed_torch, ref_num_routed), \
        f"num_routed_tokens mismatch:\nExpected:\n{ref_num_routed}\nGot:\n{num_routed_torch}"

    # Verify routed_tokens and routed_weights
    for expert_idx in range(num_experts):
        count = ref_num_routed[expert_idx].item()

        if count > 0:
            # Check valid tokens (up to count)
            actual_tokens = routed_tokens_torch[expert_idx, :count]
            expected_tokens = ref_routed_tokens[expert_idx, :count]

            # Sort tokens for comparison (order may differ)
            actual_sorted = torch.sort(actual_tokens)[0]
            expected_sorted = torch.sort(expected_tokens)[0]

            assert torch.equal(actual_sorted, expected_sorted), \
                f"Expert {expert_idx} token mismatch:\nExpected:\n{expected_sorted}\nGot:\n{actual_sorted}"

            # Check weights correspond to correct tokens
            for i in range(count):
                token_idx = actual_tokens[i].item()
                weight = routed_weights_torch[expert_idx, i]

                # Find this token in the original selection
                for t_idx in range(num_tokens):
                    for k in range(top_k):
                        if selected_experts_np[t_idx, k] == expert_idx and t_idx == token_idx:
                            expected_weight = routing_weights_np[t_idx, k]
                            assert torch.allclose(weight.unsqueeze(0), expected_weight.unsqueeze(0), atol=1e-2), \
                                f"Weight mismatch for expert {expert_idx}, token {token_idx}"
                            break

        # Check padding (tokens beyond count should be invalid)
        if count < num_tokens:
            padding_tokens = routed_tokens_torch[expert_idx, count:]
            # Invalid tokens are marked as 0xFFFFFFFF (or -1 in signed int)
            assert torch.all((padding_tokens == -1) | (padding_tokens == 0xFFFFFFFF)), \
                f"Expert {expert_idx} padding not properly set"

            padding_weights = routed_weights_torch[expert_idx, count:]
            assert torch.all(padding_weights == 0), \
                f"Expert {expert_idx} padding weights not zero"

    print(f"✓ Test passed: num_tokens={num_tokens}, top_k={top_k}, num_experts={num_experts}")
