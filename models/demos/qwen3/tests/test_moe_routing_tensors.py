# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import tt_lock


def reference_prepare_moe_routing_tensors(selected_experts, routing_weights, device_expert_mapping):
    """
    Reference implementation for prepare_moe_routing_tensors (device-local).

    Args:
        selected_experts: (T, K) tensor of GLOBAL expert indices
        routing_weights: (T, K) tensor of routing weights
        device_expert_mapping: (E/D,) tensor of GLOBAL expert IDs assigned to this device

    Returns:
        - num_routed_tokens: (E/D,) count of tokens per LOCAL expert
        - routed_tokens: (E/D, max_tokens) token indices per LOCAL expert
        - routed_token_weights: (E/D, max_tokens) weights per LOCAL expert
    """
    num_tokens, top_k = selected_experts.shape
    num_local_experts = device_expert_mapping.shape[0]
    max_tokens_per_expert = num_tokens

    # Build reverse mapping: global_expert_id -> local_expert_id
    global_to_local = {}
    for local_idx in range(num_local_experts):
        global_idx = device_expert_mapping[local_idx].item()
        global_to_local[global_idx] = local_idx

    # Initialize outputs (device-local)
    num_routed_tokens = torch.zeros(num_local_experts, dtype=torch.int32)
    routed_tokens = torch.full((num_local_experts, max_tokens_per_expert), -1, dtype=torch.int32)
    routed_token_weights = torch.zeros((num_local_experts, max_tokens_per_expert), dtype=torch.bfloat16)

    # Build routing (filter by device mapping)
    for token_idx in range(num_tokens):
        for k in range(top_k):
            global_expert_idx = selected_experts[token_idx, k].item()
            weight = routing_weights[token_idx, k]

            # Check if this expert is on this device
            if global_expert_idx in global_to_local:
                local_expert_idx = global_to_local[global_expert_idx]
                count = num_routed_tokens[local_expert_idx].item()
                if count < max_tokens_per_expert:
                    routed_tokens[local_expert_idx, count] = token_idx
                    routed_token_weights[local_expert_idx, count] = weight
                    num_routed_tokens[local_expert_idx] = count + 1

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

    # Create device-expert mapping (uniform partitioning across devices)
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

    # Stack mappings (D, E/D) then shard across devices
    device_expert_mapping_np = torch.stack(device_expert_mappings, dim=0)

    # Get reference output for first device
    ref_num_routed, ref_routed_tokens, ref_routed_weights = reference_prepare_moe_routing_tensors(
        selected_experts_np, routing_weights_np, device_expert_mappings[0]
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

    device_expert_mapping_tt = ttnn.from_torch(
        device_expert_mapping_np,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Call the operation
    num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping_tt, num_experts
    )

    # Verify output shapes (device-local: E/D per device)
    assert num_routed.shape[0] == experts_per_device  # 1D tensor with E/D elements
    assert routed_tokens.shape[0] == experts_per_device  # Device-local size (E/D)
    assert routed_tokens.shape[1] == num_tokens  # max_tokens_per_expert
    assert routed_weights.shape == routed_tokens.shape

    # Convert to PyTorch for verification (sharded outputs)
    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_weights_torch = ttnn.to_torch(routed_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Each device has different output (sharded by experts)
    # Verify first device's output
    num_routed_torch_device0 = num_routed_torch[:experts_per_device]  # First device, all values (1D)
    routed_tokens_torch_device0 = routed_tokens_torch[:experts_per_device]  # First E/D experts
    routed_weights_torch_device0 = routed_weights_torch[:experts_per_device]

    # Verify num_routed_tokens (device-local)
    assert torch.equal(num_routed_torch_device0, ref_num_routed), \
        f"num_routed_tokens mismatch:\nExpected:\n{ref_num_routed}\nGot:\n{num_routed_torch_device0}"

    # Verify routed_tokens and routed_weights (device-local expert indices)
    for local_expert_idx in range(experts_per_device):
        count = ref_num_routed[local_expert_idx].item()

        if count > 0:
            # Check valid tokens (up to count)
            actual_tokens = routed_tokens_torch_device0[local_expert_idx, :count]
            expected_tokens = ref_routed_tokens[local_expert_idx, :count]

            # Sort tokens for comparison (order may differ)
            actual_sorted = torch.sort(actual_tokens)[0]
            expected_sorted = torch.sort(expected_tokens)[0]

            assert torch.equal(actual_sorted, expected_sorted), \
                f"Local expert {local_expert_idx} token mismatch:\nExpected:\n{expected_sorted}\nGot:\n{actual_sorted}"

            # Check weights correspond to correct tokens
            global_expert_idx = device_expert_mappings[0][local_expert_idx].item()
            for i in range(count):
                token_idx = actual_tokens[i].item()
                weight = routed_weights_torch_device0[local_expert_idx, i]

                # Find this token in the original selection
                for t_idx in range(num_tokens):
                    for k in range(top_k):
                        if selected_experts_np[t_idx, k] == global_expert_idx and t_idx == token_idx:
                            expected_weight = routing_weights_np[t_idx, k]
                            assert torch.allclose(weight.unsqueeze(0), expected_weight.unsqueeze(0), atol=1e-2), \
                                f"Weight mismatch for local expert {local_expert_idx} (global {global_expert_idx}), token {token_idx}"
                            break

        # Check padding (tokens beyond count should be invalid)
        if count < num_tokens:
            padding_tokens = routed_tokens_torch_device0[local_expert_idx, count:]
            # Invalid tokens are marked as 0xFFFFFFFF (or -1 in signed int)
            assert torch.all((padding_tokens == -1) | (padding_tokens == 0xFFFFFFFF)), \
                f"Local expert {local_expert_idx} padding not properly set"

            padding_weights = routed_weights_torch_device0[local_expert_idx, count:]
            assert torch.all(padding_weights == 0), \
                f"Local expert {local_expert_idx} padding weights not zero"

    print(f"✓ Test passed: num_tokens={num_tokens}, top_k={top_k}, num_experts={num_experts}")
