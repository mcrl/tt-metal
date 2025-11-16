import ttnn
import pytest
import torch


def reference_prepare_moe_routing_tensors(selected_experts, routing_weights, device_expert_mapping):
    """
    Reference implementation for prepare_moe_routing_tensors (device-local).

    Args:
        selected_experts: (T, K) tensor of GLOBAL expert indices
        routing_weights: (T, K) tensor of routing weights
        device_expert_mapping: (E/D,) tensor of GLOBAL expert IDs assigned to this device

    Returns:
        - num_routed_tokens: (E/D,) count of tokens per LOCAL expert (1D tensor)
        - routed_tokens: (E/D, max_tokens) token indices per LOCAL expert
        - routed_token_weights: (E/D, max_tokens) weights per LOCAL expert
        - tokenidx_expertlocal_to_global: (E/D, max_tokens) mapping from expert-local to global token index
    """
    num_tokens, top_k = selected_experts.shape
    num_local_experts = device_expert_mapping.shape[0]
    max_tokens_per_expert = num_tokens

    global_to_local = {}
    for local_idx in range(num_local_experts):
        global_idx = device_expert_mapping[local_idx].item()
        global_to_local[global_idx] = local_idx

    num_routed_tokens = torch.zeros(num_local_experts, dtype=torch.int32)
    routed_tokens = torch.full((num_local_experts, max_tokens_per_expert), -1, dtype=torch.int32)
    routed_token_weights = torch.zeros((num_local_experts, max_tokens_per_expert), dtype=torch.bfloat16)
    tokenidx_expertlocal_to_global = torch.full((num_local_experts, max_tokens_per_expert), -1, dtype=torch.int32)

    for token_idx in range(num_tokens):
        for k in range(top_k):
            global_expert_idx = selected_experts[token_idx, k].item()
            weight = routing_weights[token_idx, k]

            if global_expert_idx in global_to_local:
                local_expert_idx = global_to_local[global_expert_idx]
                count = num_routed_tokens[local_expert_idx].item()
                if count < max_tokens_per_expert:
                    routed_tokens[local_expert_idx, count] = token_idx
                    routed_token_weights[local_expert_idx, count] = weight
                    tokenidx_expertlocal_to_global[local_expert_idx, count] = token_idx  # Expert-local index -> global token index
                    num_routed_tokens[local_expert_idx] = count + 1

    return num_routed_tokens, routed_tokens, routed_token_weights, tokenidx_expertlocal_to_global


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("num_experts", [8, 32, 128])
def test_prepare_moe_routing_tensors(mesh_device, num_tokens, top_k, num_experts):
    """
    Test prepare_moe_routing_tensors on mesh_device.
    """
    import random

    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

    random.seed(42)
    torch.manual_seed(42)

    selected_experts_np = torch.zeros((num_tokens, top_k), dtype=torch.int32)
    for t in range(num_tokens):
        experts = torch.randperm(num_experts)[:top_k]
        selected_experts_np[t] = experts

    routing_weights_np = torch.rand(num_tokens, top_k, dtype=torch.bfloat16)
    routing_weights_np = routing_weights_np / routing_weights_np.sum(dim=1, keepdim=True)

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

    ref_num_routed_all = []
    ref_routed_tokens_all = []
    ref_routed_weights_all = []
    ref_tokenidx_map_all = []

    for device_idx in range(num_devices):
        ref_num_routed, ref_routed_tokens, ref_routed_weights, ref_tokenidx_map = reference_prepare_moe_routing_tensors(
            selected_experts_np, routing_weights_np, device_expert_mappings[device_idx]
        )
        ref_num_routed_all.append(ref_num_routed)
        ref_routed_tokens_all.append(ref_routed_tokens)
        ref_routed_weights_all.append(ref_routed_weights)
        ref_tokenidx_map_all.append(ref_tokenidx_map)

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

    num_routed, routed_tokens, routed_weights, tokenidx_map = ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping_tt, num_experts
    )

    assert len(num_routed.shape) == 1, f"Expected 1D tensor, got shape {num_routed.shape}"
    assert num_routed.shape[0] == experts_per_device  # 1D tensor (E/D,)
    assert routed_tokens.shape[0] == experts_per_device  # Device-local size (E/D)
    assert routed_tokens.shape[1] == num_tokens  # max_tokens_per_expert
    assert routed_weights.shape == routed_tokens.shape
    assert tokenidx_map.shape == routed_tokens.shape  # Same shape as routed_tokens

    num_routed_torch = ttnn.to_torch(num_routed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_tokens_torch = ttnn.to_torch(routed_tokens, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    routed_weights_torch = ttnn.to_torch(routed_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tokenidx_map_torch = ttnn.to_torch(tokenidx_map, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    for device_idx in range(num_devices):
        start = device_idx * experts_per_device
        end = (device_idx + 1) * experts_per_device
        num_routed_torch_device = num_routed_torch[start:end]
        ref_num_routed_device = ref_num_routed_all[device_idx]
        assert torch.equal(num_routed_torch_device, ref_num_routed_device), \
            f"Device {device_idx}: num_routed_tokens mismatch:\nExpected:\n{ref_num_routed_device}\nGot:\n{num_routed_torch_device}"

    for device_idx in range(num_devices):
        start = device_idx * experts_per_device
        end = (device_idx + 1) * experts_per_device

        num_routed_torch_device = num_routed_torch[start:end]
        routed_tokens_torch_device = routed_tokens_torch[start:end]
        routed_weights_torch_device = routed_weights_torch[start:end]
        tokenidx_map_torch_device = tokenidx_map_torch[start:end]

        ref_num_routed = ref_num_routed_all[device_idx]
        ref_routed_tokens = ref_routed_tokens_all[device_idx]
        ref_routed_weights = ref_routed_weights_all[device_idx]
        ref_tokenidx_map = ref_tokenidx_map_all[device_idx]

        for local_expert_idx in range(experts_per_device):
            count = ref_num_routed[local_expert_idx].item()

            if count > 0:
                actual_tokens = routed_tokens_torch_device[local_expert_idx, :count]
                expected_tokens = ref_routed_tokens[local_expert_idx, :count]

                actual_sorted = torch.sort(actual_tokens)[0]
                expected_sorted = torch.sort(expected_tokens)[0]

                assert torch.equal(actual_sorted, expected_sorted), \
                    f"Device {device_idx}, Local expert {local_expert_idx} token mismatch:\nExpected:\n{expected_sorted}\nGot:\n{actual_sorted}"

                actual_tokenidx_map = tokenidx_map_torch_device[local_expert_idx, :count]
                expected_tokenidx_map = ref_tokenidx_map[local_expert_idx, :count]

                assert torch.equal(actual_tokenidx_map, actual_tokens), \
                    f"Device {device_idx}, Local expert {local_expert_idx} tokenidx mapping should match routed_tokens:\nMapping:\n{actual_tokenidx_map}\nTokens:\n{actual_tokens}"

                actual_tokenidx_sorted = torch.sort(actual_tokenidx_map)[0]
                expected_tokenidx_sorted = torch.sort(expected_tokenidx_map)[0]
                assert torch.equal(actual_tokenidx_sorted, expected_tokenidx_sorted), \
                    f"Device {device_idx}, Local expert {local_expert_idx} tokenidx mapping mismatch:\nExpected:\n{expected_tokenidx_sorted}\nGot:\n{actual_tokenidx_sorted}"

                global_expert_idx = device_expert_mappings[device_idx][local_expert_idx].item()
                for i in range(count):
                    token_idx = actual_tokens[i].item()
                    weight = routed_weights_torch_device[local_expert_idx, i]

                    for t_idx in range(num_tokens):
                        for k in range(top_k):
                            if selected_experts_np[t_idx, k] == global_expert_idx and t_idx == token_idx:
                                expected_weight = routing_weights_np[t_idx, k]
                                assert torch.allclose(weight.unsqueeze(0), expected_weight.unsqueeze(0), atol=1e-2), \
                                    f"Device {device_idx}, Weight mismatch for local expert {local_expert_idx} (global {global_expert_idx}), token {token_idx}"
                                break

            if count < num_tokens:
                padding_tokens = routed_tokens_torch_device[local_expert_idx, count:]
                assert torch.all((padding_tokens == -1) | (padding_tokens == 0xFFFFFFFF)), \
                    f"Device {device_idx}, Local expert {local_expert_idx} padding not properly set"

                padding_weights = routed_weights_torch_device[local_expert_idx, count:]
                assert torch.all(padding_weights == 0), \
                    f"Device {device_idx}, Local expert {local_expert_idx} padding weights not zero"

                padding_tokenidx = tokenidx_map_torch_device[local_expert_idx, count:]
                assert torch.all((padding_tokenidx == -1) | (padding_tokenidx == 0xFFFFFFFF)), \
                    f"Device {device_idx}, Local expert {local_expert_idx} tokenidx mapping padding not properly set"