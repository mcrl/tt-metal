// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_routing_tensors_op.hpp"
#include "prepare_moe_routing_tensors_program_factory.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::moe {

void PrepareMoeRoutingTensors::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& selected_experts = input_tensors.at(0);
    const auto& routing_weights = input_tensors.at(1);
    const auto& device_expert_mapping = input_tensors.at(2);

    TT_FATAL(input_tensors.size() == 3, "Expected 3 input tensors (selected_experts, routing_weights, device_expert_mapping)");

    TT_FATAL(selected_experts.storage_type() == StorageType::DEVICE, "selected_experts must be on device");
    TT_FATAL(routing_weights.storage_type() == StorageType::DEVICE, "routing_weights must be on device");
    TT_FATAL(device_expert_mapping.storage_type() == StorageType::DEVICE, "device_expert_mapping must be on device");

    TT_FATAL(selected_experts.dtype() == DataType::UINT32, "selected_experts must be UINT32");
    TT_FATAL(routing_weights.dtype() == DataType::BFLOAT16, "routing_weights must be BFLOAT16");
    TT_FATAL(device_expert_mapping.dtype() == DataType::INT32, "device_expert_mapping must be INT32");

    TT_FATAL(selected_experts.layout() == Layout::ROW_MAJOR, "selected_experts must be ROW_MAJOR layout");
    TT_FATAL(routing_weights.layout() == Layout::ROW_MAJOR, "routing_weights must be ROW_MAJOR layout");
    TT_FATAL(device_expert_mapping.layout() == Layout::ROW_MAJOR, "device_expert_mapping must be ROW_MAJOR layout");

    TT_FATAL(selected_experts.buffer() != nullptr, "selected_experts buffer is null");
    TT_FATAL(routing_weights.buffer() != nullptr, "routing_weights buffer is null");
    TT_FATAL(device_expert_mapping.buffer() != nullptr, "device_expert_mapping buffer is null");

    const auto& experts_shape = selected_experts.padded_shape();
    const auto& weights_shape = routing_weights.padded_shape();
    const auto& mapping_shape = device_expert_mapping.padded_shape();

    TT_FATAL(experts_shape == weights_shape, "selected_experts and routing_weights must have same shape");
    TT_FATAL(experts_shape.rank() == 2, "selected_experts and routing_weights must be 2D tensors");

    // device_expert_mapping can be either 1D (E/D,) or 2D (1, E/D)
    TT_FATAL(mapping_shape.rank() == 1 || mapping_shape.rank() == 2, "device_expert_mapping must be 1D or 2D tensor");
    uint32_t mapping_size;
    if (mapping_shape.rank() == 1) {
        mapping_size = mapping_shape[0];
    } else {
        TT_FATAL(mapping_shape[0] == 1, "device_expert_mapping 2D tensor must have shape (1, E/D)");
        mapping_size = mapping_shape[1];
    }
    TT_FATAL(mapping_size <= num_experts, "device_expert_mapping size ({}) cannot exceed num_experts ({})", mapping_size, num_experts);

    const uint32_t top_k = experts_shape[1];
    TT_FATAL(top_k <= num_experts, "top_k ({}) cannot exceed num_experts ({})", top_k, num_experts);
}

std::vector<TensorSpec> PrepareMoeRoutingTensors::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& selected_experts = input_tensors[0];
    const auto& device_expert_mapping = input_tensors[2];

    const auto& experts_shape = selected_experts.padded_shape();
    const auto& mapping_shape = device_expert_mapping.padded_shape();

    const uint32_t num_tokens = experts_shape[0];

    // Extract num_local_experts from device_expert_mapping shape
    uint32_t num_local_experts;
    if (mapping_shape.rank() == 1) {
        num_local_experts = mapping_shape[0];
    } else {
        // 2D tensor: (1, E/D) or (E/D, 1)
        num_local_experts = (mapping_shape[0] == 1) ? mapping_shape[1] : mapping_shape[0];
    }

    // Maximum tokens that can be routed to any single expert (worst case: all tokens choose that expert)
    const uint32_t max_tokens_per_expert = num_tokens;

    // Output 1: num_routed_tokens (E/D, 1) - device-local expert count (2D tensor for per-element pages)
    // Using 2D shape (E/D, 1) ensures each element is in its own page for safe multi-core writes
    ttnn::Shape num_routed_shape({num_local_experts, 1});
    auto num_routed_spec = TensorSpec(
        num_routed_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    // Output 2: routed_tokens (E/D, max_tokens) uint32 2D tensor - device-local routing table
    ttnn::Shape routed_tokens_shape({num_local_experts, max_tokens_per_expert});
    auto routed_tokens_spec = TensorSpec(
        routed_tokens_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    // Output 3: routed_token_weights (E/D, max_tokens) bfloat16 2D tensor - device-local routing weights
    ttnn::Shape routed_weights_shape({num_local_experts, max_tokens_per_expert});
    auto routed_weights_spec = TensorSpec(
        routed_weights_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    // Output 4: token_idx_map (E/D, max_tokens) uint32 2D tensor - local to global token index mapping
    ttnn::Shape tokenidx_map_shape({num_local_experts, max_tokens_per_expert});
    auto tokenidx_map_spec = TensorSpec(
        tokenidx_map_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    return {num_routed_spec, routed_tokens_spec, routed_weights_spec, tokenidx_map_spec};
}

std::vector<Tensor> PrepareMoeRoutingTensors::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {
        create_device_tensor(output_specs[0], input_tensor.device()),
        create_device_tensor(output_specs[1], input_tensor.device()),
        create_device_tensor(output_specs[2], input_tensor.device()),
        create_device_tensor(output_specs[3], input_tensor.device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks PrepareMoeRoutingTensors::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& selected_experts = input_tensors.at(0);
    const auto& routing_weights = input_tensors.at(1);
    const auto& device_expert_mapping = input_tensors.at(2);
    auto& num_routed_tokens = output_tensors.at(0);
    auto& routed_tokens = output_tensors.at(1);
    auto& routed_token_weights = output_tensors.at(2);
    auto& token_idx_map = output_tensors.at(3);

    // Extract dimensions from input tensors
    const auto& experts_shape = selected_experts.padded_shape();
    const auto& mapping_shape = device_expert_mapping.padded_shape();

    const uint32_t num_tokens = experts_shape[0];

    // Extract num_local_experts from device_expert_mapping shape
    uint32_t num_local_experts;
    if (mapping_shape.rank() == 1) {
        num_local_experts = mapping_shape[0];
    } else {
        // 2D tensor: (1, E/D) or (E/D, 1)
        num_local_experts = (mapping_shape[0] == 1) ? mapping_shape[1] : mapping_shape[0];
    }

    const uint32_t max_tokens_per_expert = num_tokens;

    return prepare_moe_routing_tensors_multi_core(
        selected_experts,
        routing_weights,
        device_expert_mapping,
        num_routed_tokens,
        routed_tokens,
        routed_token_weights,
        token_idx_map,
        num_experts,
        num_local_experts,
        max_tokens_per_expert);
}

}  // namespace ttnn::operations::experimental::moe