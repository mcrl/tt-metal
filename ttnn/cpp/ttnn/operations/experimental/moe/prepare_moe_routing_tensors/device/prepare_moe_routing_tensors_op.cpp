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

    TT_FATAL(input_tensors.size() == 2, "Expected 2 input tensors (selected_experts, routing_weights)");

    TT_FATAL(selected_experts.storage_type() == StorageType::DEVICE, "selected_experts must be on device");
    TT_FATAL(routing_weights.storage_type() == StorageType::DEVICE, "routing_weights must be on device");

    TT_FATAL(selected_experts.dtype() == DataType::UINT32, "selected_experts must be UINT32");
    TT_FATAL(routing_weights.dtype() == DataType::BFLOAT16, "routing_weights must be BFLOAT16");

    TT_FATAL(selected_experts.layout() == Layout::ROW_MAJOR, "selected_experts must be ROW_MAJOR layout");
    TT_FATAL(routing_weights.layout() == Layout::ROW_MAJOR, "routing_weights must be ROW_MAJOR layout");

    TT_FATAL(selected_experts.buffer() != nullptr, "selected_experts buffer is null");
    TT_FATAL(routing_weights.buffer() != nullptr, "routing_weights buffer is null");

    const auto& experts_shape = selected_experts.padded_shape();
    const auto& weights_shape = routing_weights.padded_shape();

    TT_FATAL(experts_shape == weights_shape, "selected_experts and routing_weights must have same shape");
    TT_FATAL(experts_shape.rank() == 2, "Inputs must be 2D tensors");

    const uint32_t top_k = experts_shape[1];
    TT_FATAL(top_k <= num_experts, "top_k ({}) cannot exceed num_experts ({})", top_k, num_experts);
}

std::vector<TensorSpec> PrepareMoeRoutingTensors::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& selected_experts = input_tensors[0];
    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    // Output 1: num_routed_tokens (E)
    ttnn::Shape num_routed_shape({1, num_experts});
    auto num_routed_spec = TensorSpec(
        num_routed_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    // Output 2: routed_tokens (E × max_tokens)
    ttnn::Shape routed_tokens_shape({num_experts, max_tokens_per_expert});
    auto routed_tokens_spec = TensorSpec(
        routed_tokens_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    // Output 3: routed_token_weights (E × max_tokens)
    ttnn::Shape routed_weights_shape({num_experts, max_tokens_per_expert});
    auto routed_weights_spec = TensorSpec(
        routed_weights_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    return {num_routed_spec, routed_tokens_spec, routed_weights_spec};
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
    };
}

tt::tt_metal::operation::ProgramWithCallbacks PrepareMoeRoutingTensors::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& selected_experts = input_tensors.at(0);
    const auto& routing_weights = input_tensors.at(1);
    auto& num_routed_tokens = output_tensors.at(0);
    auto& routed_tokens = output_tensors.at(1);
    auto& routed_token_weights = output_tensors.at(2);

    return prepare_moe_routing_tensors_single_core(
        selected_experts,
        routing_weights,
        num_routed_tokens,
        routed_tokens,
        routed_token_weights,
        num_experts,
        max_tokens_per_expert);
}

}  // namespace ttnn::operations::experimental::moe