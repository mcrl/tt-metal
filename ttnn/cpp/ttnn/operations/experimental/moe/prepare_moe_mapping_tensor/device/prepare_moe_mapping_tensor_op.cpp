// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_mapping_tensor_op.hpp"
#include "prepare_moe_mapping_tensor_program_factory.hpp"

namespace ttnn::operations::experimental::moe {

void PrepareMoeMappingTensor::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Expected 2 input tensors (selected_experts and routing_weights)");

    const auto& selected_experts = input_tensors[0];
    const auto& routing_weights = input_tensors[1];

    TT_FATAL(selected_experts.layout() == Layout::ROW_MAJOR, "selected_experts must be ROW_MAJOR layout");
    // Accept either ROW_MAJOR or TILE layout for routing_weights
    // ROW_MAJOR is easier to process in the kernel

    // Use logical shapes for comparison (TILE layout adds padding to padded_shape)
    const auto& experts_shape = selected_experts.logical_shape();
    const auto& weights_shape = routing_weights.logical_shape();

    TT_FATAL(experts_shape.rank() == 2, "selected_experts must be 2D tensor (T x K)");
    TT_FATAL(weights_shape.rank() == 2, "routing_weights must be 2D tensor (T x K)");
    TT_FATAL(
        experts_shape[0] == weights_shape[0] && experts_shape[1] == weights_shape[1],
        "selected_experts and routing_weights must have same logical shape");
}

std::vector<TensorSpec> PrepareMoeMappingTensor::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& selected_experts = input_tensors[0];
    const auto& routing_weights = input_tensors[1];

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];

    // Output shape: (num_tokens, num_experts)
    // Pad num_experts to tile boundary (32) for TILE layout compatibility
    const uint32_t padded_num_experts = (num_experts + 31) & ~31;

    auto logical_shape = ttnn::Shape(std::array<uint32_t, 2>{num_tokens, num_experts});
    auto padded_shape = ttnn::Shape(std::array<uint32_t, 2>{num_tokens, padded_num_experts});

    return {TensorSpec(
        logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            routing_weights.dtype(),
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            memory_config,
            logical_shape,
            padded_shape))};
}

tt::tt_metal::operation::ProgramWithCallbacks PrepareMoeMappingTensor::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& selected_experts = input_tensors[0];
    const auto& routing_weights = input_tensors[1];
    auto& output = output_tensors[0];

    return detail::prepare_moe_mapping_tensor_single_core(
        selected_experts, routing_weights, output, num_experts);
}

}  // namespace ttnn::operations::experimental::moe
