// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_intermediate_op.hpp"
#include "projection_to_intermediate_program_factory.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::moe {

void ProjectionToIntermediate::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 4, "Expected 4 input tensors");

    const auto& hidden_states = input_tensors.at(0);
    const auto& routed_tokens = input_tensors.at(1);
    const auto& num_routed_tokens = input_tensors.at(2);
    const auto& expert_weights = input_tensors.at(3);

    // Validate storage
    TT_FATAL(hidden_states.storage_type() == StorageType::DEVICE, "hidden_states must be on device");
    TT_FATAL(routed_tokens.storage_type() == StorageType::DEVICE, "routed_tokens must be on device");
    TT_FATAL(num_routed_tokens.storage_type() == StorageType::DEVICE, "num_routed_tokens must be on device");
    TT_FATAL(expert_weights.storage_type() == StorageType::DEVICE, "expert_weights must be on device");

    // Validate dtypes
    TT_FATAL(hidden_states.dtype() == DataType::BFLOAT16, "hidden_states must be BFLOAT16");
    TT_FATAL(routed_tokens.dtype() == DataType::UINT32, "routed_tokens must be UINT32");
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be UINT32");
    TT_FATAL(expert_weights.dtype() == DataType::BFLOAT16, "expert_weights must be BFLOAT16");

    // Validate layouts
    TT_FATAL(hidden_states.layout() == Layout::ROW_MAJOR, "hidden_states must be ROW_MAJOR layout");
    TT_FATAL(routed_tokens.layout() == Layout::ROW_MAJOR, "routed_tokens must be ROW_MAJOR layout");
    TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR layout");
    // NOTE: Kernel currently only supports ROW_MAJOR for expert_weights
    // TILE layout support requires different addressing logic in kernel
    TT_FATAL(expert_weights.layout() == Layout::ROW_MAJOR, "expert_weights must be ROW_MAJOR layout (TILE not yet supported)");

    // Validate buffers
    TT_FATAL(hidden_states.buffer() != nullptr, "hidden_states buffer is null");
    TT_FATAL(routed_tokens.buffer() != nullptr, "routed_tokens buffer is null");
    TT_FATAL(num_routed_tokens.buffer() != nullptr, "num_routed_tokens buffer is null");
    TT_FATAL(expert_weights.buffer() != nullptr, "expert_weights buffer is null");

    // Validate shapes and consistency
    const auto& hidden_shape = hidden_states.padded_shape();
    const auto& routing_shape = routed_tokens.padded_shape();
    const auto& weights_shape = expert_weights.padded_shape();

    TT_FATAL(hidden_shape.rank() == 2, "hidden_states must be 2D");
    TT_FATAL(routing_shape.rank() == 2, "routed_tokens must be 2D");
    TT_FATAL(weights_shape.rank() == 3, "expert_weights must be 3D");

    // Validate consistency: hidden_dim must match between hidden_states and expert_weights
    TT_FATAL(hidden_shape[1] == weights_shape[1],
        "hidden_states dim [1] ({}) must match expert_weights dim [1] ({})",
        hidden_shape[1], weights_shape[1]);

    // Validate routing tensor dimensions
    TT_FATAL(routing_shape[0] == weights_shape[0],
        "routed_tokens dim [0] ({}) must match expert_weights dim [0] ({})",
        routing_shape[0], weights_shape[0]);
}

std::vector<TensorSpec> ProjectionToIntermediate::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& hidden_states = input_tensors.at(0);
    const auto& expert_weights = input_tensors.at(3);

    const auto& hidden_shape = hidden_states.padded_shape();
    const auto& weights_shape = expert_weights.padded_shape();

    const uint32_t num_tokens = hidden_shape[0];
    const uint32_t expert_dim = weights_shape[2];

    // Output shape: (output_size, expert_dim)
    // output_size is K*T (conservative upper bound)
    const uint32_t output_size = top_k * num_tokens;
    ttnn::Shape output_shape({output_size, expert_dim});

    auto output_spec = TensorSpec(
        output_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    return {output_spec};
}

std::vector<Tensor> ProjectionToIntermediate::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {
        create_device_tensor(output_specs[0], input_tensor.device()),
    };
}

tt::tt_metal::operation::ProgramWithCallbacks ProjectionToIntermediate::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& hidden_states = input_tensors.at(0);
    const auto& routed_tokens = input_tensors.at(1);
    const auto& num_routed_tokens = input_tensors.at(2);
    const auto& expert_weights = input_tensors.at(3);
    auto& output = output_tensors.at(0);

    // Extract dimensions from input tensors
    const auto& hidden_shape = hidden_states.padded_shape();
    const auto& routing_shape = routed_tokens.padded_shape();
    const auto& weights_shape = expert_weights.padded_shape();

    const uint32_t num_tokens = hidden_shape[0];
    const uint32_t hidden_dim = hidden_shape[1];
    const uint32_t expert_dim = weights_shape[2];
    const uint32_t experts_per_device = weights_shape[0];
    const uint32_t max_tokens_per_expert = routing_shape[1];
    const uint32_t output_size = top_k * num_tokens;

    return projection_to_intermediate_single_core(
        hidden_states,
        routed_tokens,
        num_routed_tokens,
        expert_weights,
        output,
        num_tokens,
        hidden_dim,
        expert_dim,
        experts_per_device,
        max_tokens_per_expert,
        output_size);
}

}  // namespace ttnn::operations::experimental::moe
