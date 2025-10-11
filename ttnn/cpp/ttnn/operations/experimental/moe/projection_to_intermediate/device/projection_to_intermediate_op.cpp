// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_intermediate_op.hpp"
#include "projection_to_intermediate_program_factory.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::moe {

void ProjectionToIntermediate::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 5, "Expected 5 input tensors");

    const auto& hidden_states = input_tensors.at(0);
    const auto& routed_tokens = input_tensors.at(1);
    const auto& num_routed_tokens = input_tensors.at(2);
    const auto& expert_weights = input_tensors.at(3);
    const auto& device_expert_mapping = input_tensors.at(4);

    // Validate storage
    TT_FATAL(hidden_states.storage_type() == StorageType::DEVICE, "hidden_states must be on device");
    TT_FATAL(routed_tokens.storage_type() == StorageType::DEVICE, "routed_tokens must be on device");
    TT_FATAL(num_routed_tokens.storage_type() == StorageType::DEVICE, "num_routed_tokens must be on device");
    TT_FATAL(expert_weights.storage_type() == StorageType::DEVICE, "expert_weights must be on device");
    TT_FATAL(device_expert_mapping.storage_type() == StorageType::DEVICE, "device_expert_mapping must be on device");

    // Validate dtypes
    TT_FATAL(hidden_states.dtype() == DataType::BFLOAT16, "hidden_states must be BFLOAT16");
    TT_FATAL(routed_tokens.dtype() == DataType::UINT32, "routed_tokens must be UINT32");
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be UINT32");
    TT_FATAL(expert_weights.dtype() == DataType::BFLOAT16, "expert_weights must be BFLOAT16");
    TT_FATAL(device_expert_mapping.dtype() == DataType::INT32, "device_expert_mapping must be INT32");

    // Validate layouts
    TT_FATAL(hidden_states.layout() == Layout::ROW_MAJOR, "hidden_states must be ROW_MAJOR layout");
    TT_FATAL(routed_tokens.layout() == Layout::ROW_MAJOR, "routed_tokens must be ROW_MAJOR layout");
    TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR layout");
    // NOTE: Kernel currently only supports ROW_MAJOR for expert_weights
    // TILE layout support requires different addressing logic in kernel
    TT_FATAL(expert_weights.layout() == Layout::ROW_MAJOR, "expert_weights must be ROW_MAJOR layout (TILE not yet supported)");
    TT_FATAL(device_expert_mapping.layout() == Layout::ROW_MAJOR, "device_expert_mapping must be ROW_MAJOR layout");

    // Validate buffers
    TT_FATAL(hidden_states.buffer() != nullptr, "hidden_states buffer is null");
    TT_FATAL(routed_tokens.buffer() != nullptr, "routed_tokens buffer is null");
    TT_FATAL(num_routed_tokens.buffer() != nullptr, "num_routed_tokens buffer is null");
    TT_FATAL(expert_weights.buffer() != nullptr, "expert_weights buffer is null");
    TT_FATAL(device_expert_mapping.buffer() != nullptr, "device_expert_mapping buffer is null");

    // Validate shapes
    const auto& hidden_shape = hidden_states.padded_shape();
    const auto& weights_shape = expert_weights.padded_shape();
    const auto& mapping_shape = device_expert_mapping.padded_shape();

    TT_FATAL(hidden_shape.rank() == 2, "hidden_states must be 2D");
    TT_FATAL(weights_shape.rank() == 3, "expert_weights must be 3D");

    TT_FATAL(hidden_shape[0] == num_tokens, "hidden_states[0] must match num_tokens");
    TT_FATAL(hidden_shape[1] == hidden_dim, "hidden_states[1] must match hidden_dim");
    TT_FATAL(weights_shape[0] == experts_per_device, "expert_weights[0] must match experts_per_device");
    TT_FATAL(weights_shape[1] == hidden_dim, "expert_weights[1] must match hidden_dim");
    TT_FATAL(weights_shape[2] == expert_dim, "expert_weights[2] must match expert_dim");

    // Validate device_expert_mapping shape
    TT_FATAL(mapping_shape[0] == 1 || mapping_shape[0] == experts_per_device,
        "device_expert_mapping must have shape (E/D) or (1, E/D)");
    const uint32_t mapping_size = mapping_shape.rank() == 2 ? mapping_shape[1] : mapping_shape[0];
    TT_FATAL(mapping_size == experts_per_device, "device_expert_mapping size must match experts_per_device");
}

std::vector<TensorSpec> ProjectionToIntermediate::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    // Output shape: (output_size, expert_dim)
    // output_size is K*T (conservative upper bound)
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
    const auto& device_expert_mapping = input_tensors.at(4);
    auto& output = output_tensors.at(0);

    return projection_to_intermediate_single_core(
        hidden_states,
        routed_tokens,
        num_routed_tokens,
        expert_weights,
        device_expert_mapping,
        output,
        num_tokens,
        hidden_dim,
        expert_dim,
        experts_per_device,
        max_tokens_per_expert,
        output_size);
}

}  // namespace ttnn::operations::experimental::moe
