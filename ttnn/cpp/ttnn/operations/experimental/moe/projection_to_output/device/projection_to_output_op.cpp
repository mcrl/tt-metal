// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_output_op.hpp"
#include "projection_to_output_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe {

void ProjectionToOutput::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 5, "ProjectionToOutput requires 5 input tensors");

    const auto& combined_activations = input_tensors[0];
    const auto& routed_tokens = input_tensors[1];
    const auto& num_routed_tokens = input_tensors[2];
    const auto& routed_token_weights = input_tensors[3];
    const auto& down_proj_weights = input_tensors[4];

    // Validate storage types
    TT_FATAL(combined_activations.storage_type() == StorageType::DEVICE, "combined_activations must be on device");
    TT_FATAL(routed_tokens.storage_type() == StorageType::DEVICE, "routed_tokens must be on device");
    TT_FATAL(num_routed_tokens.storage_type() == StorageType::DEVICE, "num_routed_tokens must be on device");
    TT_FATAL(routed_token_weights.storage_type() == StorageType::DEVICE, "routed_token_weights must be on device");
    TT_FATAL(down_proj_weights.storage_type() == StorageType::DEVICE, "down_proj_weights must be on device");

    // Validate layouts
    TT_FATAL(combined_activations.layout() == Layout::ROW_MAJOR, "combined_activations must be ROW_MAJOR");
    TT_FATAL(routed_tokens.layout() == Layout::ROW_MAJOR, "routed_tokens must be ROW_MAJOR");
    TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR");
    TT_FATAL(routed_token_weights.layout() == Layout::ROW_MAJOR, "routed_token_weights must be ROW_MAJOR");
    TT_FATAL(down_proj_weights.layout() == Layout::ROW_MAJOR, "down_proj_weights must be ROW_MAJOR layout");

    // Validate dtypes
    TT_FATAL(combined_activations.dtype() == DataType::BFLOAT16, "combined_activations must be BFLOAT16");
    TT_FATAL(routed_tokens.dtype() == DataType::UINT32, "routed_tokens must be UINT32");
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be UINT32");
    TT_FATAL(routed_token_weights.dtype() == DataType::BFLOAT16, "routed_token_weights must be BFLOAT16");
    TT_FATAL(down_proj_weights.dtype() == DataType::BFLOAT16, "down_proj_weights must be BFLOAT16");

    // Validate shapes and consistency
    const auto& combined_shape = combined_activations.padded_shape();
    const auto& routed_shape = routed_tokens.padded_shape();
    const auto& weights_routing_shape = routed_token_weights.padded_shape();
    const auto& weights_logical_shape = down_proj_weights.logical_shape();

    TT_FATAL(combined_shape.rank() == 2, "combined_activations must be 2D");
    TT_FATAL(routed_shape.rank() == 2, "routed_tokens must be 2D");
    TT_FATAL(weights_logical_shape.rank() == 3, "down_proj_weights must be 3D");

    // Validate consistency: expert_dim must match between combined_activations and down_proj_weights
    TT_FATAL(combined_shape[1] == weights_logical_shape[1],
        "combined_activations width ({}) must match down_proj_weights input dim ({})",
        combined_shape[1], weights_logical_shape[1]);

    // Validate routing tensor shapes match
    TT_FATAL(weights_routing_shape == routed_shape,
        "routed_token_weights shape must match routed_tokens shape");

    // Validate experts_per_device matches between routing and weights
    TT_FATAL(routed_shape[0] == weights_logical_shape[0],
        "routed_tokens experts ({}) must match down_proj_weights experts ({})",
        routed_shape[0], weights_logical_shape[0]);
}

std::vector<TensorSpec> ProjectionToOutput::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& down_proj_weights = input_tensors[4];
    const auto& weights_logical_shape = down_proj_weights.logical_shape();

    const uint32_t hidden_dim = weights_logical_shape[2];

    // Output shape is (num_tokens, hidden_dim)
    ttnn::Shape output_shape({num_tokens, hidden_dim});

    auto output_spec = TensorSpec(
        output_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config));

    return {output_spec};
}

std::vector<Tensor> ProjectionToOutput::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto output_specs = compute_output_specs(input_tensors);
    return {
        create_device_tensor(
            output_specs.at(0),
            input_tensors.at(0).device()
        )
    };
}

tt::tt_metal::operation::ProgramWithCallbacks ProjectionToOutput::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {

    // Extract dimensions from input tensors
    const auto& combined_activations = input_tensors[0];
    const auto& routed_tokens = input_tensors[1];
    const auto& down_proj_weights = input_tensors[4];

    const auto& combined_shape = combined_activations.padded_shape();
    const auto& routed_shape = routed_tokens.padded_shape();
    const auto& weights_logical_shape = down_proj_weights.logical_shape();

    const uint32_t expert_dim = combined_shape[1];
    const uint32_t hidden_dim = weights_logical_shape[2];
    const uint32_t experts_per_device = weights_logical_shape[0];
    const uint32_t max_tokens_per_expert = routed_shape[1];

    return projection_to_output_single_core(
        input_tensors[0],  // combined_activations
        input_tensors[1],  // routed_tokens
        input_tensors[2],  // num_routed_tokens
        input_tensors[3],  // routed_token_weights
        input_tensors[4],  // down_proj_weights
        output_tensors[0],
        num_tokens,
        hidden_dim,
        expert_dim,
        experts_per_device,
        max_tokens_per_expert,
        top_k
    );
}

}  // namespace ttnn::operations::experimental::moe