// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_output_op.hpp"
#include "projection_to_output_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe {

void ProjectionToOutput::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 6, "ProjectionToOutput requires 6 input tensors");

    const auto& combined_activations = input_tensors[0];
    const auto& routed_tokens = input_tensors[1];
    const auto& num_routed_tokens = input_tensors[2];
    const auto& routed_token_weights = input_tensors[3];
    const auto& down_proj_weights = input_tensors[4];
    const auto& device_expert_mapping = input_tensors[5];

    // Validate combined_activations
    TT_FATAL(combined_activations.storage_type() == StorageType::DEVICE, "combined_activations must be on device");
    TT_FATAL(combined_activations.layout() == Layout::ROW_MAJOR, "combined_activations must be ROW_MAJOR");
    TT_FATAL(combined_activations.dtype() == DataType::BFLOAT16, "combined_activations must be BFLOAT16");

    const auto& combined_shape = combined_activations.padded_shape();
    TT_FATAL(combined_shape[1] == expert_dim,
        "combined_activations width {} doesn't match expert_dim {}", combined_shape[1], expert_dim);

    // Validate routed_tokens
    TT_FATAL(routed_tokens.storage_type() == StorageType::DEVICE, "routed_tokens must be on device");
    TT_FATAL(routed_tokens.layout() == Layout::ROW_MAJOR, "routed_tokens must be ROW_MAJOR");
    TT_FATAL(routed_tokens.dtype() == DataType::UINT32, "routed_tokens must be UINT32");

    const auto& routed_shape = routed_tokens.padded_shape();
    TT_FATAL(routed_shape[1] == max_tokens_per_expert,
        "routed_tokens width {} doesn't match max_tokens_per_expert {}", routed_shape[1], max_tokens_per_expert);

    // Validate num_routed_tokens
    TT_FATAL(num_routed_tokens.storage_type() == StorageType::DEVICE, "num_routed_tokens must be on device");
    TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR");
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be UINT32");

    // Validate routed_token_weights
    TT_FATAL(routed_token_weights.storage_type() == StorageType::DEVICE, "routed_token_weights must be on device");
    TT_FATAL(routed_token_weights.layout() == Layout::ROW_MAJOR, "routed_token_weights must be ROW_MAJOR");
    TT_FATAL(routed_token_weights.dtype() == DataType::BFLOAT16, "routed_token_weights must be BFLOAT16");

    const auto& weights_routing_shape = routed_token_weights.padded_shape();
    TT_FATAL(weights_routing_shape == routed_shape,
        "routed_token_weights shape must match routed_tokens shape");

    // Validate down_proj_weights
    TT_FATAL(down_proj_weights.storage_type() == StorageType::DEVICE, "down_proj_weights must be on device");
    TT_FATAL(down_proj_weights.layout() == Layout::ROW_MAJOR, "down_proj_weights must be ROW_MAJOR layout");
    TT_FATAL(down_proj_weights.dtype() == DataType::BFLOAT16, "down_proj_weights must be BFLOAT16");

    // Use logical shape for dimension validation to get actual dimensions without padding
    const auto& weights_logical_shape = down_proj_weights.logical_shape();
    TT_FATAL(weights_logical_shape[0] == experts_per_device,
        "down_proj_weights experts {} doesn't match experts_per_device {}", weights_logical_shape[0], experts_per_device);
    TT_FATAL(weights_logical_shape[1] == expert_dim,
        "down_proj_weights input dim {} doesn't match expert_dim {}", weights_logical_shape[1], expert_dim);
    TT_FATAL(weights_logical_shape[2] == hidden_dim,
        "down_proj_weights output dim {} doesn't match hidden_dim {}", weights_logical_shape[2], hidden_dim);

    // Validate device_expert_mapping
    TT_FATAL(device_expert_mapping.storage_type() == StorageType::DEVICE, "device_expert_mapping must be on device");
    TT_FATAL(device_expert_mapping.layout() == Layout::ROW_MAJOR, "device_expert_mapping must be ROW_MAJOR");
    TT_FATAL(device_expert_mapping.dtype() == DataType::INT32, "device_expert_mapping must be INT32");

    const auto& mapping_shape = device_expert_mapping.padded_shape();
    // Accept both (1, E/D) for single device and (1, 1, E/D) for sharded multi-device
    bool valid_shape = false;
    if (mapping_shape.rank() == 2 && mapping_shape[0] == 1 && mapping_shape[1] == experts_per_device) {
        valid_shape = true;  // Single device: (1, E/D)
    } else if (mapping_shape.rank() == 3 && mapping_shape[0] == 1 && mapping_shape[1] == 1 && mapping_shape[2] == experts_per_device) {
        valid_shape = true;  // Multi-device sharded: (1, 1, E/D)
    }
    TT_FATAL(valid_shape,
        "device_expert_mapping shape must be (1, {}) or (1, 1, {}), got rank {} shape",
        experts_per_device, experts_per_device, mapping_shape.rank());
}

std::vector<TensorSpec> ProjectionToOutput::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& combined_activations = input_tensors[0];

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

    return projection_to_output_multi_core(
        input_tensors[0],  // combined_activations
        input_tensors[1],  // routed_tokens
        input_tensors[2],  // num_routed_tokens
        input_tensors[3],  // routed_token_weights
        input_tensors[4],  // down_proj_weights
        input_tensors[5],  // device_expert_mapping
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