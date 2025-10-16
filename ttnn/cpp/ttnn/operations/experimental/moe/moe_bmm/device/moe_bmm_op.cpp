// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_bmm_op.hpp"
#include "moe_bmm_program_factory.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::moe {

void MoEBMM::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 3, "Expected 3 input tensors");

    const auto& input = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    const auto& num_routed_tokens = input_tensors.at(2);

    // Validate storage
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(weights.storage_type() == StorageType::DEVICE, "weights must be on device");
    TT_FATAL(num_routed_tokens.storage_type() == StorageType::DEVICE, "num_routed_tokens must be on device");

    // Validate dtypes
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "input must be BFLOAT16");
    TT_FATAL(weights.dtype() == DataType::BFLOAT16, "weights must be BFLOAT16");
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be UINT32");

    // Validate layouts
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(weights.layout() == Layout::TILE, "weights must be TILE layout");
    // TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR layout");

    // Validate buffers
    TT_FATAL(input.buffer() != nullptr, "input buffer is null");
    TT_FATAL(weights.buffer() != nullptr, "weights buffer is null");
    TT_FATAL(num_routed_tokens.buffer() != nullptr, "num_routed_tokens buffer is null");

    // Validate shapes and consistency
    const auto& input_shape = input.padded_shape();
    const auto& weights_shape = weights.padded_shape();
    const auto& num_routed_shape = num_routed_tokens.padded_shape();

    TT_FATAL(input_shape.rank() == 3, "input must be 3D (E/D, T, H_in)");
    TT_FATAL(weights_shape.rank() == 3, "weights must be 3D (E/D, H_in, H_out)");
    TT_FATAL(num_routed_shape.rank() == 2, "num_routed_tokens must be 2D (E/D, 1)");

    // Validate num_experts consistency
    TT_FATAL(input_shape[0] == weights_shape[0],
        "input dim [0] ({}) must match weights dim [0] ({})",
        input_shape[0], weights_shape[0]);
    // TT_FATAL(input_shape[0] == num_routed_shape[0],
    //     "input dim [0] ({}) must match num_routed_tokens dim [0] ({})",
    //     input_shape[0], num_routed_shape[0]);

    // Validate H_in consistency
    TT_FATAL(input_shape[2] == weights_shape[1],
        "input dim [2] ({}) must match weights dim [1] ({})",
        input_shape[2], weights_shape[1]);

    // Validate num_routed_tokens shape
    // TT_FATAL(num_routed_shape[1] == 1,
    //     "num_routed_tokens dim [1] must be 1, got {}",
    //     num_routed_shape[1]);

    // Validate TILE alignment for input and weights
    TT_FATAL(input_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "input dim [1] ({}) must be multiple of TILE_HEIGHT ({})",
        input_shape[1], tt::constants::TILE_HEIGHT);
    TT_FATAL(input_shape[2] % tt::constants::TILE_WIDTH == 0,
        "input dim [2] ({}) must be multiple of TILE_WIDTH ({})",
        input_shape[2], tt::constants::TILE_WIDTH);
    TT_FATAL(weights_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "weights dim [1] ({}) must be multiple of TILE_HEIGHT ({})",
        weights_shape[1], tt::constants::TILE_HEIGHT);
    TT_FATAL(weights_shape[2] % tt::constants::TILE_WIDTH == 0,
        "weights dim [2] ({}) must be multiple of TILE_WIDTH ({})",
        weights_shape[2], tt::constants::TILE_WIDTH);
}

std::vector<TensorSpec> MoEBMM::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);

    const auto& input_shape = input.padded_shape();
    const auto& weights_shape = weights.padded_shape();

    const uint32_t num_experts = input_shape[0];
    const uint32_t max_tokens = input_shape[1];
    const uint32_t h_out = weights_shape[2];

    // Output shape: (E/D, T, H_out)
    ttnn::Shape output_shape({num_experts, max_tokens, h_out});

    auto output_spec = TensorSpec(
        output_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), output_mem_config));

    return {output_spec};
}

std::vector<Tensor> MoEBMM::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {
        create_device_tensor(output_specs[0], input_tensor.device()),
    };
}

operation::ProgramWithCallbacks MoEBMM::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input = input_tensors.at(0);
    const auto& weights = input_tensors.at(1);
    const auto& num_routed_tokens = input_tensors.at(2);
    auto& output = output_tensors.at(0);

    const auto& input_shape = input.padded_shape();
    const auto& weights_shape = weights.padded_shape();

    const uint32_t num_experts = input_shape[0];
    const uint32_t max_tokens = input_shape[1];
    const uint32_t h_in = input_shape[2];
    const uint32_t h_out = weights_shape[2];

    return moe_bmm_single_core(
        input,
        weights,
        num_routed_tokens,
        output,
        num_experts,
        max_tokens,
        h_in,
        h_out);
}

}  // namespace ttnn::operations::experimental::moe
