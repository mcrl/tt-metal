// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "local_reduce_moe_output_op.hpp"
#include "local_reduce_moe_output_program_factory.hpp"

namespace ttnn::operations::experimental::moe {

void LocalReduceMoeOutput::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& token_idx_map = input_tensors.at(1);
    const auto& routed_token_weights = input_tensors.at(2);
    const auto& num_routed_tokens = input_tensors.at(3);

    // Validate input_hidden_state: (E/D, T, H) bfloat16, ROW_MAJOR
    TT_FATAL(
        input_hidden_state.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "input_hidden_state must be bfloat16, got {}",
        input_hidden_state.dtype());

    TT_FATAL(
        input_hidden_state.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "input_hidden_state must be ROW_MAJOR layout, got {}",
        input_hidden_state.layout());

    const auto& input_shape = input_hidden_state.padded_shape();
    TT_FATAL(
        input_shape.rank() == 3,
        "input_hidden_state must be 3D (E/D, T, H), got rank {}",
        input_shape.rank());

    uint32_t num_local_experts = input_shape[-3];
    uint32_t max_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    TT_FATAL(
        max_tokens == num_tokens,
        "input_hidden_state tokens dimension must match num_tokens: {} vs {}",
        max_tokens, num_tokens);

    // Validate token_idx_map: (E/D, T) uint32, ROW_MAJOR
    TT_FATAL(
        token_idx_map.dtype() == tt::tt_metal::DataType::UINT32,
        "token_idx_map must be uint32, got {}",
        token_idx_map.dtype());

    TT_FATAL(
        token_idx_map.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "token_idx_map must be ROW_MAJOR layout, got {}",
        token_idx_map.layout());

    const auto& token_idx_shape = token_idx_map.padded_shape();
    TT_FATAL(
        token_idx_shape.rank() == 2,
        "token_idx_map must be 2D (E/D, T), got rank {}",
        token_idx_shape.rank());

    TT_FATAL(
        token_idx_shape[-2] == num_local_experts && token_idx_shape[-1] == num_tokens,
        "token_idx_map shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts, num_tokens, token_idx_shape[-2], token_idx_shape[-1]);

    // Validate routed_token_weights: (E/D, T) bfloat16, ROW_MAJOR
    TT_FATAL(
        routed_token_weights.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "routed_token_weights must be bfloat16, got {}",
        routed_token_weights.dtype());

    TT_FATAL(
        routed_token_weights.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "routed_token_weights must be ROW_MAJOR layout, got {}",
        routed_token_weights.layout());

    const auto& weights_shape = routed_token_weights.padded_shape();
    TT_FATAL(
        weights_shape.rank() == 2,
        "routed_token_weights must be 2D (E/D, T), got rank {}",
        weights_shape.rank());

    TT_FATAL(
        weights_shape[-2] == num_local_experts && weights_shape[-1] == num_tokens,
        "routed_token_weights shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts, num_tokens, weights_shape[-2], weights_shape[-1]);

    // Validate num_routed_tokens: (E/D, 1) uint32, ROW_MAJOR
    TT_FATAL(
        num_routed_tokens.dtype() == tt::tt_metal::DataType::UINT32,
        "num_routed_tokens must be uint32, got {}",
        num_routed_tokens.dtype());

    TT_FATAL(
        num_routed_tokens.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "num_routed_tokens must be ROW_MAJOR layout, got {}",
        num_routed_tokens.layout());

    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    TT_FATAL(
        num_routed_shape.rank() == 2,
        "num_routed_tokens must be 2D (E/D, 1), got rank {}",
        num_routed_shape.rank());

    TT_FATAL(
        num_routed_shape[-2] == num_local_experts && num_routed_shape[-1] == 1,
        "num_routed_tokens shape mismatch: expected ({}, 1), got ({}, {})",
        num_local_experts, num_routed_shape[-2], num_routed_shape[-1]);

    // Validate same device
    TT_FATAL(
        input_hidden_state.device() == token_idx_map.device(),
        "All input tensors must be on the same device");

    TT_FATAL(
        input_hidden_state.device() == routed_token_weights.device(),
        "All input tensors must be on the same device");

    TT_FATAL(
        input_hidden_state.device() == num_routed_tokens.device(),
        "All input tensors must be on the same device");
}

std::vector<TensorSpec> LocalReduceMoeOutput::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& input_shape = input_hidden_state.padded_shape();
    uint32_t hidden_dim = input_shape[-1];

    // Output shape: (T, H)
    ttnn::Shape output_shape({num_tokens, hidden_dim});

    return {TensorSpec(
        output_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config))};
}

std::vector<Tensor> LocalReduceMoeOutput::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {create_device_tensor(output_specs[0], input_tensor.device())};
}

operation::ProgramWithCallbacks LocalReduceMoeOutput::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& token_idx_map = input_tensors.at(1);
    const auto& routed_token_weights = input_tensors.at(2);
    const auto& num_routed_tokens = input_tensors.at(3);
    auto& output = output_tensors.at(0);

    // Always use multi-core implementation
    return detail::local_reduce_moe_output_multi_core(
        input_hidden_state, token_idx_map, routed_token_weights,
        num_routed_tokens, num_tokens, output);
}

}  // namespace ttnn::operations::experimental::moe
