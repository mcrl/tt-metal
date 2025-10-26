// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_attention_input_prefill_op.hpp"
#include "extract_attention_input_prefill_program_factory.hpp"

namespace ttnn::operations::experimental::attention {

void ExtractAttentionInputPrefill::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto& hidden_state = input_tensors.at(0);

    // Validate input: [B, S, H] bfloat16, TILE_LAYOUT
    TT_FATAL(
        hidden_state.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "hidden_state must be bfloat16, got {}",
        hidden_state.dtype());

    TT_FATAL(
        hidden_state.layout() == tt::tt_metal::Layout::TILE,
        "hidden_state must be TILE layout, got {}",
        hidden_state.layout());

    const auto& input_shape = hidden_state.padded_shape();
    TT_FATAL(
        input_shape.rank() == 3,
        "hidden_state must be 3D [B, S, H], got rank {}",
        input_shape.rank());

    uint32_t batch_size = input_shape[0];
    uint32_t seq_len = input_shape[1];
    uint32_t hidden_dim = input_shape[2];

    // Validate divisibility
    TT_FATAL(
        batch_size % dp == 0,
        "Batch size {} must be divisible by dp {}",
        batch_size, dp);

    uint32_t batch_per_device = batch_size / dp;

    // Validate tile alignment
    TT_FATAL(
        (batch_size * seq_len) % 32 == 0,
        "Input rows (B * S = {} * {}) must be tile-aligned (divisible by 32)",
        batch_size, seq_len);

    TT_FATAL(
        hidden_dim % 32 == 0,
        "Hidden dimension {} must be tile-aligned (divisible by 32)",
        hidden_dim);

    TT_FATAL(
        batch_per_device % 32 == 0,
        "Batch per device {} must be tile-aligned (divisible by 32)",
        batch_per_device);

    // Validate output dtype
    TT_FATAL(
        output_dtype == DataType::BFLOAT16 || output_dtype == DataType::BFLOAT8_B,
        "output_dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        output_dtype);
}

std::vector<TensorSpec> ExtractAttentionInputPrefill::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& hidden_state = input_tensors.at(0);
    const auto& input_shape = hidden_state.padded_shape();

    uint32_t batch_size = input_shape[0];
    uint32_t seq_len = input_shape[1];
    uint32_t hidden_dim = input_shape[2];

    uint32_t batch_per_device = batch_size / dp;

    // Output shape: [B//dp, 1, S, H]
    ttnn::Shape output_shape({batch_per_device, 1, seq_len, hidden_dim});

    return {TensorSpec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(Layout::TILE), output_mem_config))};
}

std::vector<Tensor> ExtractAttentionInputPrefill::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {create_device_tensor(output_specs[0], input_tensor.device())};
}

operation::ProgramWithCallbacks ExtractAttentionInputPrefill::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& hidden_state = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    return detail::extract_attention_input_prefill_single_core(
        hidden_state, output, dp, output_dtype);
}

}  // namespace ttnn::operations::experimental::attention
