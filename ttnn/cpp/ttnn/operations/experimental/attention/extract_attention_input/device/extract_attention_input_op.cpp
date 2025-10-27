// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_attention_input_op.hpp"
#include "extract_attention_input_program_factory.hpp"

namespace ttnn::operations::experimental::attention {

void ExtractAttentionInput::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    TT_FATAL(
        input_tensors.size() == 2,
        "Expected 2 input tensors (hidden_state, dp_degree), got {}",
        input_tensors.size());

    const auto& hidden_state = input_tensors.at(0);
    const auto& dp_degree = input_tensors.at(1);

    // Validate input dtype and layout
    TT_FATAL(
        hidden_state.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "hidden_state must be bfloat16, got {}",
        hidden_state.dtype());

    TT_FATAL(
        hidden_state.layout() == tt::tt_metal::Layout::TILE,
        "hidden_state must be TILE layout, got {}",
        hidden_state.layout());

    const auto& input_shape = hidden_state.padded_shape();

    // Detect mode based on tensor rank
    bool is_prefill = (input_shape.rank() == 3);  // Prefill: [B, S, H], Decode: [1, 1, B, H]

    TT_FATAL(
        input_shape.rank() == 3 || input_shape.rank() == 4,
        "hidden_state must be rank 3 (prefill mode) or rank 4 (decode mode), got rank {}",
        input_shape.rank());

    uint32_t batch_size, seq_len, hidden_dim;

    if (is_prefill) {
        // Prefill mode: [B, S, H]
        batch_size = input_shape[0];
        seq_len = input_shape[1];
        hidden_dim = input_shape[2];

        // Validate tile alignment for prefill
        TT_FATAL(
            (batch_size * seq_len) % 32 == 0,
            "Input rows (B * S = {} * {}) must be tile-aligned (divisible by 32)",
            batch_size, seq_len);
    } else {
        // Decode mode: [1, 1, B, H]
        TT_FATAL(
            input_shape[0] == 1 && input_shape[1] == 1,
            "hidden_state shape must be [1, 1, B, H] for decode mode, got [{}, {}, {}, {}]",
            input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        batch_size = input_shape[2];
        hidden_dim = input_shape[3];

        // Validate tile alignment for decode
        TT_FATAL(
            batch_size % 32 == 0,
            "Batch size {} must be tile-aligned (divisible by 32)",
            batch_size);
    }

    // Common validations
    TT_FATAL(
        batch_size % dp == 0,
        "Batch size {} must be divisible by dp {}",
        batch_size, dp);

    uint32_t batch_per_device = batch_size / dp;

    TT_FATAL(
        batch_per_device % 32 == 0,
        "Batch per device {} must be tile-aligned (divisible by 32)",
        batch_per_device);

    TT_FATAL(
        hidden_dim % 32 == 0,
        "Hidden dimension {} must be tile-aligned (divisible by 32)",
        hidden_dim);

    // Validate dp_degree tensor: should be scalar integer tensor [1]
    const auto& dp_shape = dp_degree.padded_shape();
    TT_FATAL(
        dp_shape.rank() == 1 && dp_shape[0] == 1,
        "dp_degree must be a scalar tensor [1], got shape with rank {} and size {}",
        dp_shape.rank(), dp_shape[0]);

    TT_FATAL(
        dp_degree.dtype() == DataType::UINT32 || dp_degree.dtype() == DataType::INT32,
        "dp_degree must be integer type (UINT32 or INT32), got {}",
        dp_degree.dtype());

    // Validate output dtype
    TT_FATAL(
        output_dtype == DataType::BFLOAT16 || output_dtype == DataType::BFLOAT8_B,
        "output_dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        output_dtype);
}

std::vector<TensorSpec> ExtractAttentionInput::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& hidden_state = input_tensors.at(0);
    const auto& input_shape = hidden_state.padded_shape();

    bool is_prefill = (input_shape.rank() == 3);

    ttnn::Shape output_shape;

    if (is_prefill) {
        // Prefill mode: [B, S, H] → [B//dp, 1, S, H]
        uint32_t batch_size = input_shape[0];
        uint32_t seq_len = input_shape[1];
        uint32_t hidden_dim = input_shape[2];
        uint32_t batch_per_device = batch_size / dp;

        output_shape = ttnn::Shape({batch_per_device, 1, seq_len, hidden_dim});
    } else {
        // Decode mode: [1, 1, B, H] → [1, 1, B//dp, H]
        uint32_t batch_size = input_shape[2];
        uint32_t hidden_dim = input_shape[3];
        uint32_t batch_per_device = batch_size / dp;

        output_shape = ttnn::Shape({1, 1, batch_per_device, hidden_dim});
    }

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), output_mem_config))};
}

std::vector<Tensor> ExtractAttentionInput::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {create_device_tensor(output_specs[0], input_tensor.device())};
}

tt::tt_metal::operation::ProgramWithCallbacks ExtractAttentionInput::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& hidden_state = input_tensors.at(0);
    const auto& dp_degree = input_tensors.at(1);
    auto& output = output_tensors.at(0);

    return detail::extract_attention_input_single_core(
        hidden_state, dp_degree, output, dp, output_dtype);
}

}  // namespace ttnn::operations::experimental::attention
