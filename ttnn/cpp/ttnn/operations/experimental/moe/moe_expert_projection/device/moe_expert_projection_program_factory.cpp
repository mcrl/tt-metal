// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_projection_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks moe_expert_projection_single_core(
    const Tensor& hidden_states,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& expert_weights,
    const Tensor& device_expert_mapping,
    Tensor& output,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t expert_dim,
    uint32_t experts_per_device,
    uint32_t max_tokens_per_expert,
    uint32_t output_size) {

    Program program{};

    IDevice* device = hidden_states.device();
    CoreCoord core = {0, 0};

    // Circular buffer indices
    const uint32_t cb_hidden_row = CBIndex::c_0;           // Input: hidden states row (on-demand)
    const uint32_t cb_routed_row = CBIndex::c_1;           // Input: routed tokens row
    const uint32_t cb_num_routed_row = CBIndex::c_2;       // Input: num routed tokens row
    const uint32_t cb_mapping = CBIndex::c_3;              // Input: device expert mapping
    const uint32_t cb_expert_weights = CBIndex::c_4;       // Input: expert weight row (H' elements)
    const uint32_t cb_output_row = CBIndex::c_16;          // Output: projection output row

    // Data formats
    tt::DataFormat hidden_data_format = tt_metal::datatype_to_dataformat_converter(hidden_states.dtype());
    tt::DataFormat routed_data_format = tt_metal::datatype_to_dataformat_converter(routed_tokens.dtype());
    tt::DataFormat num_routed_data_format = tt_metal::datatype_to_dataformat_converter(num_routed_tokens.dtype());
    tt::DataFormat mapping_data_format = tt_metal::datatype_to_dataformat_converter(device_expert_mapping.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(expert_weights.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);

    // Buffer sizes (use logical dimensions from runtime args)
    const uint32_t hidden_row_bytes = hidden_dim * sizeof(uint16_t);
    const uint32_t routed_row_bytes = max_tokens_per_expert * sizeof(uint32_t);
    // num_routed_tokens has shape (1, E_padded) - need to read entire expert count array
    // Pad to next power of 2 for E to match padded shape
    const uint32_t num_experts_padded = (num_routed_tokens.padded_shape().rank() == 2) ?
        num_routed_tokens.padded_shape()[1] : num_routed_tokens.padded_shape()[0];
    const uint32_t num_routed_row_bytes = num_experts_padded * sizeof(uint32_t);
    const uint32_t mapping_bytes = experts_per_device * sizeof(int32_t);
    const uint32_t output_row_bytes = expert_dim * sizeof(uint16_t);
    const uint32_t expert_weights_row_bytes = expert_dim * sizeof(uint16_t);

    // Create circular buffers
    tt_metal::CircularBufferConfig hidden_cb_config =
        tt_metal::CircularBufferConfig(hidden_row_bytes, {{cb_hidden_row, hidden_data_format}})
            .set_page_size(cb_hidden_row, hidden_row_bytes);
    CreateCircularBuffer(program, core, hidden_cb_config);

    tt_metal::CircularBufferConfig routed_cb_config =
        tt_metal::CircularBufferConfig(routed_row_bytes, {{cb_routed_row, routed_data_format}})
            .set_page_size(cb_routed_row, routed_row_bytes);
    CreateCircularBuffer(program, core, routed_cb_config);

    tt_metal::CircularBufferConfig num_routed_cb_config =
        tt_metal::CircularBufferConfig(num_routed_row_bytes, {{cb_num_routed_row, num_routed_data_format}})
            .set_page_size(cb_num_routed_row, num_routed_row_bytes);
    CreateCircularBuffer(program, core, num_routed_cb_config);

    tt_metal::CircularBufferConfig mapping_cb_config =
        tt_metal::CircularBufferConfig(mapping_bytes, {{cb_mapping, mapping_data_format}})
            .set_page_size(cb_mapping, mapping_bytes);
    CreateCircularBuffer(program, core, mapping_cb_config);

    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(output_row_bytes, {{cb_output_row, output_data_format}})
            .set_page_size(cb_output_row, output_row_bytes);
    CreateCircularBuffer(program, core, output_cb_config);

    // Buffer size for weights row + FP32 accumulators
    uint32_t weights_cb_size = expert_weights_row_bytes + (expert_dim * sizeof(float));
    tt_metal::CircularBufferConfig weights_cb_config =
        tt_metal::CircularBufferConfig(weights_cb_size, {{cb_expert_weights, weights_data_format}})
            .set_page_size(cb_expert_weights, weights_cb_size);
    CreateCircularBuffer(program, core, weights_cb_config);

    // Compile-time arguments for TensorAccessor
    std::vector<uint32_t> compile_time_args = {};

    // Add TensorAccessor compile-time args for all buffers
    TensorAccessorArgs(*hidden_states.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*expert_weights.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*device_expert_mapping.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(compile_time_args);

    std::vector<uint32_t> runtime_args = {
        hidden_states.buffer()->address(),
        routed_tokens.buffer()->address(),
        num_routed_tokens.buffer()->address(),
        expert_weights.buffer()->address(),
        device_expert_mapping.buffer()->address(),
        output.buffer()->address(),
        num_tokens,
        hidden_dim,
        expert_dim,
        experts_per_device,
        max_tokens_per_expert,
        output_size,
        num_experts_padded};

    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_expert_projection/device/kernels/dataflow/reader_writer_expert_projection.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    SetRuntimeArgs(program, kernel, core, runtime_args);

    auto override_runtime_arguments_callback = [kernel](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& hidden_states = input_tensors[0];
        const auto& routed_tokens = input_tensors[1];
        const auto& num_routed_tokens = input_tensors[2];
        const auto& expert_weights = input_tensors[3];
        const auto& device_expert_mapping = input_tensors[4];
        const auto& output = output_tensors[0];

        auto& runtime_args = GetRuntimeArgs(program, kernel, {0, 0});
        runtime_args[0] = hidden_states.buffer()->address();
        runtime_args[1] = routed_tokens.buffer()->address();
        runtime_args[2] = num_routed_tokens.buffer()->address();
        runtime_args[3] = expert_weights.buffer()->address();
        runtime_args[4] = device_expert_mapping.buffer()->address();
        runtime_args[5] = output.buffer()->address();
    };

    return {std::move(program), override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe
