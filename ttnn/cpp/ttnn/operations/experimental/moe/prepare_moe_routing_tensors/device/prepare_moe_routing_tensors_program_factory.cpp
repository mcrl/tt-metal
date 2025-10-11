// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_routing_tensors_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks prepare_moe_routing_tensors_single_core(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    Tensor& num_routed_tokens,
    Tensor& routed_tokens,
    Tensor& routed_token_weights,
    uint32_t num_experts,
    uint32_t max_tokens_per_expert) {

    Program program{};

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    IDevice* device = selected_experts.device();
    CoreCoord core = {0, 0};

    // Circular buffer indices
    const uint32_t cb_experts = CBIndex::c_0;
    const uint32_t cb_weights = CBIndex::c_1;
    const uint32_t cb_num_routed = CBIndex::c_16;
    const uint32_t cb_routed_tokens = CBIndex::c_17;
    const uint32_t cb_routed_weights = CBIndex::c_18;
    const uint32_t cb_scratch = CBIndex::c_24;

    // Data formats
    tt::DataFormat experts_data_format = tt_metal::datatype_to_dataformat_converter(selected_experts.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(routing_weights.dtype());
    tt::DataFormat num_routed_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    tt::DataFormat routed_tokens_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    tt::DataFormat routed_weights_data_format = tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);

    // Buffer sizes
    const uint32_t experts_row_bytes = top_k * sizeof(uint32_t);
    const uint32_t weights_row_bytes = top_k * sizeof(uint16_t);
    const uint32_t num_routed_bytes = num_experts * sizeof(uint32_t);
    const uint32_t routed_tokens_row_bytes = max_tokens_per_expert * sizeof(uint32_t);
    const uint32_t routed_weights_row_bytes = max_tokens_per_expert * sizeof(uint16_t);

    // Scratch buffer for collecting tokens per expert
    // Size: num_experts * max_tokens_per_expert * (sizeof(uint32_t) + sizeof(uint16_t))
    // With correct max_tokens_per_expert = T, this should fit in L1
    const uint32_t scratch_bytes = num_experts * max_tokens_per_expert * (sizeof(uint32_t) + sizeof(uint16_t));

    // Create circular buffers
    tt_metal::CircularBufferConfig experts_cb_config =
        tt_metal::CircularBufferConfig(experts_row_bytes, {{cb_experts, experts_data_format}})
            .set_page_size(cb_experts, experts_row_bytes);
    CreateCircularBuffer(program, core, experts_cb_config);

    tt_metal::CircularBufferConfig weights_cb_config =
        tt_metal::CircularBufferConfig(weights_row_bytes, {{cb_weights, weights_data_format}})
            .set_page_size(cb_weights, weights_row_bytes);
    CreateCircularBuffer(program, core, weights_cb_config);

    tt_metal::CircularBufferConfig num_routed_cb_config =
        tt_metal::CircularBufferConfig(num_routed_bytes, {{cb_num_routed, num_routed_data_format}})
            .set_page_size(cb_num_routed, num_routed_bytes);
    CreateCircularBuffer(program, core, num_routed_cb_config);

    tt_metal::CircularBufferConfig routed_tokens_cb_config =
        tt_metal::CircularBufferConfig(routed_tokens_row_bytes, {{cb_routed_tokens, routed_tokens_data_format}})
            .set_page_size(cb_routed_tokens, routed_tokens_row_bytes);
    CreateCircularBuffer(program, core, routed_tokens_cb_config);

    tt_metal::CircularBufferConfig routed_weights_cb_config =
        tt_metal::CircularBufferConfig(routed_weights_row_bytes, {{cb_routed_weights, routed_weights_data_format}})
            .set_page_size(cb_routed_weights, routed_weights_row_bytes);
    CreateCircularBuffer(program, core, routed_weights_cb_config);

    // Large scratch buffer for intermediate data
    tt_metal::CircularBufferConfig scratch_cb_config =
        tt_metal::CircularBufferConfig(scratch_bytes, {{cb_scratch, num_routed_data_format}})
            .set_page_size(cb_scratch, scratch_bytes);
    CreateCircularBuffer(program, core, scratch_cb_config);

    // Compile-time arguments for TensorAccessor
    std::vector<uint32_t> compile_time_args = {};

    // Add TensorAccessor compile-time args for all buffers
    TensorAccessorArgs(*selected_experts.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routing_weights.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routed_token_weights.buffer()).append_to(compile_time_args);

    std::vector<uint32_t> runtime_args = {
        selected_experts.buffer()->address(),
        routing_weights.buffer()->address(),
        num_routed_tokens.buffer()->address(),
        routed_tokens.buffer()->address(),
        routed_token_weights.buffer()->address(),
        num_tokens,
        top_k,
        num_experts,
        max_tokens_per_expert};

    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/prepare_moe_routing_tensors/device/kernels/dataflow/reader_writer_moe_routing.cpp",
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
        const auto& selected_experts = input_tensors[0];
        const auto& routing_weights = input_tensors[1];
        const auto& num_routed_tokens = output_tensors[0];
        const auto& routed_tokens = output_tensors[1];
        const auto& routed_token_weights = output_tensors[2];

        auto& runtime_args = GetRuntimeArgs(program, kernel, {0, 0});
        runtime_args[0] = selected_experts.buffer()->address();
        runtime_args[1] = routing_weights.buffer()->address();
        runtime_args[2] = num_routed_tokens.buffer()->address();
        runtime_args[3] = routed_tokens.buffer()->address();
        runtime_args[4] = routed_token_weights.buffer()->address();
    };

    return {std::move(program), override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe