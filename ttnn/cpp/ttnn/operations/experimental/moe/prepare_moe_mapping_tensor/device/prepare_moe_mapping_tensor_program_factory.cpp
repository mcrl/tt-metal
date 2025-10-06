// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_mapping_tensor_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::moe::detail {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks prepare_moe_mapping_tensor_single_core(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    Tensor& output,
    uint32_t num_experts) {
    Program program{};

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    IDevice* device = selected_experts.device();
    CoreCoord core = {0, 0};

    // Circular buffers
    const uint32_t experts_cb_index = CBIndex::c_0;
    const uint32_t weights_cb_index = CBIndex::c_1;
    const uint32_t output_cb_index = CBIndex::c_16;

    tt::DataFormat experts_data_format = tt_metal::datatype_to_dataformat_converter(selected_experts.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(routing_weights.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    // For ROW_MAJOR data, CB size should match the actual row size, not tile size
    const uint32_t experts_row_bytes = top_k * sizeof(uint32_t);
    const uint32_t weights_row_bytes = top_k * sizeof(uint16_t);
    const uint32_t padded_num_experts = (num_experts + 31) & ~31;
    const uint32_t output_row_bytes = padded_num_experts * sizeof(uint16_t);

    // Create circular buffers with sizes matching row data
    tt_metal::CircularBufferConfig experts_cb_config =
        tt_metal::CircularBufferConfig(experts_row_bytes, {{experts_cb_index, experts_data_format}})
            .set_page_size(experts_cb_index, experts_row_bytes);
    auto cb_experts = tt_metal::CreateCircularBuffer(program, core, experts_cb_config);

    tt_metal::CircularBufferConfig weights_cb_config =
        tt_metal::CircularBufferConfig(weights_row_bytes, {{weights_cb_index, weights_data_format}})
            .set_page_size(weights_cb_index, weights_row_bytes);
    auto cb_weights = tt_metal::CreateCircularBuffer(program, core, weights_cb_config);

    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(output_row_bytes, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_row_bytes);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, output_cb_config);

    // Compile-time arguments for TensorAccessor
    std::vector<uint32_t> compile_time_args = {};

    // Add TensorAccessor compile-time args for buffers in order: experts, weights, output
    // Each accessor needs to know the page size (row size) for proper DRAM addressing
    TensorAccessorArgs(*selected_experts.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routing_weights.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(compile_time_args);

    std::vector<uint32_t> runtime_args = {
        selected_experts.buffer()->address(),
        routing_weights.buffer()->address(),
        output.buffer()->address(),
        num_tokens,
        top_k,
        num_experts};

    // TODO: Consider using multiple cores for better parallelism
    // Current implementation uses single core (0,0) - inefficient for large batches
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/prepare_moe_mapping_tensor/device/kernels/dataflow/reader_writer_moe_mapping.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_time_args});

    SetRuntimeArgs(program, kernel, core, runtime_args);

    auto override_runtime_arguments_callback = [kernel](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& selected_experts = input_tensors[0];
        const auto& routing_weights = input_tensors[1];
        const auto& output = output_tensors[0];

        auto& runtime_args = GetRuntimeArgs(program, kernel, {0, 0});
        runtime_args[0] = selected_experts.buffer()->address();
        runtime_args[1] = routing_weights.buffer()->address();
        runtime_args[2] = output.buffer()->address();
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe::detail
