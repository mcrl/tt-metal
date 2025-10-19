// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "local_reduce_moe_output_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::experimental::moe::detail {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::operation::ProgramWithCallbacks local_reduce_moe_output_multi_core(
    const Tensor& input_hidden_state,
    const Tensor& token_idx_map,
    const Tensor& routed_token_weights,
    const Tensor& num_routed_tokens,
    uint32_t num_tokens,
    Tensor& output) {

    Program program{};

    // Get tensor shapes
    const auto& input_shape = input_hidden_state.padded_shape();
    uint32_t num_local_experts = input_shape[-3];
    uint32_t max_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    // Get buffers
    auto input_buffer = input_hidden_state.buffer();
    auto token_idx_buffer = token_idx_map.buffer();
    auto weights_buffer = routed_token_weights.buffer();
    auto num_routed_buffer = num_routed_tokens.buffer();
    auto output_buffer = output.buffer();

    // Determine buffer types (DRAM vs L1)
    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool token_idx_is_dram = token_idx_buffer->buffer_type() == BufferType::DRAM;
    bool weights_is_dram = weights_buffer->buffer_type() == BufferType::DRAM;
    bool num_routed_is_dram = num_routed_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    // Data formats
    DataFormat input_cb_data_format = datatype_to_dataformat_converter(input_hidden_state.dtype());
    uint32_t input_element_size = input_hidden_state.element_size();

    DataFormat weights_cb_data_format = datatype_to_dataformat_converter(routed_token_weights.dtype());
    uint32_t weights_element_size = routed_token_weights.element_size();

    // Get device and calculate work distribution
    auto device = input_hidden_state.device();
    auto grid_size = device->compute_with_storage_grid_size();

    // Split tokens across cores
    auto [num_cores_used, all_cores, core_range_1, core_range_2,
          tokens_per_core_1, tokens_per_core_2] =
        split_work_to_cores(grid_size, num_tokens);

    // Debug: log the configuration
    log_debug(tt::LogOp, "local_reduce_moe_output_multi_core: num_tokens={}, num_cores_used={}, tokens_per_core_1={}, tokens_per_core_2={}",
              num_tokens, num_cores_used, tokens_per_core_1, tokens_per_core_2);
    log_debug(tt::LogOp, "Grid size: {}x{}", grid_size.x, grid_size.y);

    // Circular buffers (same configuration per core)
    uint32_t row_size_bytes = hidden_dim * input_element_size;
    uint32_t aligned_row_size_bytes = round_up_to_mul32(row_size_bytes);

    // CB 0: Input hidden state row buffer (one row from expert output)
    uint32_t cb_id_input = tt::CBIndex::c_0;
    CircularBufferConfig cb_input_config =
        CircularBufferConfig(aligned_row_size_bytes * 2, {{cb_id_input, input_cb_data_format}})
            .set_page_size(cb_id_input, aligned_row_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    // CB 1: Output accumulator buffer (one row for output token)
    uint32_t cb_id_output = tt::CBIndex::c_1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(aligned_row_size_bytes, {{cb_id_output, input_cb_data_format}})
            .set_page_size(cb_id_output, aligned_row_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    // CB 2: Routing info buffer (token_idx_map + weights row for one expert)
    uint32_t cb_id_weight = tt::CBIndex::c_2;
    uint32_t routing_buffer_size = max_tokens * sizeof(uint32_t) + max_tokens * sizeof(uint16_t);
    uint32_t aligned_routing_buffer_size = round_up_to_mul32(routing_buffer_size);
    CircularBufferConfig cb_weight_config =
        CircularBufferConfig(aligned_routing_buffer_size, {{cb_id_weight, DataFormat::UInt32}})
            .set_page_size(cb_id_weight, aligned_routing_buffer_size);
    CreateCircularBuffer(program, all_cores, cb_weight_config);

    // Compile-time arguments (same for all cores)
    std::vector<uint32_t> compile_time_args = {
        cb_id_input,
        cb_id_output,
        cb_id_weight,
        (uint32_t)input_is_dram,
        (uint32_t)token_idx_is_dram,
        (uint32_t)weights_is_dram,
        (uint32_t)num_routed_is_dram,
        (uint32_t)output_is_dram,
        hidden_dim,
        num_tokens,
        num_local_experts,
        max_tokens,
        row_size_bytes,
    };

    // Create kernel on all active cores
    KernelHandle reader_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/local_reduce_moe_output/device/kernels/dataflow/"
        "reader_writer_local_reduce_moe_output.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args));

    // Set per-core runtime arguments
    uint32_t current_token = 0;
    uint32_t g1_numcores = core_range_1.num_cores();

    // Prepare runtime args first
    std::vector<std::vector<uint32_t>> all_runtime_args;
    all_runtime_args.reserve(num_cores_used);

    for (uint32_t i = 0; i < num_cores_used; ++i) {
        uint32_t tokens_per_core;
        if (i < g1_numcores) {
            tokens_per_core = tokens_per_core_1;
        } else {
            tokens_per_core = tokens_per_core_2;
        }

        std::vector<uint32_t> runtime_args = {
            input_buffer->address(),
            token_idx_buffer->address(),
            weights_buffer->address(),
            num_routed_buffer->address(),
            output_buffer->address(),
            current_token,                      // start_token_idx
            current_token + tokens_per_core     // end_token_idx
        };
        all_runtime_args.push_back(runtime_args);
        current_token += tokens_per_core;
    }

    // Now set them for each core - use all_cores directly
    uint32_t core_idx = 0;
    for (const auto& core_range : all_cores.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                if (core_idx >= num_cores_used) break;
                CoreCoord core(x, y);
                SetRuntimeArgs(program, reader_writer_kernel_id, core, all_runtime_args[core_idx]);
                core_idx++;
            }
            if (core_idx >= num_cores_used) break;
        }
        if (core_idx >= num_cores_used) break;
    }

    // Callback to update buffer addresses when tensors move
    auto override_runtime_args_callback =
        [reader_writer_kernel_id, all_cores, num_cores_used, core_range_1, tokens_per_core_1, tokens_per_core_2](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_buffer = input_tensors.at(0).buffer();
            auto token_idx_buffer = input_tensors.at(1).buffer();
            auto weights_buffer = input_tensors.at(2).buffer();
            auto num_routed_buffer = input_tensors.at(3).buffer();
            auto output_buffer = output_tensors.at(0).buffer();

            // Update runtime args for each core
            uint32_t current_token = 0;
            uint32_t g1_numcores = core_range_1.num_cores();

            uint32_t core_idx = 0;
            for (const auto& core_range : all_cores.ranges()) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        if (core_idx >= num_cores_used) return;
                        CoreCoord core(x, y);

                        uint32_t tokens_per_core;
                        if (core_idx < g1_numcores) {
                            tokens_per_core = tokens_per_core_1;
                        } else {
                            tokens_per_core = tokens_per_core_2;
                        }

                        auto& runtime_args = GetRuntimeArgs(program, reader_writer_kernel_id, core);
                        runtime_args[0] = input_buffer->address();
                        runtime_args[1] = token_idx_buffer->address();
                        runtime_args[2] = weights_buffer->address();
                        runtime_args[3] = num_routed_buffer->address();
                        runtime_args[4] = output_buffer->address();
                        runtime_args[5] = current_token;
                        runtime_args[6] = current_token + tokens_per_core;
                        current_token += tokens_per_core;
                        core_idx++;
                    }
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::moe::detail
