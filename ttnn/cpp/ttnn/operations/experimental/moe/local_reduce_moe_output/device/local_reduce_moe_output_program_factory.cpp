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

tt::tt_metal::operation::ProgramWithCallbacks local_reduce_moe_output(
    const Tensor& input_hidden_state,
    const Tensor& token_idx_map,
    const Tensor& routed_token_weights,
    const Tensor& num_routed_tokens,
    uint32_t num_tokens,
    Tensor& output) {

    Program program{};

    // Get tensor shapes
    const auto& input_shape = input_hidden_state.padded_shape();
    const uint32_t num_local_experts = input_shape[-3];
    const uint32_t max_tokens = input_shape[-2];
    const uint32_t hidden_dim = input_shape[-1];

    // Get buffers
    const auto input_buffer = input_hidden_state.buffer();
    const auto token_idx_buffer = token_idx_map.buffer();
    const auto weights_buffer = routed_token_weights.buffer();
    const auto num_routed_buffer = num_routed_tokens.buffer();
    const auto output_buffer = output.buffer();

    // Determine buffer types (DRAM vs L1)
    const bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    const bool token_idx_is_dram = token_idx_buffer->buffer_type() == BufferType::DRAM;
    const bool weights_is_dram = weights_buffer->buffer_type() == BufferType::DRAM;
    const bool num_routed_is_dram = num_routed_buffer->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    // Data formats
    const DataFormat input_cb_data_format = datatype_to_dataformat_converter(input_hidden_state.dtype());
    const uint32_t input_element_size = input_hidden_state.element_size();

    const DataFormat weights_cb_data_format = datatype_to_dataformat_converter(routed_token_weights.dtype());
    const uint32_t weights_element_size = routed_token_weights.element_size();

    // Get device and calculate work distribution
    const auto device = input_hidden_state.device();
    const auto grid_size = device->compute_with_storage_grid_size();

    // Split tokens across cores
    const auto [num_cores_used, all_cores, core_range_1, core_range_2,
          tokens_per_core_1, tokens_per_core_2] =
        split_work_to_cores(grid_size, num_tokens);

    // Debug: log the configuration
    log_debug(tt::LogOp, "local_reduce_moe_output: num_tokens={}, num_cores_used={}, tokens_per_core_1={}, tokens_per_core_2={}",
              num_tokens, num_cores_used, tokens_per_core_1, tokens_per_core_2);
    log_debug(tt::LogOp, "Grid size: {}x{}", grid_size.x, grid_size.y);

    // Circular buffer sizing constants
    constexpr uint32_t INPUT_CB_NUM_PAGES = 2;   // Double buffering for read-compute overlap
    constexpr uint32_t OUTPUT_CB_NUM_PAGES = 1;  // Single output buffer

    // Circular buffers (same configuration per core)
    const uint32_t row_size_bytes = hidden_dim * input_element_size;
    const uint32_t aligned_row_size_bytes = round_up_to_mul32(row_size_bytes);

    // CB 0: Input hidden state row buffer (reader → compute, double buffered)
    constexpr uint32_t cb_id_input = tt::CBIndex::c_0;
    constexpr uint32_t input_tile_size = 1024 * sizeof(uint16_t);
    const CircularBufferConfig cb_input_config =
        CircularBufferConfig(input_tile_size * INPUT_CB_NUM_PAGES, {{cb_id_input, input_cb_data_format}})
            .set_page_size(cb_id_input, input_tile_size);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    // CB 1: Token index map buffer (reader → compute, one page per expert)
    constexpr uint32_t cb_id_token_idx = tt::CBIndex::c_1;
    const uint32_t token_idx_row_size = max_tokens * sizeof(uint32_t);
    const uint32_t aligned_token_idx_row_size = round_up_to_mul32(token_idx_row_size);
    const uint32_t aligned_token_idx_cb_size = num_local_experts * aligned_token_idx_row_size;
    const CircularBufferConfig cb_token_idx_config =
        CircularBufferConfig(aligned_token_idx_cb_size, {{cb_id_token_idx, DataFormat::UInt32}})
            .set_page_size(cb_id_token_idx, aligned_token_idx_row_size);
    CreateCircularBuffer(program, all_cores, cb_token_idx_config);

    // CB 2: Weights buffer (reader → compute, one page per expert)
    constexpr uint32_t cb_id_weights = tt::CBIndex::c_2;
    const uint32_t weights_row_size = max_tokens * sizeof(uint16_t);
    const uint32_t aligned_weights_row_size = round_up_to_mul32(weights_row_size);
    const uint32_t aligned_weights_cb_size = num_local_experts * aligned_weights_row_size;
    const CircularBufferConfig cb_weights_config =
        CircularBufferConfig(aligned_weights_cb_size, {{cb_id_weights, DataFormat::UInt16}})
            .set_page_size(cb_id_weights, aligned_weights_row_size);
    CreateCircularBuffer(program, all_cores, cb_weights_config);

    // CB 3: Num routed tokens buffer (reader → compute)
    constexpr uint32_t cb_id_num_routed = tt::CBIndex::c_3;
    const uint32_t num_routed_size = num_local_experts * sizeof(uint32_t);
    const uint32_t aligned_num_routed_size = round_up_to_mul32(num_routed_size);
    const CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(aligned_num_routed_size, {{cb_id_num_routed, DataFormat::UInt32}})
            .set_page_size(cb_id_num_routed, aligned_num_routed_size);
    CreateCircularBuffer(program, all_cores, cb_num_routed_config);

    // CB 4: Weight scalar buffer (reader → compute, single tile of bfloat16)
    constexpr uint32_t cb_id_weight_scalar = tt::CBIndex::c_4;
    constexpr uint32_t tile_size = 32 * 32 * sizeof(uint16_t);  // One tile: 32x32 bfloat16 elements
    const uint32_t aligned_weight_scalar_size = round_up_to_mul32(tile_size);
    const CircularBufferConfig cb_weight_scalar_config =
        CircularBufferConfig(aligned_weight_scalar_size, {{cb_id_weight_scalar, DataFormat::Float16_b}})
            .set_page_size(cb_id_weight_scalar, aligned_weight_scalar_size);
    CreateCircularBuffer(program, all_cores, cb_weight_scalar_config);

    // CB 5: Accumulation buffer (compute internal, for accumulating multiple expert outputs)
    constexpr uint32_t cb_id_accum = tt::CBIndex::c_5;
    const uint32_t num_tiles_per_row = (hidden_dim + 1023) / 1024;  // Ceiling division
    const uint32_t accum_cb_size = num_tiles_per_row * tile_size;
    const CircularBufferConfig cb_accum_config =
        CircularBufferConfig(accum_cb_size, {{cb_id_accum, input_cb_data_format}})
            .set_page_size(cb_id_accum, tile_size);
    CreateCircularBuffer(program, all_cores, cb_accum_config);

    // CB 24: Tilized input buffer (compute internal, for ROW_MAJOR → TILE conversion)
    // Single tile buffer - matches tilize_init(..., 1, ...) parameter
    constexpr uint32_t cb_id_input_tile = tt::CBIndex::c_24;
    const CircularBufferConfig cb_input_tile_config =
        CircularBufferConfig(tile_size, {{cb_id_input_tile, input_cb_data_format}})
            .set_page_size(cb_id_input_tile, tile_size);
    CreateCircularBuffer(program, all_cores, cb_input_tile_config);

    // CB 16: Output buffer (compute → writer)
    constexpr uint32_t cb_id_output = tt::CBIndex::c_16;
    const CircularBufferConfig cb_output_config =
        CircularBufferConfig(aligned_row_size_bytes * OUTPUT_CB_NUM_PAGES, {{cb_id_output, input_cb_data_format}})
            .set_page_size(cb_id_output, aligned_row_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    // Reader kernel compile-time arguments
    const std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)input_is_dram,
        (uint32_t)token_idx_is_dram,
        (uint32_t)weights_is_dram,
        (uint32_t)num_routed_is_dram,
        hidden_dim,
        num_local_experts,
        max_tokens,
        row_size_bytes,
    };

    // Compute kernel compile-time arguments (same for all cores)
    // Note: Each core will process a different number of tokens, handled via runtime loop
    const std::vector<uint32_t> compute_compile_time_args = {
        hidden_dim,
        num_local_experts,
        max_tokens,
        tokens_per_core_1,  // Max tokens per core (cores may process fewer)
    };

    // Writer kernel compile-time arguments
    const std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_is_dram,
        row_size_bytes,
    };

    // Create reader kernel
    const KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/local_reduce_moe_output/device/kernels/dataflow/"
        "reader_local_reduce_moe_output.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    // Create compute kernel with MATH_ONLY define to restrict execution to MATH thread
    const std::map<std::string, std::string> compute_defines = {
        {"MATH_ONLY", "1"}
    };

    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/local_reduce_moe_output/device/kernels/compute/"
        "local_reduce_moe_output.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_compile_time_args,
            .defines = compute_defines
        });

    // Create writer kernel
    const KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/local_reduce_moe_output/device/kernels/dataflow/"
        "writer_local_reduce_moe_output.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    // Set per-core runtime arguments
    uint32_t current_token = 0;
    const uint32_t g1_numcores = core_range_1.num_cores();

    // Prepare runtime args for all cores
    uint32_t core_idx = 0;
    for (const auto& core_range : all_cores.ranges()) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                if (core_idx >= num_cores_used) break;
                CoreCoord core(x, y);

                const uint32_t tokens_per_core = (core_idx < g1_numcores) ? tokens_per_core_1 : tokens_per_core_2;

                // Reader kernel runtime args
                const std::vector<uint32_t> reader_runtime_args = {
                    input_buffer->address(),
                    token_idx_buffer->address(),
                    weights_buffer->address(),
                    num_routed_buffer->address(),
                    current_token,                      // start_token_idx
                    current_token + tokens_per_core     // end_token_idx
                };
                SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

                // Compute kernel runtime args
                const std::vector<uint32_t> compute_runtime_args = {
                    tokens_per_core,  // Number of tokens this core will process
                    current_token     // Start token index for this core
                };
                SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

                // Writer kernel runtime args
                const std::vector<uint32_t> writer_runtime_args = {
                    output_buffer->address(),
                    current_token,                      // start_token_idx
                    current_token + tokens_per_core     // end_token_idx
                };
                SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

                current_token += tokens_per_core;
                core_idx++;
            }
            if (core_idx >= num_cores_used) break;
        }
        if (core_idx >= num_cores_used) break;
    }

    // Callback to update buffer addresses when tensors move
    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, all_cores, num_cores_used, core_range_1, tokens_per_core_1, tokens_per_core_2](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto input_buffer = input_tensors.at(0).buffer();
            const auto token_idx_buffer = input_tensors.at(1).buffer();
            const auto weights_buffer = input_tensors.at(2).buffer();
            const auto num_routed_buffer = input_tensors.at(3).buffer();
            const auto output_buffer = output_tensors.at(0).buffer();

            // Update runtime args for each core
            uint32_t current_token = 0;
            const uint32_t g1_numcores = core_range_1.num_cores();

            uint32_t core_idx = 0;
            for (const auto& core_range : all_cores.ranges()) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                        if (core_idx >= num_cores_used) return;
                        const CoreCoord core(x, y);

                        const uint32_t tokens_per_core = (core_idx < g1_numcores) ? tokens_per_core_1 : tokens_per_core_2;

                        // Update reader kernel args
                        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, core);
                        reader_args[0] = input_buffer->address();
                        reader_args[1] = token_idx_buffer->address();
                        reader_args[2] = weights_buffer->address();
                        reader_args[3] = num_routed_buffer->address();
                        reader_args[4] = current_token;
                        reader_args[5] = current_token + tokens_per_core;

                        // Update writer kernel args
                        auto& writer_args = GetRuntimeArgs(program, writer_kernel_id, core);
                        writer_args[0] = output_buffer->address();
                        writer_args[1] = current_token;
                        writer_args[2] = current_token + tokens_per_core;

                        current_token += tokens_per_core;
                        core_idx++;
                    }
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::moe::detail
