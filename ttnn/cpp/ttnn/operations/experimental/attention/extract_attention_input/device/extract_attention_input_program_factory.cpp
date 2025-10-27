// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_attention_input_program_factory.hpp"

#include <fmt/format.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::attention::detail {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks extract_attention_input_single_core(
    const ttnn::Tensor& hidden_state,
    const ttnn::Tensor& dp_degree,
    ttnn::Tensor& output,
    uint32_t dp,
    DataType output_dtype) {

    Program program = CreateProgram();

    // Get input dimensions and detect mode
    const auto& input_shape = hidden_state.padded_shape();
    bool is_prefill = (input_shape.rank() == 3);

    uint32_t batch_size, hidden_dim, num_tile_rows_per_device;

    if (is_prefill) {
        // Prefill mode: [B, S, H] → [B//dp, 1, S, H]
        batch_size = input_shape[0];
        uint32_t seq_len = input_shape[1];
        hidden_dim = input_shape[2];
        uint32_t batch_per_device = batch_size / dp;
        num_tile_rows_per_device = (batch_per_device * seq_len) / 32;
    } else {
        // Decode mode: [1, 1, B, H] → [1, 1, B//dp, H]
        batch_size = input_shape[2];
        hidden_dim = input_shape[3];
        uint32_t batch_per_device = batch_size / dp;
        num_tile_rows_per_device = batch_per_device / 32;
    }

    // Calculate tile dimensions (common for both modes)
    constexpr uint32_t TILE_SIZE = 32;
    uint32_t num_tile_cols = hidden_dim / TILE_SIZE;
    uint32_t tiles_per_device = num_tile_rows_per_device * num_tile_cols;

    // Detect if format conversion is needed
    bool needs_format_conversion = (output_dtype != hidden_state.dtype());

    // Calculate tile sizes for input and output formats
    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(hidden_state.dtype());
    uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format);

    // Get buffer info
    auto input_buffer = hidden_state.buffer();
    auto dp_degree_buffer = dp_degree.buffer();
    auto output_buffer = output.buffer();
    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool dp_degree_is_dram = dp_degree_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    // Multi-core configuration: 8x8 grid (64 cores)
    constexpr uint32_t num_cores_x = 8;
    constexpr uint32_t num_cores_y = 8;
    constexpr uint32_t total_cores = num_cores_x * num_cores_y;  // 64 cores

    CoreRange all_cores_range({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    CoreRangeSet all_cores({all_cores_range});

    // Manual work distribution across cores
    uint32_t tiles_per_core = tiles_per_device / total_cores;
    uint32_t remainder_tiles = tiles_per_device % total_cores;
    // First 'remainder_tiles' cores get (tiles_per_core + 1) tiles
    // Remaining cores get tiles_per_core tiles

    // Create circular buffers - 2 tiles for double buffering
    // CB 0: Input buffer (bfloat16)
    uint32_t num_input_tiles = 2;  // Double buffer for pipelining
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(num_input_tiles * input_tile_size, {{0, input_data_format}})
        .set_page_size(0, input_tile_size);
    auto cb_in = CreateCircularBuffer(program, all_cores, cb_in_config);

    // CB 1: dp_degree buffer (uint32, single value)
    CircularBufferConfig cb_dp_degree_config =
        CircularBufferConfig(sizeof(uint32_t), {{1, tt_metal::datatype_to_dataformat_converter(dp_degree.dtype())}})
        .set_page_size(1, sizeof(uint32_t));
    auto cb_dp_degree = CreateCircularBuffer(program, all_cores, cb_dp_degree_config);

    // CB 16: Output buffer (only needed when format conversion is required)
    if (needs_format_conversion) {
        tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype);
        uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format);
        uint32_t num_output_tiles = 2;  // Double buffer for pipelining
        CircularBufferConfig cb_out_config =
            CircularBufferConfig(num_output_tiles * output_tile_size, {{16, output_data_format}})
            .set_page_size(16, output_tile_size);
        auto cb_out = CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    // Compile-time args for reader (TensorAccessor args + dp_degree buffer type)
    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_args);
    reader_compile_args.push_back((uint32_t)dp_degree_is_dram);  // dp_degree buffer type

    // Compile-time args for writer (TensorAccessor pattern + CB index + data format)
    tt::DataFormat output_data_format = needs_format_conversion
        ? tt_metal::datatype_to_dataformat_converter(output_dtype)
        : input_data_format;

    std::vector<uint32_t> writer_compile_args = {
        needs_format_conversion ? 16u : 0u  // 0: CB index (0 for 2-kernel, 16 for 3-kernel)
    };
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
    writer_compile_args.push_back((uint32_t)output_data_format);  // data format

    // Create kernels
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input/device/kernels/dataflow/reader_extract_attention_input.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_args));

    // Conditionally create compute kernel for format conversion
    KernelHandle compute_kernel = 0;  // Initialize to silence warning
    if (needs_format_conversion) {
        // Define TYPECAST_LLK macro for format conversion
        // This expands to: typecast_tile<input_format_enum, output_format_enum>
        std::map<std::string, std::string> compute_defines;
        compute_defines["TYPECAST_LLK"] = fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(input_data_format),
            static_cast<uint32_t>(tt_metal::datatype_to_dataformat_converter(output_dtype)));

        // Note: num_tiles is now a runtime arg, no compile args needed
        std::vector<uint32_t> compute_compile_args = {};

        compute_kernel = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input/device/kernels/compute/copy_tiles_format_conversion.cpp",
            all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .bfp8_pack_precise = false,
                .math_approx_mode = false,
                .compile_args = compute_compile_args,
                .defines = compute_defines
            });
    }

    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input/device/kernels/dataflow/writer_extract_attention_input.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_args));

    // Set runtime args for each core
    // Note: Each reader will read dp_degree and calculate global start offset on device
    uint32_t num_tiles_assigned = 0;
    for (uint32_t core_idx = 0; core_idx < total_cores; core_idx++) {
        CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

        // Calculate tiles for this core
        uint32_t num_tiles_for_core = tiles_per_core;
        if (core_idx < remainder_tiles) {
            num_tiles_for_core += 1;  // First 'remainder' cores get +1 tile
        }

        // Tile offset within device's assigned tiles
        uint32_t local_tile_offset = num_tiles_assigned;
        uint32_t output_start_tile_id = num_tiles_assigned;

        // Reader runtime args: input_addr, dp_degree_addr, num_tiles, local_offset, tiles_per_device
        SetRuntimeArgs(program, reader_kernel, core, {
            input_buffer->address(),
            dp_degree_buffer->address(),
            num_tiles_for_core,
            local_tile_offset,
            tiles_per_device
        });

        // Writer runtime args: output_addr, num_tiles, start_tile_id
        SetRuntimeArgs(program, writer_kernel, core, {
            output_buffer->address(),
            num_tiles_for_core,
            output_start_tile_id
        });

        // Compute runtime args (if format conversion): num_tiles
        if (needs_format_conversion) {
            SetRuntimeArgs(program, compute_kernel, core, {
                num_tiles_for_core
            });
        }

        num_tiles_assigned += num_tiles_for_core;
    }

    // Callbacks to update buffer addresses when tensors move
    auto override_runtime_args_callback = [
        reader_kernel, writer_kernel, compute_kernel,
        needs_format_conversion, num_cores_y, total_cores](
        const void* operation,
        const Program& program,
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::vector<std::optional<const ttnn::Tensor>>&,
        const std::vector<ttnn::Tensor>& output_tensors) {

        auto src_buffer_addr = input_tensors.at(0).buffer()->address();
        auto dst_buffer_addr = output_tensors.at(0).buffer()->address();

        // Update buffer addresses for all cores
        for (uint32_t core_idx = 0; core_idx < total_cores; core_idx++) {
            CoreCoord core = {core_idx / num_cores_y, core_idx % num_cores_y};

            auto& reader_args = GetRuntimeArgs(program, reader_kernel, core);
            reader_args[0] = src_buffer_addr;

            auto& writer_args = GetRuntimeArgs(program, writer_kernel, core);
            writer_args[0] = dst_buffer_addr;
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::attention::detail
