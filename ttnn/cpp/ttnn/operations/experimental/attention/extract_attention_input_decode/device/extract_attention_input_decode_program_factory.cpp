// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_attention_input_decode_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::attention::detail {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks extract_attention_input_decode_single_core(
    const ttnn::Tensor& hidden_state,
    ttnn::Tensor& output,
    uint32_t dp,
    DataType output_dtype) {

    Program program = CreateProgram();

    const auto& input_shape = hidden_state.padded_shape();
    uint32_t batch_size = input_shape[2];
    uint32_t hidden_dim = input_shape[3];
    uint32_t batch_per_device = batch_size / dp;

    // Use single core (0, 0) for dummy implementation
    CoreCoord core = {0, 0};
    CoreRangeSet all_cores({CoreRange(core, core)});

    // Calculate tile counts (convert DataType to DataFormat)
    auto output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t tile_size = tt_metal::detail::TileSize(output_data_format);

    // Minimal circular buffer - just 1 tile for dummy
    uint32_t cb_size = tile_size;

    // Create circular buffers (minimal for dummy)
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(cb_size, {{0, tt_metal::datatype_to_dataformat_converter(hidden_state.dtype())}})
        .set_page_size(0, tile_size);
    auto cb_in = CreateCircularBuffer(program, all_cores, cb_in_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(cb_size, {{16, tt_metal::datatype_to_dataformat_converter(output_dtype)}})
        .set_page_size(16, tile_size);
    auto cb_out = CreateCircularBuffer(program, all_cores, cb_out_config);

    // Compile-time args (for testing kernel compilation)
    std::vector<uint32_t> reader_compile_args = {
        batch_size,
        hidden_dim,
        dp
    };

    std::vector<uint32_t> writer_compile_args = {
        batch_per_device,
        hidden_dim
    };

    // Create kernels (dummy kernels that do nothing)
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input_decode/device/kernels/dataflow/reader_extract_attention_input_decode.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_args));

    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input_decode/device/kernels/dataflow/writer_extract_attention_input_decode.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_args));

    // Runtime args (for testing runtime arg passing)
    std::vector<uint32_t> reader_runtime_args = {
        hidden_state.buffer()->address()
    };

    std::vector<uint32_t> writer_runtime_args = {
        output.buffer()->address()
    };

    SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);

    // Callbacks to update buffer addresses
    auto override_runtime_args_callback = [reader_kernel, writer_kernel, core](
        const void* operation,
        const Program& program,
        const std::vector<ttnn::Tensor>& input_tensors,
        const std::vector<std::optional<const ttnn::Tensor>>&,
        const std::vector<ttnn::Tensor>& output_tensors) {

        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel, core);

        reader_runtime_args[0] = input_tensors.at(0).buffer()->address();
        writer_runtime_args[0] = output_tensors.at(0).buffer()->address();
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::attention::detail
