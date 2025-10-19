// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_bmm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace ttnn;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks moe_bmm_single_core(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    Tensor& output,
    uint32_t num_experts,
    uint32_t max_tokens,
    uint32_t h_in,
    uint32_t h_out) {

    Program program{};

    IDevice* device = input.device();
    CoreCoord core = {0, 0};

    // Calculate tile dimensions
    uint32_t Kt = h_in / tt::constants::TILE_WIDTH;      // Number of tiles in K dimension
    uint32_t Nt = h_out / tt::constants::TILE_WIDTH;     // Number of tiles in N dimension (output columns)
    uint32_t Mt_max = max_tokens / tt::constants::TILE_HEIGHT;  // Maximum number of tiles in M dimension (tokens)

    // Circular buffer indices
    const uint32_t cb_in0 = CBIndex::c_0;            // Input tiles from input tensor
    const uint32_t cb_in1 = CBIndex::c_1;            // Weight tiles from weights tensor
    const uint32_t cb_num_routed = CBIndex::c_2;     // num_routed_tokens values
    const uint32_t cb_num_rows = CBIndex::c_3;       // num_rows buffer
    const uint32_t cb_out = CBIndex::c_16;           // Output tiles

    // Data formats
    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat num_routed_data_format = tt_metal::datatype_to_dataformat_converter(num_routed_tokens.dtype());
    tt::DataFormat num_rows_data_format = tt_metal::datatype_to_dataformat_converter(num_routed_tokens.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Tile sizes
    uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format);
    uint32_t weights_tile_size = tt_metal::detail::TileSize(weights_data_format);
    uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format);
    uint32_t num_routed_element_size = sizeof(uint32_t);

    // Create circular buffers
    // cb_in0: Input tiles (double buffered)
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(2 * input_tile_size, {{cb_in0, input_data_format}})
            .set_page_size(cb_in0, input_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    // cb_in1: Weight tiles (double buffered)
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(2 * weights_tile_size, {{cb_in1, weights_data_format}})
            .set_page_size(cb_in1, weights_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // cb_num_routed: num_routed_tokens values (one element per expert)
    CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(num_routed_element_size, {{cb_num_routed, num_routed_data_format}})
            .set_page_size(cb_num_routed, num_routed_element_size);
    CreateCircularBuffer(program, core, cb_num_routed_config);

    CircularBufferConfig cb_num_rows_config =
        CircularBufferConfig(sizeof(uint32_t), {{cb_num_rows, num_rows_data_format}})
            .set_page_size(cb_num_rows, sizeof(uint32_t));
    CreateCircularBuffer(program, core, cb_num_rows_config);

    // cb_out: Output tiles (double buffered)
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * output_tile_size, {{cb_out, output_data_format}})
            .set_page_size(cb_out, output_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // Compile-time arguments for TensorAccessor
    std::vector<uint32_t> reader_writer_compile_time_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(reader_writer_compile_time_args);
    TensorAccessorArgs(*weights.buffer()).append_to(reader_writer_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(reader_writer_compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(reader_writer_compile_time_args);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
        num_experts,  // Number of experts to process
        Mt_max,  // Maximum number of output tile rows per expert
        Kt,      // Number of tiles in K dimension
        Nt       // Number of tiles in N dimension (output columns)
    };

    // Runtime arguments
    std::vector<uint32_t> reader_writer_runtime_args = {
        input.buffer()->address(),
        weights.buffer()->address(),
        num_routed_tokens.buffer()->address(),
        output.buffer()->address(),
        num_experts,
        max_tokens,
        h_in,
        h_out,
        Kt,
        Nt,
        Mt_max
    };

    // Create kernels
    auto reader_writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/reader_writer_moe_bmm.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_writer_compile_time_args});

    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/compute/moe_bmm.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_time_args});

    // Set runtime arguments
    SetRuntimeArgs(program, reader_writer_id, core, reader_writer_runtime_args);

    auto override_runtime_arguments_callback = [reader_writer_id, compute_id](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) {

        auto input_buffer = input_tensors.at(0).buffer();
        auto weights_buffer = input_tensors.at(1).buffer();
        auto num_routed_buffer = input_tensors.at(2).buffer();
        auto output_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        auto& reader_writer_runtime_args = GetRuntimeArgs(program, reader_writer_id, core);
        reader_writer_runtime_args[0] = input_buffer->address();
        reader_writer_runtime_args[1] = weights_buffer->address();
        reader_writer_runtime_args[2] = num_routed_buffer->address();
        reader_writer_runtime_args[3] = output_buffer->address();
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks moe_bmm_multi_core(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    Tensor& output,
    uint32_t num_experts,
    uint32_t max_tokens,
    uint32_t h_in,
    uint32_t h_out) {

    Program program{};

    IDevice* device = input.device();

    // Calculate tile dimensions
    uint32_t Kt = h_in / tt::constants::TILE_WIDTH;      // Number of tiles in K dimension
    uint32_t Nt = h_out / tt::constants::TILE_WIDTH;     // Number of tiles in N dimension (output columns)
    uint32_t Mt_max = max_tokens / tt::constants::TILE_HEIGHT;  // Maximum number of tiles in M dimension (tokens)

    // Get the compute grid and split work across cores
    auto core_grid = device->compute_with_storage_grid_size();

    CoreRangeSet all_cores;
    uint32_t num_cores_x = core_grid.x, num_cores_y = core_grid.y;
    all_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Circular buffer indices
    const uint32_t cb_in0 = CBIndex::c_0;            // Input tiles from input tensor
    const uint32_t cb_in1 = CBIndex::c_1;            // Weight tiles from weights tensor
    const uint32_t cb_num_tiles = CBIndex::c_2;       // num_tiled_tokens values
    const uint32_t cb_out = CBIndex::c_16;           // Output tiles

    // Data formats
    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Tile sizes
    uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format);
    uint32_t weights_tile_size = tt_metal::detail::TileSize(weights_data_format);
    uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format);

    // Create circular buffers on all cores (2 tiles for double buffering)
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(2 * input_tile_size, {{cb_in0, input_data_format}})
            .set_page_size(cb_in0, input_tile_size);
    CreateCircularBuffer(program, all_cores, cb_in0_config);

    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(2 * weights_tile_size, {{cb_in1, weights_data_format}})
            .set_page_size(cb_in1, weights_tile_size);
    CreateCircularBuffer(program, all_cores, cb_in1_config);

    CircularBufferConfig cb_num_tiles_config =
        CircularBufferConfig(sizeof(uint32_t), {{cb_num_tiles, tt::DataFormat::UInt32}})
            .set_page_size(cb_num_tiles, sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_num_tiles_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * output_tile_size, {{cb_out, output_data_format}})
            .set_page_size(cb_out, output_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*weights.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(reader_compile_time_args);

    // Compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(writer_compile_time_args);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
        Kt,      // Number of tiles in K dimension
        Nt       // Number of tiles in N dimension (output columns)
    };

    // Create reader kernel (RISCV_1)
    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/reader_moe_bmm_multi_core.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Create writer kernel (RISCV_0)
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/writer_moe_bmm_multi_core.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/compute/moe_bmm_multi_core.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_time_args});

    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            // Reader kernel arguments
            std::vector<uint32_t> reader_runtime_args = {
                input.buffer()->address(),
                weights.buffer()->address(),
                num_routed_tokens.buffer()->address(),
                num_experts,
                max_tokens,
                h_in,
                h_out,
                Kt,
                Nt,
                Mt_max,
                core.x,
                core.y
            };

            // Writer kernel arguments
            std::vector<uint32_t> writer_runtime_args = {
                output.buffer()->address(),
                num_routed_tokens.buffer()->address(),
                num_experts,
                Nt,
                Mt_max,
                core.x,
                core.y
            };

            // Compute kernel arguments
            std::vector<uint32_t> compute_runtime_args = {
                core.x,
                core.y
            };

            SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
            SetRuntimeArgs(program, writer_id, core, writer_runtime_args);
            SetRuntimeArgs(program, compute_id, core, compute_runtime_args);
        }
    }

    auto override_runtime_arguments_callback = [reader_id, writer_id, compute_id, all_cores](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) {

        auto input_buffer = input_tensors.at(0).buffer();
        auto weights_buffer = input_tensors.at(1).buffer();
        auto num_routed_buffer = input_tensors.at(2).buffer();
        auto output_buffer = output_tensors.at(0).buffer();

        for (const auto& range : all_cores.ranges()) {
            for (const auto& core : range) {
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_id, core);
                reader_runtime_args[0] = input_buffer->address();
                reader_runtime_args[1] = weights_buffer->address();
                reader_runtime_args[2] = num_routed_buffer->address();

                auto& writer_runtime_args = GetRuntimeArgs(program, writer_id, core);
                writer_runtime_args[0] = output_buffer->address();
                writer_runtime_args[1] = num_routed_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe
