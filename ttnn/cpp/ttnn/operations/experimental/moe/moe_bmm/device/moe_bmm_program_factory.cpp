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
    all_cores = CoreRangeSet(CoreRange({0, 0}, {8 - 1, 8 - 1}));

    // Circular buffer indices
    const uint32_t cb_in0 = CBIndex::c_0;            // Input tiles from input tensor
    const uint32_t cb_in1 = CBIndex::c_1;            // Weight tiles from weights tensor
    const uint32_t cb_num_tiles = CBIndex::c_2;      // Total output tiles (single uint32_t)
    const uint32_t cb_num_routed = CBIndex::c_3;     // num_routed_tokens array (num_experts * uint32_t)
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

    // CB for shared num_routed_tokens data (num_experts * uint32_t)
    CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(num_experts * sizeof(uint32_t), {{cb_num_routed, tt::DataFormat::UInt32}})
            .set_page_size(cb_num_routed, num_experts * sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_num_routed_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * output_tile_size, {{cb_out, output_data_format}})
            .set_page_size(cb_out, output_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    // Create unified semaphores for multicast (shared by all cores)
    auto sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*weights.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back((uint32_t)sender_semaphore);
    reader_compile_time_args.push_back((uint32_t)receiver_semaphore);

    // Compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
        Kt,      // Number of tiles in K dimension
        Nt       // Number of tiles in N dimension (output columns)
    };


    // Create reader kernel (RISCV_0)
    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/reader_moe_bmm_multi_core.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // Create writer kernel (RISCV_1)
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/writer_moe_bmm_multi_core.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/compute/moe_bmm_multi_core.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
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
                Mt_max
            };

            // Writer kernel arguments
            std::vector<uint32_t> writer_runtime_args = {
                output.buffer()->address(),
                num_experts,
                Nt,
                Mt_max
            };

            // Compute kernel arguments (no runtime args needed - gets coords internally)
            std::vector<uint32_t> compute_runtime_args = {};

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
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks moe_bmm_multi_core_optimized(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    Tensor& output,
    uint32_t num_experts,
    uint32_t max_tokens,
    uint32_t h_in,
    uint32_t h_out) {

    Program program{};

    const uint32_t TILE_SIZE = 32;
    IDevice* device = input.device();

    // Calculate tile dimensions
    uint32_t Kt = h_in / TILE_SIZE;      // Number of tiles in K dimension
    uint32_t Nt = h_out / TILE_SIZE;     // Number of tiles in N dimension (output columns)
    uint32_t Mt_max = max_tokens / TILE_SIZE;  // Maximum number of tiles in M dimension (tokens)

    // Get the compute grid and split work across cores
    auto core_grid = device->compute_with_storage_grid_size();

    uint32_t PH = 8, PW = 8;
    uint32_t ph, pw;
    assert(num_experts <= 64);
    if (num_experts > 8) {
        ph = 1;
        pw = PW / (num_experts / PH);
    }
    else {
        ph = PH / num_experts;
        pw = PW;
    }
    uint32_t cores_per_expert = ph * pw;

    CoreRangeSet all_cores;
    all_cores = CoreRangeSet(CoreRange({0, 0}, {PH - 1, PW - 1}));

    // Circular buffer indices
    const uint32_t cb_input = CBIndex::c_0;            // Input tiles from input tensor
    const uint32_t cb_weights = CBIndex::c_1;          // Weight tiles from weights tensor
    const uint32_t cb_num_routed = CBIndex::c_2;       // num_routed_tokens array (num_experts * uint32_t)
    const uint32_t cb_metadata = CBIndex::c_3;         // Metadata from reader to compute/writer (16 * uint32_t)
    const uint32_t cb_out = CBIndex::c_16;             // Output tiles
    const uint32_t cb_out_buffer = CBIndex::c_24;      // Intermediate buffer for spilling (K-dimension accumulation)

    // Data formats
    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    // FIXME: How to set this properly? Maybe we need to distinguish prefill/decode
    uint32_t BMt = 1;
    uint32_t BNt = 1; // ???
    uint32_t BKt = 1;
    uint32_t SBMt = 1;
    uint32_t SBNt = 1;
    for (int i = 1; i <= std::min(4, (int)BNt); i++) {
        if (BNt % i == 0) {
            SBNt = i;
        }
    }
    uint32_t metadata_size = 16;
    uint32_t pipeline_factor = 2;

    TT_FATAL(Nt % BNt == 0, "Nt ({}) must be divisible by BNt ({})", Nt, BNt);

    // std::cout << "Nt: " << Nt << " cores_per_expert: " << cores_per_expert << std::endl;
    // std::cout << "BMt: " << BMt << " BNt: " << BNt << " BKt: " << BKt << " SBMt: " << SBMt << " SBNt: " << SBNt << std::endl;

    // Tile sizes
    uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format);
    uint32_t weights_tile_size = tt_metal::detail::TileSize(weights_data_format);
    uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format);

    // Create circular buffers on all cores (2 tiles for double buffering)
    CircularBufferConfig cb_input_config =
        CircularBufferConfig(pipeline_factor * BMt * BKt * input_tile_size, {{cb_input, input_data_format}})
            .set_page_size(cb_input, input_tile_size);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    CircularBufferConfig cb_weights_config =
        CircularBufferConfig(pipeline_factor * BKt * BNt * weights_tile_size, {{cb_weights, weights_data_format}})
            .set_page_size(cb_weights, weights_tile_size);
    CreateCircularBuffer(program, all_cores, cb_weights_config);

    // CB for shared num_routed_tokens data (num_experts * uint32_t)
    CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(num_experts * sizeof(uint32_t), {{cb_num_routed, tt::DataFormat::UInt32}})
            .set_page_size(cb_num_routed, num_experts * sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_num_routed_config);

    // CB for metadata communication from reader to compute/writer (16 * uint32_t)
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(metadata_size * sizeof(uint32_t), {{cb_metadata, tt::DataFormat::UInt32}})
            .set_page_size(cb_metadata, 16 * sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_metadata_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(pipeline_factor * BMt * BNt * output_tile_size, {{cb_out, output_data_format}})
            .set_page_size(cb_out, output_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    // CB for intermediate accumulation buffer (spilling for K-dimension reduction)
    // Use Float16_b for intermediate accumulation to maintain precision
    uint32_t intermediate_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    CircularBufferConfig cb_out_buffer_config =
        CircularBufferConfig(pipeline_factor * BMt * BNt * intermediate_tile_size, {{cb_out_buffer, tt::DataFormat::Float16_b}})
            .set_page_size(cb_out_buffer, intermediate_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_buffer_config);

    // Create unified semaphores for multicast (shared by all cores)
    auto sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Compile-time arguments for reader kernel
    std::vector<uint32_t> reader_compile_time_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*weights.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(reader_compile_time_args);

    // Compile-time arguments for writer kernel
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
    };

    // Create reader kernel (RISCV_0)
    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/reader_moe_bmm_multi_core_optimized.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // Create writer kernel (RISCV_1)
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/writer_moe_bmm_multi_core_optimized.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Create compute kernel
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/compute/moe_bmm_multi_core_optimized.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});


    uint32_t col_nblocks = Nt / BNt / PW;

    for (uint32_t x = 0; x < PH; x++) {
        for (uint32_t y = 0; y < PW; y++) {
            CoreCoord core_coord = {x, y};
            // Reader kernel arguments
            std::vector<uint32_t> reader_runtime_args = {
                input.buffer()->address(),
                weights.buffer()->address(),
                num_routed_tokens.buffer()->address(),
                num_experts,
                ph,
                pw,
                Mt_max,
                Nt,
                Kt,
                BMt,
                BNt,
                BKt,
                (uint32_t) sender_semaphore,
                (uint32_t) receiver_semaphore,
            };

            // Writer kernel arguments
            std::vector<uint32_t> writer_runtime_args = {
                output.buffer()->address(),
                num_experts,
                ph,
                pw,
                Mt_max,
                Nt,
                Kt,
                BMt,
                BNt,
                BKt,
                SBMt,
                SBNt,
            };

            // Compute kernel arguments 
            std::vector<uint32_t> compute_runtime_args = {
                Mt_max, Nt, Kt, BMt, BNt, BKt, SBMt, SBNt,
            };

            SetRuntimeArgs(program, reader_id, core_coord, reader_runtime_args);
            SetRuntimeArgs(program, writer_id, core_coord, writer_runtime_args);
            SetRuntimeArgs(program, compute_id, core_coord, compute_runtime_args);
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
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe
