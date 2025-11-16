#include "moe_bmm_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace ttnn;
using namespace tt::tt_metal;

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

    uint32_t Kt = h_in / TILE_SIZE;      // Number of tiles in K dimension
    uint32_t Nt = h_out / TILE_SIZE;     // Number of tiles in N dimension (output columns)
    uint32_t Mt_max = max_tokens / TILE_SIZE;  // Maximum number of tiles in M dimension (tokens)

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

    const uint32_t cb_input = CBIndex::c_0;            // Input tiles from input tensor
    const uint32_t cb_weights = CBIndex::c_1;          // Weight tiles from weights tensor
    const uint32_t cb_num_routed = CBIndex::c_2;       // num_routed_tokens array (num_experts * uint32_t)
    const uint32_t cb_metadata = CBIndex::c_3;         // Metadata from reader to compute/writer (16 * uint32_t)
    const uint32_t cb_out = CBIndex::c_16;             // Output tiles
    const uint32_t cb_out_buffer = CBIndex::c_24;      // Intermediate buffer for spilling (K-dimension accumulation)

    tt::DataFormat input_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    // FIXME: How to set this properly? Maybe we need to distinguish prefill/decode
    uint32_t BMt = 1;
    uint32_t BNt = Nt / cores_per_expert;
    uint32_t BKt = 4;
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

    uint32_t input_tile_size = tile_size(input_data_format);
    uint32_t weights_tile_size = tile_size(weights_data_format);
    uint32_t output_tile_size = tile_size(output_data_format);

    CircularBufferConfig cb_input_config =
        CircularBufferConfig(pipeline_factor * BMt * BKt * input_tile_size, {{cb_input, input_data_format}})
            .set_page_size(cb_input, input_tile_size);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    CircularBufferConfig cb_weights_config =
        CircularBufferConfig(pipeline_factor * BKt * BNt * weights_tile_size, {{cb_weights, weights_data_format}})
            .set_page_size(cb_weights, weights_tile_size);
    CreateCircularBuffer(program, all_cores, cb_weights_config);

    CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(num_experts * sizeof(uint32_t), {{cb_num_routed, tt::DataFormat::UInt32}})
            .set_page_size(cb_num_routed, num_experts * sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_num_routed_config);

    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(metadata_size * sizeof(uint32_t), {{cb_metadata, tt::DataFormat::UInt32}})
            .set_page_size(cb_metadata, 16 * sizeof(uint32_t));
    CreateCircularBuffer(program, all_cores, cb_metadata_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(pipeline_factor * BMt * BNt * output_tile_size, {{cb_out, output_data_format}})
            .set_page_size(cb_out, output_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_config);

    uint32_t intermediate_tile_size = tile_size(tt::DataFormat::Float16_b);
    CircularBufferConfig cb_out_buffer_config =
        CircularBufferConfig(pipeline_factor * BMt * BNt * intermediate_tile_size, {{cb_out_buffer, tt::DataFormat::Float16_b}})
            .set_page_size(cb_out_buffer, intermediate_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_buffer_config);

    auto sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    auto input_master_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto input_slave_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    auto weight_master_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto weight_slave_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    // Unified compile-time args for both AreadCwrite and Bread kernels
    std::vector<uint32_t> unified_compile_time_args = {};
    TensorAccessorArgs(*input.buffer()).append_to(unified_compile_time_args);
    TensorAccessorArgs(*weights.buffer()).append_to(unified_compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(unified_compile_time_args);
    TensorAccessorArgs(*output.buffer()).append_to(unified_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
    };

    auto AreadCwrite_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/AreadCwrite.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = unified_compile_time_args});

    auto Bread_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/dataflow/Bread.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = unified_compile_time_args});

    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_bmm/device/kernels/compute/moe_bmm_multi_core_optimized.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    for (uint32_t x = 0; x < PH; x++) {
        for (uint32_t y = 0; y < PW; y++) {
            CoreCoord core_coord = {x, y};
            // Unified runtime args for both AreadCwrite and Bread kernels
            std::vector<uint32_t> unified_runtime_args = {
                input.buffer()->address(),           // 0
                weights.buffer()->address(),         // 1
                num_routed_tokens.buffer()->address(), // 2
                output.buffer()->address(),          // 3
                num_experts,                         // 4
                ph,                                  // 5
                pw,                                  // 6
                Mt_max,                              // 7
                Nt,                                  // 8
                Kt,                                  // 9
                BMt,                                 // 10
                BNt,                                 // 11
                BKt,                                 // 12
                SBMt,                                // 13
                SBNt,                                // 14
                (uint32_t) sender_semaphore,         // 15
                (uint32_t) receiver_semaphore,       // 16
                (uint32_t) input_master_semaphore,   // 17
                (uint32_t) input_slave_semaphore,    // 18
                (uint32_t) weight_master_semaphore,  // 19
                (uint32_t) weight_slave_semaphore,   // 20
            };

            std::vector<uint32_t> compute_runtime_args = {
                Mt_max, Nt, Kt, BMt, BNt, BKt, SBMt, SBNt,
            };

            SetRuntimeArgs(program, AreadCwrite_id, core_coord, unified_runtime_args);
            SetRuntimeArgs(program, Bread_id, core_coord, unified_runtime_args);
            SetRuntimeArgs(program, compute_id, core_coord, compute_runtime_args);
        }
    }

    auto override_runtime_arguments_callback = [AreadCwrite_id, Bread_id, compute_id, all_cores](
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
                auto& AreadCwrite_runtime_args = GetRuntimeArgs(program, AreadCwrite_id, core);
                AreadCwrite_runtime_args[0] = input_buffer->address();
                AreadCwrite_runtime_args[1] = weights_buffer->address();
                AreadCwrite_runtime_args[2] = num_routed_buffer->address();
                AreadCwrite_runtime_args[3] = output_buffer->address();

                auto& Bread_runtime_args = GetRuntimeArgs(program, Bread_id, core);
                Bread_runtime_args[0] = input_buffer->address();
                Bread_runtime_args[1] = weights_buffer->address();
                Bread_runtime_args[2] = num_routed_buffer->address();
                Bread_runtime_args[3] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}
