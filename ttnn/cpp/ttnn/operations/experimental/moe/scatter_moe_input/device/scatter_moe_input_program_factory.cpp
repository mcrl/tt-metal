#include "scatter_moe_input_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::experimental::moe::detail {

using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::operation::ProgramWithCallbacks scatter_moe_input_multi_core(
    const Tensor& input_hidden_state,
    const Tensor& num_routed_tokens,
    const Tensor& routed_tokens,
    Tensor& output) {

    Program program{};

    const auto& input_shape = input_hidden_state.padded_shape();
    uint32_t num_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    uint32_t num_local_experts = num_routed_shape[-1];  // 1D tensor (E/D)

    auto input_buffer = input_hidden_state.buffer();
    auto num_routed_buffer = num_routed_tokens.buffer();
    auto routed_tokens_buffer = routed_tokens.buffer();
    auto output_buffer = output.buffer();

    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool num_routed_is_dram = num_routed_buffer->buffer_type() == BufferType::DRAM;
    bool routed_tokens_is_dram = routed_tokens_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    DataFormat input_cb_data_format = datatype_to_dataformat_converter(input_hidden_state.dtype());
    uint32_t input_element_size = input_hidden_state.element_size();

    IDevice* device = input_hidden_state.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores = std::min(num_local_experts, static_cast<uint32_t>(compute_with_storage_grid_size.x * compute_with_storage_grid_size.y));
    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    uint32_t row_size_bytes = hidden_dim * input_element_size;
    uint32_t aligned_row_size_bytes = round_up_to_mul32(row_size_bytes);

    // Configuration
    constexpr uint32_t BMt = 16;  // Batch size (tokens per batch)
    constexpr uint32_t pipeline_factor = 2;  // Double buffering

    // CB 1: Routed tokens (reader only)
    uint32_t cb_id_routed = tt::CBIndex::c_1;
    uint32_t routed_size = num_tokens * sizeof(uint32_t);
    uint32_t aligned_routed_size = round_up_to_mul32(routed_size);
    DataFormat routed_data_format = datatype_to_dataformat_converter(DataType::UINT32);
    CircularBufferConfig cb_routed_config =
        CircularBufferConfig(aligned_routed_size, {{cb_id_routed, routed_data_format}})
            .set_page_size(cb_id_routed, aligned_routed_size);
    CreateCircularBuffer(program, all_cores, cb_routed_config);

    // CB 2: Buffer for num_routed_tokens (shared read-only)
    uint32_t cb_id_num_routed = tt::CBIndex::c_2;
    uint32_t num_routed_bytes = num_local_experts * sizeof(uint32_t);
    uint32_t aligned_num_routed_bytes = round_up_to_mul32(num_routed_bytes);
    DataFormat num_routed_data_format = datatype_to_dataformat_converter(DataType::UINT32);
    CircularBufferConfig cb_num_routed_config =
        CircularBufferConfig(aligned_num_routed_bytes, {{cb_id_num_routed, num_routed_data_format}})
            .set_page_size(cb_id_num_routed, aligned_num_routed_bytes);
    CreateCircularBuffer(program, all_cores, cb_num_routed_config);

    // CB 3: Metadata (reader → writer)
    uint32_t cb_id_metadata = tt::CBIndex::c_3;
    uint32_t metadata_size = 16 * sizeof(uint32_t);  // 64 bytes
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(metadata_size, {{cb_id_metadata, num_routed_data_format}})
            .set_page_size(cb_id_metadata, metadata_size);
    CreateCircularBuffer(program, all_cores, cb_metadata_config);

    // CB 4: Gathered tokens (reader → writer)
    uint32_t cb_id_gathered = tt::CBIndex::c_4;
    uint32_t gathered_cb_size = pipeline_factor * BMt * aligned_row_size_bytes;
    CircularBufferConfig cb_gathered_config =
        CircularBufferConfig(gathered_cb_size, {{cb_id_gathered, input_cb_data_format}})
            .set_page_size(cb_id_gathered, aligned_row_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_gathered_config);

    // Reader kernel (RISCV_0)
    std::vector<uint32_t> reader_compile_time_args = {
        cb_id_routed,
        cb_id_num_routed,
        cb_id_metadata,
        cb_id_gathered,
        (uint32_t)input_is_dram,
        (uint32_t)num_routed_is_dram,
        (uint32_t)routed_tokens_is_dram,
        hidden_dim,
        num_tokens,
        num_local_experts,
        row_size_bytes,
        BMt,
    };

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/scatter_moe_input/device/kernels/dataflow/"
        "reader_scatter_moe_input.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args
        });

    // Writer kernel (RISCV_1)
    std::vector<uint32_t> writer_compile_time_args = {
        cb_id_num_routed,
        cb_id_metadata,
        cb_id_gathered,
        (uint32_t)output_is_dram,
        num_tokens,
        num_local_experts,
        row_size_bytes,
        BMt,
    };

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/scatter_moe_input/device/kernels/dataflow/"
        "writer_scatter_moe_input.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args
        });

    auto cores = corerange_to_cores(all_cores, std::nullopt, true);
    uint32_t experts_per_core = (num_local_experts + num_cores - 1) / num_cores;

    for (uint32_t core_idx = 0; core_idx < cores.size(); core_idx++) {
        const CoreCoord& core = cores[core_idx];
        uint32_t start_expert_idx = core_idx * experts_per_core;
        uint32_t end_expert_idx = std::min(start_expert_idx + experts_per_core, num_local_experts);
        uint32_t num_experts_for_this_core = end_expert_idx - start_expert_idx;

        std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(),
            num_routed_buffer->address(),
            routed_tokens_buffer->address(),
            start_expert_idx,
            num_experts_for_this_core,
        };

        std::vector<uint32_t> writer_runtime_args = {
            output_buffer->address(),
            start_expert_idx,
            num_experts_for_this_core,
        };

        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    auto override_runtime_args_callback =
        [reader_kernel_id, writer_kernel_id, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_buffer = input_tensors.at(0).buffer();
            auto num_routed_buffer = input_tensors.at(1).buffer();
            auto routed_tokens_buffer = input_tensors.at(2).buffer();
            auto output_buffer = output_tensors.at(0).buffer();

            for (const auto& core : cores) {
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                reader_runtime_args[0] = input_buffer->address();
                reader_runtime_args[1] = num_routed_buffer->address();
                reader_runtime_args[2] = routed_tokens_buffer->address();

                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args[0] = output_buffer->address();
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

}
