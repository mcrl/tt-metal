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

operation::ProgramWithCallbacks prepare_moe_routing_tensors_multi_core(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    const Tensor& device_expert_mapping,
    Tensor& num_routed_tokens,
    Tensor& routed_tokens,
    Tensor& routed_token_weights,
    Tensor& tokenidx_expertlocal_to_global,
    uint32_t num_experts,
    uint32_t num_local_experts,
    uint32_t max_tokens_per_expert) {

    Program program{};

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    IDevice* device = selected_experts.device();

    // Calculate core grid for multi-core expert parallelism
    // Each core processes one local expert
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores = num_local_experts;
    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    // Circular buffer indices
    const uint32_t cb_experts = CBIndex::c_0;
    const uint32_t cb_weights = CBIndex::c_1;
    const uint32_t cb_device_mapping = CBIndex::c_2;
    const uint32_t cb_num_routed = CBIndex::c_16;
    const uint32_t cb_routed_tokens = CBIndex::c_17;
    const uint32_t cb_routed_weights = CBIndex::c_18;
    const uint32_t cb_tokenidx_map = CBIndex::c_19;
    const uint32_t cb_scratch = CBIndex::c_24;

    // Data formats
    tt::DataFormat experts_data_format = tt_metal::datatype_to_dataformat_converter(selected_experts.dtype());
    tt::DataFormat weights_data_format = tt_metal::datatype_to_dataformat_converter(routing_weights.dtype());
    tt::DataFormat mapping_data_format = tt_metal::datatype_to_dataformat_converter(device_expert_mapping.dtype());
    tt::DataFormat num_routed_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    tt::DataFormat routed_tokens_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    tt::DataFormat routed_weights_data_format = tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
    tt::DataFormat tokenidx_map_data_format = tt_metal::datatype_to_dataformat_converter(DataType::UINT32);

    // Buffer sizes (per-core, each core processes one expert)
    const uint32_t experts_row_bytes = top_k * sizeof(uint32_t);
    const uint32_t weights_row_bytes = top_k * sizeof(uint16_t);
    const uint32_t mapping_bytes = sizeof(int32_t);  // Single expert mapping per core
    const uint32_t num_routed_bytes = sizeof(uint32_t);  // Single count per core
    const uint32_t routed_tokens_row_bytes = max_tokens_per_expert * sizeof(uint32_t);
    const uint32_t routed_weights_row_bytes = max_tokens_per_expert * sizeof(uint16_t);
    const uint32_t tokenidx_map_row_bytes = max_tokens_per_expert * sizeof(uint32_t);

    // Scratch buffer for collecting tokens for single expert (per-core)
    // Size: max_tokens_per_expert * (sizeof(uint32_t) + sizeof(uint16_t))
    const uint32_t scratch_bytes = max_tokens_per_expert * (sizeof(uint32_t) + sizeof(uint16_t));

    // Create circular buffers (shared across all cores)
    tt_metal::CircularBufferConfig experts_cb_config =
        tt_metal::CircularBufferConfig(experts_row_bytes, {{cb_experts, experts_data_format}})
            .set_page_size(cb_experts, experts_row_bytes);
    CreateCircularBuffer(program, all_cores, experts_cb_config);

    tt_metal::CircularBufferConfig weights_cb_config =
        tt_metal::CircularBufferConfig(weights_row_bytes, {{cb_weights, weights_data_format}})
            .set_page_size(cb_weights, weights_row_bytes);
    CreateCircularBuffer(program, all_cores, weights_cb_config);

    tt_metal::CircularBufferConfig mapping_cb_config =
        tt_metal::CircularBufferConfig(mapping_bytes, {{cb_device_mapping, mapping_data_format}})
            .set_page_size(cb_device_mapping, mapping_bytes);
    CreateCircularBuffer(program, all_cores, mapping_cb_config);

    tt_metal::CircularBufferConfig num_routed_cb_config =
        tt_metal::CircularBufferConfig(num_routed_bytes, {{cb_num_routed, num_routed_data_format}})
            .set_page_size(cb_num_routed, num_routed_bytes);
    CreateCircularBuffer(program, all_cores, num_routed_cb_config);

    tt_metal::CircularBufferConfig routed_tokens_cb_config =
        tt_metal::CircularBufferConfig(routed_tokens_row_bytes, {{cb_routed_tokens, routed_tokens_data_format}})
            .set_page_size(cb_routed_tokens, routed_tokens_row_bytes);
    CreateCircularBuffer(program, all_cores, routed_tokens_cb_config);

    tt_metal::CircularBufferConfig routed_weights_cb_config =
        tt_metal::CircularBufferConfig(routed_weights_row_bytes, {{cb_routed_weights, routed_weights_data_format}})
            .set_page_size(cb_routed_weights, routed_weights_row_bytes);
    CreateCircularBuffer(program, all_cores, routed_weights_cb_config);

    tt_metal::CircularBufferConfig tokenidx_map_cb_config =
        tt_metal::CircularBufferConfig(tokenidx_map_row_bytes, {{cb_tokenidx_map, tokenidx_map_data_format}})
            .set_page_size(cb_tokenidx_map, tokenidx_map_row_bytes);
    CreateCircularBuffer(program, all_cores, tokenidx_map_cb_config);

    // Scratch buffer for intermediate data (per core)
    tt_metal::CircularBufferConfig scratch_cb_config =
        tt_metal::CircularBufferConfig(scratch_bytes, {{cb_scratch, num_routed_data_format}})
            .set_page_size(cb_scratch, scratch_bytes);
    CreateCircularBuffer(program, all_cores, scratch_cb_config);

    // Compile-time arguments for TensorAccessor (same for all cores)
    std::vector<uint32_t> compile_time_args = {};

    // Add TensorAccessor compile-time args for all buffers
    TensorAccessorArgs(*selected_experts.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routing_weights.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*device_expert_mapping.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*num_routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routed_tokens.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*routed_token_weights.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*tokenidx_expertlocal_to_global.buffer()).append_to(compile_time_args);

    // Create kernel (shared across all cores)
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/prepare_moe_routing_tensors/device/kernels/dataflow/reader_writer_moe_routing.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    // Convert CoreRangeSet to vector of CoreCoords
    auto cores = corerange_to_cores(all_cores, std::nullopt, true);

    // Set runtime args per core (each core gets its local_expert_id)
    for (uint32_t core_idx = 0; core_idx < cores.size(); core_idx++) {
        const CoreCoord& core = cores[core_idx];
        uint32_t local_expert_id = core_idx;

        std::vector<uint32_t> runtime_args = {
            selected_experts.buffer()->address(),
            routing_weights.buffer()->address(),
            device_expert_mapping.buffer()->address(),
            num_routed_tokens.buffer()->address(),
            routed_tokens.buffer()->address(),
            routed_token_weights.buffer()->address(),
            tokenidx_expertlocal_to_global.buffer()->address(),
            num_tokens,
            top_k,
            num_experts,
            num_local_experts,
            max_tokens_per_expert,
            local_expert_id  // NEW: Each core's assigned expert
        };

        SetRuntimeArgs(program, kernel, core, runtime_args);
    }

    auto override_runtime_arguments_callback = [kernel, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& selected_experts = input_tensors[0];
        const auto& routing_weights = input_tensors[1];
        const auto& device_expert_mapping = input_tensors[2];
        const auto& num_routed_tokens = output_tensors[0];
        const auto& routed_tokens = output_tensors[1];
        const auto& routed_token_weights = output_tensors[2];
        const auto& tokenidx_expertlocal_to_global = output_tensors[3];

        // Update runtime args for all cores
        for (const auto& core : cores) {
            auto& runtime_args = GetRuntimeArgs(program, kernel, core);
            runtime_args[0] = selected_experts.buffer()->address();
            runtime_args[1] = routing_weights.buffer()->address();
            runtime_args[2] = device_expert_mapping.buffer()->address();
            runtime_args[3] = num_routed_tokens.buffer()->address();
            runtime_args[4] = routed_tokens.buffer()->address();
            runtime_args[5] = routed_token_weights.buffer()->address();
            runtime_args[6] = tokenidx_expertlocal_to_global.buffer()->address();
        }
    };

    return {std::move(program), override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::moe