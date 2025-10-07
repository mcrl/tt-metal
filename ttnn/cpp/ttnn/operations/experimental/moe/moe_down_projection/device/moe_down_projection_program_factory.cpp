// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#include "moe_down_projection_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::operation::ProgramWithCallbacks moe_down_projection_multi_core(
    const Tensor& combined_activations,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& routed_token_weights,
    const Tensor& down_proj_weights,
    const Tensor& device_expert_mapping,
    Tensor& output,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t expert_dim,
    uint32_t experts_per_device,
    uint32_t max_tokens_per_expert,
    uint32_t top_k) {

    Program program = CreateProgram();
    IDevice* device = combined_activations.device();
    auto* combined_buffer = combined_activations.buffer();
    auto* routed_buffer = routed_tokens.buffer();
    auto* num_routed_buffer = num_routed_tokens.buffer();
    auto* routed_weights_buffer = routed_token_weights.buffer();
    auto* weights_buffer = down_proj_weights.buffer();
    auto* mapping_buffer = device_expert_mapping.buffer();
    auto* output_buffer = output.buffer();

    // Get padded number of experts from num_routed_tokens tensor
    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    uint32_t num_experts_padded = num_routed_shape[1];

    // Use a single core for simplicity (can be optimized later for multi-core)
    CoreCoord core = {0, 0};
    CoreRange core_range({0, 0}, {0, 0});

    // Calculate buffer sizes
    uint32_t combined_row_size = expert_dim * sizeof(uint16_t);
    uint32_t routed_row_size = max_tokens_per_expert * sizeof(uint32_t);
    uint32_t num_routed_size = num_experts_padded * sizeof(uint32_t);
    uint32_t weights_routing_row_size = max_tokens_per_expert * sizeof(uint16_t);
    uint32_t weights_row_size = hidden_dim * sizeof(uint16_t);
    uint32_t mapping_size = experts_per_device * sizeof(int32_t);
    uint32_t output_row_size = hidden_dim * sizeof(uint16_t);

    // Determine mapping page size based on tensor rank
    const auto& mapping_shape = device_expert_mapping.padded_shape();
    uint32_t mapping_page_size = (mapping_shape.rank() == 3) ? mapping_shape[2] * sizeof(int32_t) : mapping_shape[1] * sizeof(int32_t);

    // Create circular buffers
    CircularBufferConfig cb_combined_config = CircularBufferConfig(
        combined_row_size, {{CBIndex::c_0, DataFormat::Float16_b}})
        .set_page_size(CBIndex::c_0, combined_row_size);
    CreateCircularBuffer(program, core_range, cb_combined_config);

    CircularBufferConfig cb_routed_config = CircularBufferConfig(
        routed_row_size, {{CBIndex::c_1, DataFormat::UInt32}})
        .set_page_size(CBIndex::c_1, routed_row_size);
    CreateCircularBuffer(program, core_range, cb_routed_config);

    CircularBufferConfig cb_num_routed_config = CircularBufferConfig(
        num_routed_size, {{CBIndex::c_2, DataFormat::UInt32}})
        .set_page_size(CBIndex::c_2, num_routed_size);
    CreateCircularBuffer(program, core_range, cb_num_routed_config);

    CircularBufferConfig cb_weights_routing_config = CircularBufferConfig(
        weights_routing_row_size, {{CBIndex::c_3, DataFormat::Float16_b}})
        .set_page_size(CBIndex::c_3, weights_routing_row_size);
    CreateCircularBuffer(program, core_range, cb_weights_routing_config);

    CircularBufferConfig cb_weights_config = CircularBufferConfig(
        weights_row_size, {{CBIndex::c_4, DataFormat::Float16_b}})
        .set_page_size(CBIndex::c_4, weights_row_size);
    CreateCircularBuffer(program, core_range, cb_weights_config);

    CircularBufferConfig cb_mapping_config = CircularBufferConfig(
        mapping_size, {{CBIndex::c_5, DataFormat::Int32}})
        .set_page_size(CBIndex::c_5, mapping_size);
    CreateCircularBuffer(program, core_range, cb_mapping_config);

    CircularBufferConfig cb_output_config = CircularBufferConfig(
        output_row_size, {{CBIndex::c_16, DataFormat::Float16_b}})
        .set_page_size(CBIndex::c_16, output_row_size);
    CreateCircularBuffer(program, core_range, cb_output_config);

    // Compile time args for kernel
    std::vector<uint32_t> compile_time_args;

    // Add tensor accessor args
    TensorAccessorArgs(combined_buffer).append_to(compile_time_args);
    TensorAccessorArgs(routed_buffer).append_to(compile_time_args);
    TensorAccessorArgs(num_routed_buffer).append_to(compile_time_args);
    TensorAccessorArgs(routed_weights_buffer).append_to(compile_time_args);
    TensorAccessorArgs(weights_buffer).append_to(compile_time_args);
    TensorAccessorArgs(mapping_buffer).append_to(compile_time_args);
    TensorAccessorArgs(output_buffer).append_to(compile_time_args);

    // Create dataflow kernel
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe/moe_down_projection/device/kernels/dataflow/reader_writer_down_projection.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = compile_time_args
        }
    );

    // Runtime args
    std::vector<uint32_t> runtime_args = {
        combined_buffer->address(),
        routed_buffer->address(),
        num_routed_buffer->address(),
        routed_weights_buffer->address(),
        weights_buffer->address(),
        mapping_buffer->address(),
        output_buffer->address(),
        num_tokens,
        hidden_dim,
        expert_dim,
        experts_per_device,
        max_tokens_per_expert,
        top_k,
        num_experts_padded
    };

    SetRuntimeArgs(program, kernel, core, runtime_args);

    auto override_runtime_args_callback = [kernel, core](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors) {

        auto combined_buffer = input_tensors[0].buffer();
        auto routed_buffer = input_tensors[1].buffer();
        auto num_routed_buffer = input_tensors[2].buffer();
        auto routed_weights_buffer = input_tensors[3].buffer();
        auto weights_buffer = input_tensors[4].buffer();
        auto mapping_buffer = input_tensors[5].buffer();
        auto output_buffer = output_tensors[0].buffer();

        auto& runtime_args = GetRuntimeArgs(program, kernel, core);
        runtime_args[0] = combined_buffer->address();
        runtime_args[1] = routed_buffer->address();
        runtime_args[2] = num_routed_buffer->address();
        runtime_args[3] = routed_weights_buffer->address();
        runtime_args[4] = weights_buffer->address();
        runtime_args[5] = mapping_buffer->address();
        runtime_args[6] = output_buffer->address();
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::moe