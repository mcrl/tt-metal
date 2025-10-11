// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/projection_to_output_op.hpp"
#include "ttnn/operations/experimental/moe/projection_to_output/projection_to_output.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ProjectionToOutputOperation::invoke(
    QueueId queue_id,
    const Tensor& combined_activations,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& routed_token_weights,
    const Tensor& down_proj_weights,
    const Tensor& device_expert_mapping,
    uint32_t num_tokens,
    uint32_t top_k,
    const std::optional<MemoryConfig>& memory_config) {

    // Extract dimensions from input tensors
    const auto& combined_shape = combined_activations.padded_shape();
    const auto& weights_shape = down_proj_weights.padded_shape();
    const auto& routed_shape = routed_tokens.padded_shape();

    uint32_t expert_dim = combined_shape[1];  // H'
    uint32_t hidden_dim = weights_shape[2];   // H
    uint32_t experts_per_device = weights_shape[0];  // E/D
    uint32_t max_tokens_per_expert = routed_shape[1];

    // Create the device operation
    auto operation = moe::ProjectionToOutput{
        .num_tokens = num_tokens,
        .hidden_dim = hidden_dim,
        .expert_dim = expert_dim,
        .experts_per_device = experts_per_device,
        .max_tokens_per_expert = max_tokens_per_expert,
        .top_k = top_k,
        .output_mem_config = memory_config.value_or(combined_activations.memory_config())
    };

    // Run the operation
    return tt::tt_metal::operation::run(
        operation,
        {combined_activations, routed_tokens, num_routed_tokens, routed_token_weights,
         down_proj_weights, device_expert_mapping},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental