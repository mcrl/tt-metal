// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/moe_expert_projection_op.hpp"
#include "ttnn/operations/experimental/moe/moe_expert_projection/moe_expert_projection.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor MoeExpertProjectionOperation::invoke(
    QueueId queue_id,
    const Tensor& hidden_states,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& expert_weights,
    const Tensor& device_expert_mapping,
    uint32_t top_k,
    const std::optional<MemoryConfig>& memory_config) {

    // Extract dimensions
    const auto& hidden_shape = hidden_states.padded_shape();
    const uint32_t num_tokens = hidden_shape[0];
    const uint32_t hidden_dim = hidden_shape[1];

    const auto& weights_shape = expert_weights.padded_shape();
    const uint32_t experts_per_device = weights_shape[0];
    const uint32_t expert_dim = weights_shape[2];

    const auto& routed_shape = routed_tokens.padded_shape();
    const uint32_t max_tokens_per_expert = routed_shape[1];

    // Calculate output size: K * T (top_k * num_tokens)
    const uint32_t output_size = top_k * num_tokens;

    auto output_mem_config = memory_config.value_or(hidden_states.memory_config());

    return tt::tt_metal::operation::run(
        moe::MoeExpertProjection{
            .num_tokens = num_tokens,
            .hidden_dim = hidden_dim,
            .expert_dim = expert_dim,
            .experts_per_device = experts_per_device,
            .max_tokens_per_expert = max_tokens_per_expert,
            .output_size = output_size,
            .output_mem_config = output_mem_config
        },
        {hidden_states, routed_tokens, num_routed_tokens, expert_weights, device_expert_mapping},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental
