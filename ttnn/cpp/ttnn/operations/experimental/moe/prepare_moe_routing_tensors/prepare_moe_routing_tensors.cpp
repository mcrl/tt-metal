// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/prepare_moe_routing_tensors_op.hpp"
#include "ttnn/operations/experimental/moe/prepare_moe_routing_tensors/prepare_moe_routing_tensors.hpp"

namespace ttnn::operations::experimental {

std::vector<ttnn::Tensor> PrepareMoeRoutingTensorsOperation::invoke(
    QueueId queue_id,
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    uint32_t num_experts,
    const std::optional<MemoryConfig>& memory_config) {

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    // Maximum tokens that can be routed to any single expert (worst case)
    // Each token selects top_k experts, but each token can only contribute once to each expert
    // So the maximum tokens any single expert can receive is T (all tokens choose that expert)
    const uint32_t max_tokens_per_expert = num_tokens;

    auto output_mem_config = memory_config.value_or(selected_experts.memory_config());

    return tt::tt_metal::operation::run(
        moe::PrepareMoeRoutingTensors{
            .num_experts = num_experts,
            .max_tokens_per_expert = max_tokens_per_expert,
            .output_mem_config = output_mem_config
        },
        {selected_experts, routing_weights},
        {},
        {},
        queue_id
    );
}

}  // namespace ttnn::operations::experimental