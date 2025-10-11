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
    const Tensor& device_expert_mapping,
    uint32_t num_experts,
    const std::optional<MemoryConfig>& memory_config) {

    const auto& experts_shape = selected_experts.padded_shape();
    const uint32_t num_tokens = experts_shape[0];
    const uint32_t top_k = experts_shape[1];

    // Extract num_local_experts from device_expert_mapping shape
    const auto& mapping_shape = device_expert_mapping.padded_shape();
    uint32_t num_local_experts;
    if (mapping_shape.rank() == 1) {
        num_local_experts = mapping_shape[0];
    } else if (mapping_shape.rank() == 2) {
        // Shape is (1, E/D) or (E/D, 1)
        num_local_experts = (mapping_shape[0] == 1) ? mapping_shape[1] : mapping_shape[0];
    } else {
        TT_FATAL(false, "device_expert_mapping must be 1D or 2D tensor");
    }

    // Maximum tokens that can be routed to any single expert (worst case)
    // Each token selects top_k experts, but each token can only contribute once to each expert
    // So the maximum tokens any single expert can receive is T (all tokens choose that expert)
    const uint32_t max_tokens_per_expert = num_tokens;

    auto output_mem_config = memory_config.value_or(selected_experts.memory_config());

    return tt::tt_metal::operation::run(
        moe::PrepareMoeRoutingTensors{
            .num_experts = num_experts,
            .num_local_experts = num_local_experts,
            .max_tokens_per_expert = max_tokens_per_expert,
            .output_mem_config = output_mem_config
        },
        {selected_experts, routing_weights, device_expert_mapping},
        {},
        {},
        queue_id
    );
}

}  // namespace ttnn::operations::experimental