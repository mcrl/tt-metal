#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// prepare_moe_routing_tensors Operation
//
//   Converts sparse MoE expert selection into device-local routing tensors for expert-parallel computation.
//   Filters global routing information to only include experts assigned to this device.
//   Creates four device-local tensors: num_routed_tokens, routed_tokens, routed_token_weights, and token_idx_map.
//
// INPUTS:
//   - selected_experts: (T, K) uint32 tensor, ROW_MAJOR layout - global expert IDs selected by each token
//   - routing_weights: (T, K) bfloat16 tensor, ROW_MAJOR layout - routing weights for selected experts
//   - device_expert_mapping: (E/D,) int32 1D tensor, ROW_MAJOR layout - global expert IDs assigned to this device
//   - num_experts: scalar, total number of experts (E)
//
// OUTPUTS (device-local):
//   - num_routed_tokens: (E/D) uint32 1D tensor - count of tokens routed to each local expert
//   - routed_tokens: (E/D, max_tokens) uint32 2D tensor - token indices for each local expert (padded)
//   - routed_token_weights: (E/D, max_tokens) bfloat16 2D tensor - routing weights for each local expert (padded)
//   - token_idx_map: (E/D, max_tokens) uint32 2D tensor - mapping from expert-local token index to global token index
//

namespace ttnn {
namespace operations::experimental {

struct PrepareMoeRoutingTensorsOperation {
    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id,
        const Tensor& selected_experts,
        const Tensor& routing_weights,
        const Tensor& device_expert_mapping,
        uint32_t num_experts,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}

constexpr auto prepare_moe_routing_tensors = ttnn::register_operation<
    "ttnn::prepare_moe_routing_tensors",
    ttnn::operations::experimental::PrepareMoeRoutingTensorsOperation>();

}