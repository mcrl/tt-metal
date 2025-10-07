// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// prepare_moe_routing_tensors Operation
//
// PURPOSE:
//   Converts sparse MoE expert selection into efficient routing tensors for expert-parallel computation.
//   Creates three tensors: num_routed_tokens, routed_tokens, and routed_token_weights.
//
// INPUTS:
//   - selected_experts: (T × K) uint32 tensor, ROW_MAJOR layout
//   - routing_weights: (T × K) bfloat16 tensor, ROW_MAJOR layout
//   - num_experts: scalar, total number of experts
//
// OUTPUTS:
//   - num_routed_tokens: (E) uint32 tensor - count of tokens routed to each expert
//   - routed_tokens: (E × max_tokens) uint32 tensor - token indices for each expert (padded)
//   - routed_token_weights: (E × max_tokens) bfloat16 tensor - routing weights for each expert (padded)
//
// NOTES:
//   - Each token selects top_k unique experts (no duplicates)
//   - Output tensors are padded to rectangular shape (E × max_tokens)
//   - max_tokens = T × K (worst case where all experts are on same device)

namespace ttnn {
namespace operations::experimental {

struct PrepareMoeRoutingTensorsOperation {
    static std::vector<ttnn::Tensor> invoke(
        QueueId queue_id,
        const Tensor& selected_experts,
        const Tensor& routing_weights,
        uint32_t num_experts,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto prepare_moe_routing_tensors = ttnn::register_operation<
    "ttnn::prepare_moe_routing_tensors",
    ttnn::operations::experimental::PrepareMoeRoutingTensorsOperation>();

}  // namespace ttnn