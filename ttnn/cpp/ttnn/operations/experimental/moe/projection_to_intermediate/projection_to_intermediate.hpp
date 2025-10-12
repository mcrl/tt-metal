// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// projection_to_intermediate Operation
//
// PURPOSE:
//   Performs batched matrix multiplication for MoE projection to intermediate size.
//   Each expert processes only its assigned tokens and writes outputs sequentially.
//
// INPUTS:
//   - hidden_states: (T, H) bfloat16 tensor, ROW_MAJOR layout, replicated across devices
//   - routed_tokens: (E/D, max_tokens) uint32 tensor, ROW_MAJOR layout, sharded (device-local)
//   - num_routed_tokens: (E/D,) uint32 1D tensor, ROW_MAJOR layout, sharded (device-local)
//   - expert_weights: (E/D, H, H') bfloat16 tensor, ROW_MAJOR layout, sharded across devices
//
// OUTPUTS:
//   - output: (K*T, H') bfloat16 tensor - projection outputs (compacted, padded)
//
// COMPUTATION:
//   For each local expert (0 to E/D-1):
//     1. Get token count T_e from num_routed_tokens[local_expert_idx]
//     2. Gather T_e tokens from hidden_states using routed_tokens[local_expert_idx]
//     3. Perform matmul: (T_e × H) @ (H × H') = T_e × H'
//     4. Write output sequentially to pre-allocated tensor
//
// NOTES:
//   - Used for both gate_proj and up_proj in MoE layers
//   - Each device processes E/D experts in parallel (expert parallelism)
//   - Output is zero-padded to K*T size
//   - Routing tensors are device-local (E/D per device) from prepare_moe_routing_tensors

namespace ttnn {
namespace operations::experimental {

struct ProjectionToIntermediateOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& hidden_states,
        const Tensor& routed_tokens,
        const Tensor& num_routed_tokens,
        const Tensor& expert_weights,
        uint32_t top_k,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto projection_to_intermediate = ttnn::register_operation<
    "ttnn::projection_to_intermediate",
    ttnn::operations::experimental::ProjectionToIntermediateOperation>();

}  // namespace ttnn
