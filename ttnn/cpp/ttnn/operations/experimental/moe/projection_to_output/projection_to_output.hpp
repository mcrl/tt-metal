// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// projection_to_output Operation
//
// PURPOSE:
//   Performs the down projection step in MoE layers with routing weight application and accumulation.
//   Each expert processes its assigned tokens and accumulates weighted results to final output.
//
// INPUTS:
//   - combined_activations: (T_d × H') bfloat16 tensor, ROW_MAJOR layout
//     Contains combined gate*up activations for all token-expert pairs on device
//   - routed_tokens: (E/D × max_tokens) uint32 tensor, ROW_MAJOR layout, sharded (device-local)
//   - num_routed_tokens: (1 × E/D) uint32 tensor, ROW_MAJOR layout, sharded (device-local)
//   - routed_token_weights: (E/D × max_tokens) bfloat16 tensor, ROW_MAJOR layout, sharded (device-local)
//   - down_proj_weights: (E/D × H' × H) bfloat16 tensor, ROW_MAJOR layout, sharded across devices
//   - device_expert_mapping: (E/D) int32 tensor, ROW_MAJOR layout, sharded (for validation/future use)
//
// OUTPUTS:
//   - output: (T × H) bfloat16 tensor - final accumulated MoE output
//
// COMPUTATION:
//   For each local expert (0 to E/D-1):
//     1. Get token count T_e from num_routed_tokens[local_expert_idx]
//     2. Read T_e rows from combined_activations sequentially
//     3. Perform matmul: (T_e × H') @ (H' × H) = T_e × H
//     4. Multiply each result by corresponding routing weight
//     5. ACCUMULATE (not overwrite) to output tensor at token positions
//
// NOTES:
//   - This is Step 4 of the MoE computation pipeline
//   - Results are accumulated to handle multiple experts per token
//   - Input activations are already compacted (T_d size, not T*K)
//   - Output is initialized to zeros before accumulation
//   - Routing tensors are device-local (E/D per device) from prepare_moe_routing_tensors
//   - device_expert_mapping kept for API compatibility but routing is already device-local

namespace ttnn {
namespace operations::experimental {

struct ProjectionToOutputOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& combined_activations,
        const Tensor& routed_tokens,
        const Tensor& num_routed_tokens,
        const Tensor& routed_token_weights,
        const Tensor& down_proj_weights,
        const Tensor& device_expert_mapping,
        uint32_t num_tokens,
        uint32_t top_k,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto projection_to_output = ttnn::register_operation<
    "ttnn::projection_to_output",
    ttnn::operations::experimental::ProjectionToOutputOperation>();

}  // namespace ttnn