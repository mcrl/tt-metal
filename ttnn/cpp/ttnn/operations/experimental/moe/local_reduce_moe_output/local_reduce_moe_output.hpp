// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// local_reduce_moe_output Operation
//
// PURPOSE:
//   Performs intra-device reduction by gathering expert outputs back to token order
//   and applying routing weights. Part of MoE V2 pipeline.
//
// IMPLEMENTATION (2025-10-16):
//   - Multi-core parallelization with token-parallel distribution
//   - Always uses multi-core for optimal performance
//   - Each core processes independent token range for linear scaling
//
// INPUTS:
//   - input_hidden_state: (E/D, T, H) bfloat16 tensor, ROW_MAJOR layout - expert outputs (organized by expert)
//   - token_idx_map: (E/D, T) uint32 tensor, ROW_MAJOR layout - mapping from expert-local position to global token index
//   - routed_token_weights: (E/D, T) bfloat16 tensor, ROW_MAJOR layout - routing weights for each expert-token assignment
//   - num_routed_tokens: (E/D, 1) uint32 tensor, ROW_MAJOR layout - count of tokens per local expert
//   - num_tokens: scalar uint32 - total number of tokens (T)
//
// OUTPUTS:
//   - output_hidden_state: (T, H) bfloat16 tensor, ROW_MAJOR layout - final output for all tokens on this device
//
// COMPUTATION:
//   For each global token index t in [0, T):
//     1. Initialize: output[t, :] = 0
//     2. For each local expert e in [0, E/D-1):
//        - Read t_e = num_routed_tokens[e, 0]
//        - For each expert-local position i in [0, t_e):
//          - If token_idx_map[e, i] == t:
//            - Read hidden state: hidden = input_hidden_state[e, i, :]
//            - Read routing weight: weight = routed_token_weights[e, i]
//            - Accumulate: output[t, :] += hidden * weight
//
// NOTES:
//   - Part of MoE V2 pipeline (deferred accumulation)
//   - Gathers expert outputs back to token order
//   - Applies routing weights during accumulation
//   - Performs intra-device reduction (still needs inter-device allreduce)
//   - Token-stationary parallelization (distribute tokens across cores)

namespace ttnn {
namespace operations::experimental {

struct LocalReduceMoeOutputOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_hidden_state,
        const Tensor& token_idx_map,
        const Tensor& routed_token_weights,
        const Tensor& num_routed_tokens,
        uint32_t num_tokens,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto local_reduce_moe_output = ttnn::register_operation<
    "ttnn::local_reduce_moe_output",
    ttnn::operations::experimental::LocalReduceMoeOutputOperation>();

}  // namespace ttnn
