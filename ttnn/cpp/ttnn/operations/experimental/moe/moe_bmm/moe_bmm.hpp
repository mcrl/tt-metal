// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// moe_bmm Operation
//
// PURPOSE:
//   Performs batched matrix multiplication for Mixture-of-Experts (MoE) processing.
//   Each expert processes only its assigned tokens (first num_routed_tokens[e, 0] rows).
//
// INPUTS:
//   - input: (E/D, T, H_in) bfloat16 tensor, TILE_LAYOUT, sharded across devices
//            E/D experts per device, T max tokens per expert, H_in input hidden size
//   - weights: (E/D, H_in, H_out) bfloat16 tensor, TILE_LAYOUT, sharded across devices
//              Expert weight matrices (one per local expert)
//   - num_routed_tokens: (E/D, 1) uint32 2D tensor, ROW_MAJOR layout, sharded (device-local)
//                        Access as num_routed_tokens[e, 0] for local expert e
//
// OUTPUTS:
//   - output: (E/D, T, H_out) bfloat16 tensor - batched matmul outputs
//
// COMPUTATION:
//   For each local expert e (0 to E/D-1):
//     1. Get active token count T_e from num_routed_tokens[e, 0]
//     2. Perform batched matmul: output[e, :T_e, :] = input[e, :T_e, :] @ weights[e, :, :]
//        This multiplies (T_e × H_in) @ (H_in × H_out) → (T_e × H_out)
//     3. Remaining rows (T_e to T-1) are zero (unused tokens)
//
// NOTES:
//   - Single-core implementation using TILE_LAYOUT for efficient computation
//   - Each device processes E/D experts in parallel (expert parallelism)
//   - Only first num_routed_tokens[e, 0] rows per expert produce non-zero results
//   - Output is zero-padded per expert (each expert has T rows with padding after active tokens)

namespace ttnn {
namespace operations::experimental {

struct MoEBMMOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        const Tensor& weights,
        const Tensor& num_routed_tokens,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto moe_bmm = ttnn::register_operation<
    "ttnn::experimental::moe_bmm",
    ttnn::operations::experimental::MoEBMMOperation>();
}  // namespace experimental

}  // namespace ttnn
