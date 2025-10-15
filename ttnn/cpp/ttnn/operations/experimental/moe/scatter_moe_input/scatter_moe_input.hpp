// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// scatter_moe_input Operation
//
// PURPOSE:
//   Rearranges input tokens based on expert assignments for MoE V2 pipeline.
//   Gathers all tokens assigned to each local expert into contiguous memory.
//
// INPUTS:
//   - input_hidden_state: (T, H) bfloat16 tensor, ROW_MAJOR layout - input token embeddings (replicated)
//   - num_routed_tokens: (E/D, 1) uint32 tensor, ROW_MAJOR layout - count of tokens per local expert (sharded)
//   - routed_tokens: (E/D, T) uint32 tensor, ROW_MAJOR layout - token indices per local expert (sharded)
//
// OUTPUTS:
//   - output_hidden_state: (E/D, T, H) bfloat16 tensor, ROW_MAJOR layout - scattered input organized by expert
//
// COMPUTATION:
//   For each local expert e in [0, E/D-1):
//     1. Read t_e = num_routed_tokens[e, 0] (number of tokens for this expert)
//     2. For each position i in [0, t_e):
//        - Read global token index: t_{e,i} = routed_tokens[e, i]
//        - Gather from input: output[e, i, :] = input_hidden_state[t_{e,i}, :]
//     3. For remaining positions i in [t_e, T):
//        - Zero-pad: output[e, i, :] = 0
//
// NOTES:
//   - Part of MoE V2 pipeline (separates scatter logic from projection)
//   - Enables efficient BMM operations on already-scattered data
//   - Output is zero-padded to uniform shape (E/D, T, H)
//   - Routing tensors are device-local from prepare_moe_routing_tensors

namespace ttnn {
namespace operations::experimental {

struct ScatterMoeInputOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_hidden_state,
        const Tensor& num_routed_tokens,
        const Tensor& routed_tokens,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto scatter_moe_input = ttnn::register_operation<
    "ttnn::scatter_moe_input",
    ttnn::operations::experimental::ScatterMoeInputOperation>();

}  // namespace ttnn
