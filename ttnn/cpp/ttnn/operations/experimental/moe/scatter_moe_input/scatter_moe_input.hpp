#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// scatter_moe_input Operation
//
//   Rearranges input tokens based on expert assignments.
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

}

constexpr auto scatter_moe_input = ttnn::register_operation<
    "ttnn::scatter_moe_input",
    ttnn::operations::experimental::ScatterMoeInputOperation>();

}
