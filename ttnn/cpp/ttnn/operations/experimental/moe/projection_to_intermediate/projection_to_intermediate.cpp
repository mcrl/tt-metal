// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/projection_to_intermediate_op.hpp"
#include "ttnn/operations/experimental/moe/projection_to_intermediate/projection_to_intermediate.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ProjectionToIntermediateOperation::invoke(
    QueueId queue_id,
    const Tensor& hidden_states,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& expert_weights,
    uint32_t top_k,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(hidden_states.memory_config());

    return tt::tt_metal::operation::run(
        moe::ProjectionToIntermediate{
            .top_k = top_k,
            .output_mem_config = output_mem_config
        },
        {hidden_states, routed_tokens, num_routed_tokens, expert_weights},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental
