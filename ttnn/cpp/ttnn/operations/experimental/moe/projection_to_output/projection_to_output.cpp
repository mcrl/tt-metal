// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/projection_to_output_op.hpp"
#include "ttnn/operations/experimental/moe/projection_to_output/projection_to_output.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ProjectionToOutputOperation::invoke(
    QueueId queue_id,
    const Tensor& combined_activations,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& routed_token_weights,
    const Tensor& down_proj_weights,
    uint32_t num_tokens,
    uint32_t top_k,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(combined_activations.memory_config());

    return tt::tt_metal::operation::run(
        moe::ProjectionToOutput{
            .num_tokens = num_tokens,
            .top_k = top_k,
            .output_mem_config = output_mem_config
        },
        {combined_activations, routed_tokens, num_routed_tokens, routed_token_weights,
         down_proj_weights},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental