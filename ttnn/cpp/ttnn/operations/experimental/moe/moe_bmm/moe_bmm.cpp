// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/moe_bmm_op.hpp"
#include "ttnn/operations/experimental/moe/moe_bmm/moe_bmm.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor MoEBMMOperation::invoke(
    QueueId queue_id,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    const Tensor& num_tiled_tokens,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(input.memory_config());

    return tt::tt_metal::operation::run(
        moe::MoEBMM{
            .output_mem_config = output_mem_config
        },
        {input, weights, num_routed_tokens, num_tiled_tokens},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental
