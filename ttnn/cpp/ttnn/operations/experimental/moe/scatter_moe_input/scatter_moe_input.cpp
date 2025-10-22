// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/scatter_moe_input_op.hpp"
#include "ttnn/operations/experimental/moe/scatter_moe_input/scatter_moe_input.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ScatterMoeInputOperation::invoke(
    QueueId queue_id,
    const Tensor& input_hidden_state,
    const Tensor& num_routed_tokens,
    const Tensor& routed_tokens,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(input_hidden_state.memory_config());

    return tt::tt_metal::operation::run(
        moe::ScatterMoeInput{
            .output_mem_config = output_mem_config
        },
        {input_hidden_state, num_routed_tokens, routed_tokens},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental
