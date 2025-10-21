// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct LocalReduceMoeOutputOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_hidden_state,
        const Tensor& token_idx_map,
        const Tensor& routed_token_weights,
        const Tensor& num_routed_tokens,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto local_reduce_moe_output = ttnn::register_operation<
    "ttnn::local_reduce_moe_output",
    ttnn::operations::experimental::LocalReduceMoeOutputOperation>();

}  // namespace ttnn
