// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// prepare_moe_mapping_tensor Operation
//
// PURPOSE:
//   Converts sparse MoE expert selection to dense format for efficient computation.
//   Maps (num_tokens × top_k) sparse selection to (num_tokens × num_experts) dense tensor.
//
// INPUTS:
//   - selected_experts: (T × K) uint32 tensor, ROW_MAJOR layout
//   - routing_weights: (T × K) bfloat16 tensor, ROW_MAJOR layout
//   - num_experts: scalar, total number of experts
//
// OUTPUT:
//   - Dense (T × E) bfloat16 tensor where output[t,e] = weight if expert e selected, else 0
//
// CURRENT STATUS:
//   ✓ API fully implemented and registered
//   ✓ Build integration complete
//   ✓ Operation structure correct
//   ✗ Kernel has data movement issue - outputs all zeros
//
// SEE: ttnn/cpp/ttnn/operations/experimental/moe/README.md for detailed status
// SEE: models/demos/qwen3/tests/IMPLEMENTATION_STATUS.md for debugging guide

namespace ttnn {
namespace operations::experimental {

struct PrepareMoeMappingTensorOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& selected_experts,
        const Tensor& routing_weights,
        uint32_t num_experts,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto prepare_moe_mapping_tensor = ttnn::register_operation<
    "ttnn::prepare_moe_mapping_tensor",
    ttnn::operations::experimental::PrepareMoeMappingTensorOperation>();

}  // namespace ttnn
