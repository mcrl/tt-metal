// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// extract_attention_input Operation (Unified)
//
// PURPOSE:
//   Extracts batch chunks per device for attention input.
//   Automatically detects prefill vs decode mode based on input tensor rank.
//   Replaces separate extract_attention_input_prefill and extract_attention_input_decode operations.
//
// INPUTS:
//   - hidden_state: [B, S, H] (prefill) OR [1, 1, B, H] (decode), bfloat16, TILE_LAYOUT - replicated
//   - dp_degree: [1] integer tensor - data parallelism degree for this device
//                (per-device scalar value indicating device's row index in mesh)
//   - mesh_device: MeshDevice - device mesh to extract DP from
//
// OUTPUTS:
//   - output: [B//dp, 1, S, H] (prefill) OR [1, 1, B//dp, H] (decode) - extracted batch chunk
//
// MODE DETECTION:
//   - rank == 3: Prefill mode ([B, S, H] → [B//dp, 1, S, H])
//   - rank == 4: Decode mode ([1, 1, B, H] → [1, 1, B//dp, H])
//
// COMPUTATION:
//   For device with index dev_idx in DP dimension:
//     1. Calculate dp = mesh_device.shape[0]
//     2. Calculate batch_per_device = B // dp
//     3. Extract batch slice based on mode
//
// NOTES:
//   - Input is replicated across all devices
//   - Output is sharded along DP dimension
//   - Supports output dtype conversion (bfloat16 or bfloat8_b)

namespace ttnn {
namespace operations::experimental {

struct ExtractAttentionInputOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& hidden_state,
        const Tensor& dp_degree,
        const MeshDevice& mesh_device,
        const std::optional<DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto extract_attention_input = ttnn::register_operation<
    "ttnn::extract_attention_input",
    ttnn::operations::experimental::ExtractAttentionInputOperation>();

}  // namespace ttnn
