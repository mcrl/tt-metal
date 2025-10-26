// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

// extract_attention_input_prefill Operation
//
// PURPOSE:
//   Extracts batch chunks per device for attention input in prefill mode.
//   Replaces matrix multiplication-based approach with dedicated operation.
//
// INPUTS:
//   - hidden_state: [B, S, H] bfloat16 tensor, TILE_LAYOUT - input token embeddings (replicated)
//   - mesh_device: MeshDevice - device mesh to extract DP from
//
// OUTPUTS:
//   - output: [B//dp, 1, S, H] tensor - extracted batch chunk for this device
//
// COMPUTATION:
//   For device with index dev_idx in DP dimension:
//     1. Calculate dp = mesh_device.shape[0]
//     2. Calculate batch_per_device = B // dp
//     3. Extract batch slice: output = hidden_state[dev_idx * batch_per_device : (dev_idx+1) * batch_per_device, :, :]
//     4. Reshape to [B//dp, 1, S, H]
//
// NOTES:
//   - Input is replicated across all devices
//   - Output is sharded along DP dimension
//   - Supports output dtype conversion (bfloat16 or bfloat8_b)

namespace ttnn {
namespace operations::experimental {

struct ExtractAttentionInputPrefillOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& hidden_state,
        const MeshDevice& mesh_device,
        const std::optional<DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto extract_attention_input_prefill = ttnn::register_operation<
    "ttnn::extract_attention_input_prefill",
    ttnn::operations::experimental::ExtractAttentionInputPrefillOperation>();

}  // namespace ttnn
