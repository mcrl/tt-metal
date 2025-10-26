// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/extract_attention_input_prefill_op.hpp"
#include "ttnn/operations/experimental/attention/extract_attention_input_prefill/extract_attention_input_prefill.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ExtractAttentionInputPrefillOperation::invoke(
    QueueId queue_id,
    const Tensor& hidden_state,
    const MeshDevice& mesh_device,
    const std::optional<DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config) {

    // Extract DP from mesh device
    uint32_t dp = mesh_device.shape()[0];

    // Default output dtype to input dtype
    auto out_dtype = output_dtype.value_or(hidden_state.dtype());

    // Validate output dtype
    TT_FATAL(
        out_dtype == DataType::BFLOAT16 || out_dtype == DataType::BFLOAT8_B,
        "output_dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        out_dtype);

    auto output_mem_config = memory_config.value_or(hidden_state.memory_config());

    return tt::tt_metal::operation::run(
        attention::ExtractAttentionInputPrefill{
            .output_mem_config = output_mem_config,
            .output_dtype = out_dtype,
            .dp = dp
        },
        {hidden_state},
        {},
        {},
        queue_id
    ).at(0);
}

}  // namespace ttnn::operations::experimental
