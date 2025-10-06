// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/prepare_moe_mapping_tensor_op.hpp"
#include "ttnn/operations/experimental/moe/prepare_moe_mapping_tensor/prepare_moe_mapping_tensor.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor PrepareMoeMappingTensorOperation::invoke(
    QueueId queue_id,
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    uint32_t num_experts,
    const std::optional<MemoryConfig>& memory_config) {
    return tt::tt_metal::operation::run(
               moe::PrepareMoeMappingTensor{
                   .num_experts = num_experts,
                   .memory_config = memory_config.value_or(routing_weights.memory_config())},
               {selected_experts, routing_weights},
               {},
               {},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::experimental
