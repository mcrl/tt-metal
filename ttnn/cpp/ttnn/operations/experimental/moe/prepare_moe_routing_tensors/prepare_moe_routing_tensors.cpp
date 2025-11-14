#include "device/prepare_moe_routing_tensors_op.hpp"
#include "ttnn/operations/experimental/moe/prepare_moe_routing_tensors/prepare_moe_routing_tensors.hpp"

namespace ttnn::operations::experimental {

std::vector<ttnn::Tensor> PrepareMoeRoutingTensorsOperation::invoke(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    const Tensor& device_expert_mapping,
    uint32_t num_experts,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(selected_experts.memory_config());

    return tt::tt_metal::operation::run(
        moe::PrepareMoeRoutingTensors{
            .num_experts = num_experts,
            .output_mem_config = output_mem_config
        },
        {selected_experts, routing_weights, device_expert_mapping},
        {},
        {}
    );
}

}