#include "device/moe_bmm_op.hpp"
#include "ttnn/operations/experimental/moe/moe_bmm/moe_bmm.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor MoEBMMOperation::invoke(
    QueueId queue_id,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    const std::optional<MemoryConfig>& memory_config,
    const std::string& mode) {

    auto output_mem_config = memory_config.value_or(input.memory_config());

    return tt::tt_metal::operation::run(
        moe::MoEBMM{
            .output_mem_config = output_mem_config,
            .mode = mode
        },
        {input, weights, num_routed_tokens},
        {},
        {},
        queue_id
    ).at(0);
}

}