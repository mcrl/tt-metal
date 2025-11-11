#include "device/local_reduce_moe_output_op.hpp"
#include "ttnn/operations/experimental/moe/local_reduce_moe_output/local_reduce_moe_output.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor LocalReduceMoeOutputOperation::invoke(
    QueueId queue_id,
    const Tensor& input_hidden_state,
    const Tensor& token_idx_map,
    const Tensor& routed_token_weights,
    const Tensor& num_routed_tokens,
    const std::optional<MemoryConfig>& memory_config) {

    auto output_mem_config = memory_config.value_or(input_hidden_state.memory_config());

    return tt::tt_metal::operation::run(
        moe::LocalReduceMoeOutput{
            .output_mem_config = output_mem_config
        },
        {input_hidden_state, token_idx_map, routed_token_weights, num_routed_tokens},
        {},
        {},
        queue_id
    ).at(0);
}

}
