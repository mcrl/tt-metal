#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

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

}

constexpr auto extract_attention_input = ttnn::register_operation<
    "ttnn::extract_attention_input",
    ttnn::operations::experimental::ExtractAttentionInputOperation>();

}
