#include "device/extract_attention_input_op.hpp"
#include "ttnn/operations/experimental/attention/extract_attention_input/extract_attention_input.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor ExtractAttentionInputOperation::invoke(
    const Tensor& hidden_state,
    const Tensor& dp_degree,
    const MeshDevice& mesh_device,
    const std::optional<DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config) {

    uint32_t dp = mesh_device.shape()[0];
    auto out_dtype = output_dtype.value_or(hidden_state.dtype());

    TT_FATAL(
        out_dtype == DataType::BFLOAT16 || out_dtype == DataType::BFLOAT8_B,
        "output_dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        out_dtype);

    auto output_mem_config = memory_config.value_or(hidden_state.memory_config());

    return tt::tt_metal::operation::run(
        attention::ExtractAttentionInput{
            .output_mem_config = output_mem_config,
            .output_dtype = out_dtype,
            .dp = dp
        },
        {hidden_state, dp_degree},
        {},
        {}
    ).at(0);
}

}
