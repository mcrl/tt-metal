#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe::detail {

tt::tt_metal::operation::ProgramWithCallbacks scatter_moe_input_multi_core(
    const Tensor& input_hidden_state,
    const Tensor& num_routed_tokens,
    const Tensor& routed_tokens,
    Tensor& output);

}