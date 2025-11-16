#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe::detail {

tt::tt_metal::operation::ProgramWithCallbacks local_reduce_moe_output(
    const Tensor& input_hidden_state,
    const Tensor& token_idx_map,
    const Tensor& routed_token_weights,
    const Tensor& num_routed_tokens,
    Tensor& output);

}
