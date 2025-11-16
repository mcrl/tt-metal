#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks moe_bmm_multi_core_optimized(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    Tensor& output,
    uint32_t num_experts,
    uint32_t max_tokens,
    uint32_t h_in,
    uint32_t h_out);

}