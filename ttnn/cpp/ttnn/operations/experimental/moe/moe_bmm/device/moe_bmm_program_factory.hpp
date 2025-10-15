// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt;
using namespace tt::tt_metal;

operation::ProgramWithCallbacks moe_bmm_single_core(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& num_routed_tokens,
    Tensor& output,
    uint32_t num_experts,
    uint32_t max_tokens,
    uint32_t h_in,
    uint32_t h_out);

}  // namespace ttnn::operations::experimental::moe
