// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe::detail {

tt::tt_metal::operation::ProgramWithCallbacks scatter_moe_input_single_core(
    const Tensor& input_hidden_state,
    const Tensor& num_routed_tokens,
    const Tensor& routed_tokens,
    Tensor& output);

}  // namespace ttnn::operations::experimental::moe::detail
