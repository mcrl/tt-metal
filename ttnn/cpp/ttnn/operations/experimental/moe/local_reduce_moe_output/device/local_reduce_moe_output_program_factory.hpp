// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe::detail {

tt::tt_metal::operation::ProgramWithCallbacks local_reduce_moe_output_multi_core(
    const Tensor& input_hidden_state,
    const Tensor& token_idx_map,
    const Tensor& routed_token_weights,
    const Tensor& num_routed_tokens,
    uint32_t num_tokens,
    Tensor& output);

}  // namespace ttnn::operations::experimental::moe::detail
