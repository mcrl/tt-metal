// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::attention::detail {

tt::tt_metal::operation::ProgramWithCallbacks extract_attention_input_single_core(
    const ttnn::Tensor& hidden_state,
    const ttnn::Tensor& dp_degree,
    ttnn::Tensor& output,
    uint32_t dp,
    DataType output_dtype);

}  // namespace ttnn::operations::experimental::attention::detail
