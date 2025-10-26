// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::attention::detail {

tt::tt_metal::operation::ProgramWithCallbacks extract_attention_input_decode_single_core(
    const ttnn::Tensor& hidden_state,
    ttnn::Tensor& output,
    uint32_t dp,
    tt::tt_metal::DataType output_dtype);

}  // namespace ttnn::operations::experimental::attention::detail
