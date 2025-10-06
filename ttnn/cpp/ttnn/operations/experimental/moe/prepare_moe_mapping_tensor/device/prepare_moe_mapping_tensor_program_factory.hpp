// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe::detail {

tt::tt_metal::operation::ProgramWithCallbacks prepare_moe_mapping_tensor_single_core(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    Tensor& output,
    uint32_t num_experts);

}  // namespace ttnn::operations::experimental::moe::detail
