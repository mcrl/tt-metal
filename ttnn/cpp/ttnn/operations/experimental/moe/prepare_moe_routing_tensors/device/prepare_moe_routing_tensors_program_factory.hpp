// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe {

tt::tt_metal::operation::ProgramWithCallbacks prepare_moe_routing_tensors_multi_core(
    const Tensor& selected_experts,
    const Tensor& routing_weights,
    const Tensor& device_expert_mapping,
    Tensor& num_routed_tokens,
    Tensor& routed_tokens,
    Tensor& routed_token_weights,
    Tensor& token_idx_map,
    Tensor& num_tiled_tokens,
    uint32_t num_experts,
    uint32_t num_local_experts,
    uint32_t max_tokens_per_expert);

}  // namespace ttnn::operations::experimental::moe