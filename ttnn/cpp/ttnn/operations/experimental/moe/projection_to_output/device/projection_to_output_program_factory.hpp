// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe {

tt::tt_metal::operation::ProgramWithCallbacks projection_to_output_multi_core(
    const Tensor& combined_activations,
    const Tensor& routed_tokens,
    const Tensor& num_routed_tokens,
    const Tensor& routed_token_weights,
    const Tensor& down_proj_weights,
    const Tensor& device_expert_mapping,
    Tensor& output,
    uint32_t num_tokens,
    uint32_t hidden_dim,
    uint32_t expert_dim,
    uint32_t experts_per_device,
    uint32_t max_tokens_per_expert,
    uint32_t top_k);

}  // namespace ttnn::operations::experimental::moe