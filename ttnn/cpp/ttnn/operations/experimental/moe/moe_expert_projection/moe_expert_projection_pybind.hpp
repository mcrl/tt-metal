// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::moe_expert_projection::detail {

void bind_moe_expert_projection(pybind11::module& module);

}  // namespace ttnn::operations::experimental::moe_expert_projection::detail
