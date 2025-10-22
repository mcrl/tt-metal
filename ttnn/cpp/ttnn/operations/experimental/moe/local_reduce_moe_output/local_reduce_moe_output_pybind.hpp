// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::detail {

void bind_local_reduce_moe_output(pybind11::module& module);

}  // namespace ttnn::operations::experimental::detail
