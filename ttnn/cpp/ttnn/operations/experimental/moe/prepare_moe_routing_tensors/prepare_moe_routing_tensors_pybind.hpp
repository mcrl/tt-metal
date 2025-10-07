// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::prepare_moe_routing_tensors::detail {
namespace py = pybind11;

void bind_prepare_moe_routing_tensors(py::module& module);

}  // namespace ttnn::operations::experimental::prepare_moe_routing_tensors::detail