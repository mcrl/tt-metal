// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::projection_to_output::detail {
namespace py = pybind11;
void bind_projection_to_output(py::module& module);
}  // namespace ttnn::operations::experimental::projection_to_output::detail