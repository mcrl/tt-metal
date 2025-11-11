#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::moe::detail {
namespace py = pybind11;

void bind_prepare_moe_routing_tensors(py::module& module);

}