#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::scatter_moe_input::detail {

void bind_scatter_moe_input(pybind11::module& module);

}