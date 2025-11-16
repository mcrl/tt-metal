#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::moe_bmm::detail {

void bind_moe_bmm(pybind11::module& module);

}