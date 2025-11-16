#pragma once

#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::extract_attention_input::detail {

void bind_extract_attention_input(pybind11::module& module);

}
