#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::attention {

using tt::tt_metal::MemoryConfig;
using tt::tt_metal::DataType;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorLayout;

struct ExtractAttentionInput {
    MemoryConfig output_mem_config;
    DataType output_dtype;
    uint32_t dp;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

}
