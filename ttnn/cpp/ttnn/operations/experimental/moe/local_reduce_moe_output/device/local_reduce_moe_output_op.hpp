#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::moe {

using namespace tt::tt_metal;

struct LocalReduceMoeOutput {
    MemoryConfig output_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors = {}) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}