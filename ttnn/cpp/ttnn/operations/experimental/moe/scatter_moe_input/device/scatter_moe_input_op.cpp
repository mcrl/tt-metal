#include "scatter_moe_input_op.hpp"
#include "scatter_moe_input_program_factory.hpp"

namespace ttnn::operations::experimental::moe {

void ScatterMoeInput::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& num_routed_tokens = input_tensors.at(1);
    const auto& routed_tokens = input_tensors.at(2);

    // Validate input_hidden_state: (T, H) bfloat16, ROW_MAJOR
    TT_FATAL(
        input_hidden_state.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "input_hidden_state must be bfloat16, got {}",
        input_hidden_state.dtype());

    TT_FATAL(
        input_hidden_state.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "input_hidden_state must be ROW_MAJOR layout, got {}",
        input_hidden_state.layout());

    const auto& input_shape = input_hidden_state.padded_shape();
    TT_FATAL(
        input_shape.rank() == 2,
        "input_hidden_state must be 2D (T, H), got rank {}",
        input_shape.rank());

    uint32_t num_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    // Validate num_routed_tokens: (E/D) 1D uint32, ROW_MAJOR
    TT_FATAL(
        num_routed_tokens.dtype() == tt::tt_metal::DataType::UINT32,
        "num_routed_tokens must be uint32, got {}",
        num_routed_tokens.dtype());

    TT_FATAL(
        num_routed_tokens.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "num_routed_tokens must be ROW_MAJOR layout, got {}",
        num_routed_tokens.layout());

    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    TT_FATAL(
        num_routed_shape.rank() == 1,
        "num_routed_tokens must be 1D (E/D), got rank {}",
        num_routed_shape.rank());

    uint32_t num_local_experts = num_routed_shape[-1];

    // Validate routed_tokens: (E/D, T) uint32, ROW_MAJOR
    TT_FATAL(
        routed_tokens.dtype() == tt::tt_metal::DataType::UINT32,
        "routed_tokens must be uint32, got {}",
        routed_tokens.dtype());

    TT_FATAL(
        routed_tokens.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "routed_tokens must be ROW_MAJOR layout, got {}",
        routed_tokens.layout());

    const auto& routed_shape = routed_tokens.padded_shape();
    TT_FATAL(
        routed_shape.rank() == 2,
        "routed_tokens must be 2D (E/D, T), got rank {}",
        routed_shape.rank());

    TT_FATAL(
        routed_shape[-2] == num_local_experts,
        "routed_tokens shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts, num_tokens, routed_shape[-2], routed_shape[-1]);

    TT_FATAL(
        routed_shape[-1] == num_tokens,
        "routed_tokens shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts, num_tokens, routed_shape[-2], routed_shape[-1]);

    // Validate same device
    TT_FATAL(
        input_hidden_state.device() == num_routed_tokens.device(),
        "All input tensors must be on the same device");

    TT_FATAL(
        input_hidden_state.device() == routed_tokens.device(),
        "All input tensors must be on the same device");
}

std::vector<TensorSpec> ScatterMoeInput::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& num_routed_tokens = input_tensors.at(1);

    const auto& input_shape = input_hidden_state.padded_shape();
    uint32_t num_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    uint32_t num_local_experts = num_routed_shape[-1];  // 1D tensor (E/D,)

    // Output shape: (E/D, T, H)
    ttnn::Shape output_shape({num_local_experts, num_tokens, hidden_dim});

    return {TensorSpec(
        output_shape,
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config))};
}

std::vector<Tensor> ScatterMoeInput::create_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    auto output_specs = compute_output_specs(input_tensors);
    const auto& input_tensor = input_tensors.at(0);

    return {create_device_tensor(output_specs[0], input_tensor.device())};
}

operation::ProgramWithCallbacks ScatterMoeInput::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& num_routed_tokens = input_tensors.at(1);
    const auto& routed_tokens = input_tensors.at(2);
    auto& output = output_tensors.at(0);

    return detail::scatter_moe_input_multi_core(
        input_hidden_state, num_routed_tokens, routed_tokens, output);
}

}