#include "local_reduce_moe_output_op.hpp"
#include "local_reduce_moe_output_program_factory.hpp"

namespace ttnn::operations::experimental::moe {

void LocalReduceMoeOutput::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto& input_hidden_state = input_tensors.at(0);
    const auto& token_idx_map = input_tensors.at(1);
    const auto& routed_token_weights = input_tensors.at(2);
    const auto& num_routed_tokens = input_tensors.at(3);

    const auto& input_shape = input_hidden_state.padded_shape();
    TT_FATAL(input_shape.rank() == 3, "input_hidden_state must be 3D (E/D, T, H), got rank {}", input_shape.rank());
    TT_FATAL(input_hidden_state.dtype() == DataType::BFLOAT16, "input_hidden_state must be bfloat16");
    TT_FATAL(input_hidden_state.layout() == Layout::ROW_MAJOR, "input_hidden_state must be ROW_MAJOR layout");
    TT_FATAL(input_shape[-1] % 1024 == 0, "hidden_dim must be divisible by 1024");

    uint32_t num_local_experts = input_shape[-3];
    uint32_t num_tokens = input_shape[-2];

    const auto& token_idx_shape = token_idx_map.padded_shape();
    TT_FATAL(token_idx_shape.rank() == 2, "token_idx_map must be 2D (E/D, T), got rank {}", token_idx_shape.rank());
    TT_FATAL(token_idx_map.dtype() == DataType::UINT32, "token_idx_map must be uint32");
    TT_FATAL(token_idx_map.layout() == Layout::ROW_MAJOR, "token_idx_map must be ROW_MAJOR layout");
    TT_FATAL(
        token_idx_shape[-2] == num_local_experts && token_idx_shape[-1] == num_tokens,
        "token_idx_map shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts,
        num_tokens,
        token_idx_shape[-2],
        token_idx_shape[-1]);

    const auto& weights_shape = routed_token_weights.padded_shape();
    TT_FATAL(
        weights_shape.rank() == 2, "routed_token_weights must be 2D (E/D, T), got rank {}", weights_shape.rank());
    TT_FATAL(routed_token_weights.dtype() == DataType::BFLOAT16, "routed_token_weights must be bfloat16");
    TT_FATAL(routed_token_weights.layout() == Layout::ROW_MAJOR, "routed_token_weights must be ROW_MAJOR layout");
    TT_FATAL(
        weights_shape[-2] == num_local_experts && weights_shape[-1] == num_tokens,
        "routed_token_weights shape mismatch: expected ({}, {}), got ({}, {})",
        num_local_experts,
        num_tokens,
        weights_shape[-2],
        weights_shape[-1]);

    const auto& num_routed_shape = num_routed_tokens.padded_shape();
    TT_FATAL(
        num_routed_shape.rank() == 1, "num_routed_tokens must be 1D (E/D), got rank {}", num_routed_shape.rank());
    TT_FATAL(num_routed_tokens.dtype() == DataType::UINT32, "num_routed_tokens must be uint32");
    TT_FATAL(num_routed_tokens.layout() == Layout::ROW_MAJOR, "num_routed_tokens must be ROW_MAJOR layout");
    TT_FATAL(
        num_routed_shape[-1] == num_local_experts,
        "num_routed_tokens shape mismatch: expected ({}), got ({})",
        num_local_experts,
        num_routed_shape[-1]);

    auto* device = input_hidden_state.device();
    TT_FATAL(token_idx_map.device() == device, "All input tensors must be on the same device");
    TT_FATAL(routed_token_weights.device() == device, "All input tensors must be on the same device");
    TT_FATAL(num_routed_tokens.device() == device, "All input tensors must be on the same device");
}

std::vector<TensorSpec> LocalReduceMoeOutput::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_hidden_state = input_tensors.at(0);
    const auto& input_shape = input_hidden_state.padded_shape();
    uint32_t num_tokens = input_shape[-2];
    uint32_t hidden_dim = input_shape[-1];

    ttnn::Shape output_shape({num_tokens, hidden_dim});

    return {TensorSpec(output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), output_mem_config))};
}

std::vector<Tensor> LocalReduceMoeOutput::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto output_specs = compute_output_specs(input_tensors);
    return {create_device_tensor(output_specs[0], input_tensors.at(0).device())};
}

operation::ProgramWithCallbacks LocalReduceMoeOutput::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return detail::local_reduce_moe_output(
        input_tensors.at(0),
        input_tensors.at(1),
        input_tensors.at(2),
        input_tensors.at(3),
        output_tensors.at(0));
}

}