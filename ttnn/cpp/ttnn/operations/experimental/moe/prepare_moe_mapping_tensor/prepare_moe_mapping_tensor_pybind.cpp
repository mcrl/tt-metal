// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/moe/prepare_moe_mapping_tensor/prepare_moe_mapping_tensor.hpp"
#include "ttnn/operations/experimental/moe/prepare_moe_mapping_tensor/prepare_moe_mapping_tensor_pybind.hpp"

namespace ttnn::operations::experimental::moe::detail {
namespace py = pybind11;

void bind_prepare_moe_mapping_tensor(py::module& module) {
    auto doc = R"doc(
prepare_moe_mapping_tensor(selected_experts: ttnn.Tensor, routing_weights: ttnn.Tensor, num_experts: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Prepares MoE expert mapping tensor from selected expert indices and routing weights.

This operation converts sparse expert selection (T x K) into a dense mapping tensor (T x E) where:
- T is the number of tokens
- K is the number of experts per token (top-k)
- E is the total number of experts

For each token, the output contains the routing weight at position (token, expert_idx) for selected experts,
and zero for non-selected experts.

Args:
    * :attr:`selected_experts`: Tensor of selected expert indices (T x K), ROW_MAJOR layout, int32 dtype
    * :attr:`routing_weights`: Tensor of routing weights (T x K), ROW_MAJOR layout, bfloat16 dtype
    * :attr:`num_experts`: Total number of experts (E)

Keyword Args:
    * :attr:`memory_config`: Memory configuration for the output tensor
    * :attr:`queue_id`: Command queue ID

Example:
    >>> # T=4 tokens, K=2 experts per token, E=8 total experts
    >>> selected_experts = ttnn.from_torch(torch.tensor([[0, 3], [1, 5], [2, 7], [0, 4]]), dtype=ttnn.int32)
    >>> routing_weights = ttnn.from_torch(torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]]), dtype=ttnn.bfloat16)
    >>> output = ttnn.experimental.prepare_moe_mapping_tensor(selected_experts, routing_weights, num_experts=8)
    >>> # output shape: (4, 8) with weights at selected positions, zeros elsewhere
)doc";

    using OperationType = decltype(ttnn::prepare_moe_mapping_tensor);
    ttnn::bind_registered_operation(
        module,
        ttnn::prepare_moe_mapping_tensor,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& selected_experts,
               const ttnn::Tensor& routing_weights,
               uint32_t num_experts,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, selected_experts, routing_weights, num_experts, memory_config); },
            py::arg("selected_experts").noconvert(),
            py::arg("routing_weights").noconvert(),
            py::arg("num_experts"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::moe::detail
