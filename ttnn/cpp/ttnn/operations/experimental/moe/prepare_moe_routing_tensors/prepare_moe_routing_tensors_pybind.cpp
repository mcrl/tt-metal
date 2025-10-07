// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_routing_tensors_pybind.hpp"
#include "prepare_moe_routing_tensors.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::prepare_moe_routing_tensors::detail {
namespace py = pybind11;

void bind_prepare_moe_routing_tensors(py::module& module) {
    const auto doc = R"doc(
prepare_moe_routing_tensors(selected_experts: ttnn.Tensor, routing_weights: ttnn.Tensor, num_experts: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

Converts sparse MoE expert selection into efficient routing tensors for expert-parallel computation.

This operation transforms per-token expert selection into per-expert token lists, enabling efficient
parallel processing where each expert processes its assigned tokens independently.

Args:
    * :attr:`selected_experts`: (T × K) uint32 tensor containing expert indices for each token
    * :attr:`routing_weights`: (T × K) bfloat16 tensor containing routing weights
    * :attr:`num_experts`: Total number of experts

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensors (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    Tuple of three tensors:
    * :attr:`num_routed_tokens`: (E) uint32 tensor - count of tokens routed to each expert
    * :attr:`routed_tokens`: (E × max_tokens) uint32 tensor - token indices for each expert (padded)
    * :attr:`routed_token_weights`: (E × max_tokens) bfloat16 tensor - routing weights for each expert (padded)

Example:
    >>> # T=32 tokens, K=8 top experts, E=128 total experts
    >>> selected_experts = ttnn.from_torch(torch.randint(0, 128, (32, 8), dtype=torch.int32))
    >>> routing_weights = ttnn.from_torch(torch.rand(32, 8, dtype=torch.bfloat16))
    >>> num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
    ...     selected_experts, routing_weights, num_experts=128
    ... )

Note:
    - Each token selects top_k unique experts (no duplicates per token)
    - Output tensors are padded to rectangular shape for efficient processing
    - Invalid token indices are marked as 0xFFFFFFFF
    - Padding weights are set to 0
)doc";

    using OperationType = decltype(ttnn::prepare_moe_routing_tensors);
    ttnn::bind_registered_operation(
        module,
        ttnn::prepare_moe_routing_tensors,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& selected_experts,
               const ttnn::Tensor& routing_weights,
               uint32_t num_experts,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, selected_experts, routing_weights, num_experts, memory_config);
            },
            py::arg("selected_experts").noconvert(),
            py::arg("routing_weights").noconvert(),
            py::arg("num_experts"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::prepare_moe_routing_tensors::detail