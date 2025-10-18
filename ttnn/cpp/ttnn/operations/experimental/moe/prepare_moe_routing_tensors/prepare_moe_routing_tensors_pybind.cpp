// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepare_moe_routing_tensors_pybind.hpp"
#include "prepare_moe_routing_tensors.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::moe::detail {
namespace py = pybind11;

void bind_prepare_moe_routing_tensors(py::module& module) {
    const auto doc = R"doc(
prepare_moe_routing_tensors(selected_experts: ttnn.Tensor, routing_weights: ttnn.Tensor, device_expert_mapping: ttnn.Tensor, num_experts: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]

Converts sparse MoE expert selection into device-local routing tensors for expert-parallel computation.

This operation filters global routing information to only include experts assigned to this device,
enabling efficient parallel processing where each device processes its assigned experts independently.

Args:
    * :attr:`selected_experts`: (T, K) uint32 tensor containing GLOBAL expert indices for each token
    * :attr:`routing_weights`: (T, K) bfloat16 tensor containing routing weights
    * :attr:`device_expert_mapping`: (E/D,) int32 1D tensor containing GLOBAL expert IDs assigned to this device
    * :attr:`num_experts`: Total number of experts (E)

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensors (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    Tuple of five device-local tensors:
    * :attr:`num_routed_tokens`: (E/D, 1) uint32 2D tensor - count of tokens routed to each LOCAL expert (uses 2D shape for per-element pages)
    * :attr:`routed_tokens`: (E/D, max_tokens) uint32 2D tensor - token indices for each LOCAL expert (padded)
    * :attr:`routed_token_weights`: (E/D, max_tokens) bfloat16 2D tensor - routing weights for each LOCAL expert (padded)
    * :attr:`token_idx_map`: (E/D, max_tokens) uint32 2D tensor - mapping from expert-local token index to global token index
    * :attr:`num_tiled_tokens`: (E/D, 1) uint32 2D tensor - number of tiled tokens for each LOCAL expert, computed as $(num\_routed\_tokens[e] + 31) // 32$ where TILE_SIZE=32

Example:
    >>> # T=32 tokens, K=8 top experts, E=128 total experts, D=8 devices
    >>> selected_experts = ttnn.from_torch(torch.randint(0, 128, (32, 8), dtype=torch.int32))
    >>> routing_weights = ttnn.from_torch(torch.rand(32, 8, dtype=torch.bfloat16))
    >>> # Device 0 gets experts 0-15
    >>> device_expert_mapping = ttnn.from_torch(torch.arange(0, 16, dtype=torch.int32))
    >>> num_routed, routed_tokens, routed_weights, tokenidx_map, num_tiled = ttnn.prepare_moe_routing_tensors(
    ...     selected_experts, routing_weights, device_expert_mapping, num_experts=128
    ... )

Note:
    - Each token selects top_k unique experts (no duplicates per token)
    - Output tensors are device-local (only experts assigned to this device)
    - Invalid token indices are marked as 0xFFFFFFFF
    - Padding weights are set to 0
    - token_idx_map[e][t_e] = t_g where t_e is the expert-local index (0-based) and t_g is the global token index
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
               const ttnn::Tensor& device_expert_mapping,
               uint32_t num_experts,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, selected_experts, routing_weights, device_expert_mapping, num_experts, memory_config);
            },
            py::arg("selected_experts").noconvert(),
            py::arg("routing_weights").noconvert(),
            py::arg("device_expert_mapping").noconvert(),
            py::arg("num_experts"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::moe::detail