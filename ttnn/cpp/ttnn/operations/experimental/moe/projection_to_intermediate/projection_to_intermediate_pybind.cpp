// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_intermediate_pybind.hpp"
#include "projection_to_intermediate.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::projection_to_intermediate::detail {
namespace py = pybind11;

void bind_projection_to_intermediate(py::module& module) {
    const auto doc = R"doc(
projection_to_intermediate(hidden_states: ttnn.Tensor, routed_tokens: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, expert_weights: ttnn.Tensor, device_expert_mapping: ttnn.Tensor, top_k: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Performs batched matrix multiplication for MoE expert processing with sparse routing.

Each expert processes only its assigned tokens using the routing information:
1. For each local expert (E/D per device):
   - Get global expert index from device_expert_mapping
   - Get token count T_e from num_routed_tokens[global_expert_idx]
   - Gather T_e tokens from hidden_states using routed_tokens[global_expert_idx]
   - Perform matmul: (T_e × H) @ (H × H') = T_e × H'
   - Write output sequentially to pre-allocated tensor

This operation is used for both gate_proj and up_proj in MoE layers.

Args:
    * :attr:`hidden_states`: Input token embeddings (T, H), ROW_MAJOR, replicated across devices
    * :attr:`routed_tokens`: Token indices per expert (E, max_tokens), ROW_MAJOR, replicated
    * :attr:`num_routed_tokens`: Token count per expert (1, E), ROW_MAJOR, replicated
    * :attr:`expert_weights`: Expert weight matrices (E/D, H, H'), ROW_MAJOR, sharded across devices
    * :attr:`device_expert_mapping`: Global expert indices (E/D,), ROW_MAJOR, sharded across devices
    * :attr:`top_k`: Number of experts selected per token

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor
    * :attr:`queue_id`: Command queue ID

Returns:
    Output tensor of shape (K*T, H') containing projection outputs (zero-padded)

Example:
    >>> # After preparing routing tensors
    >>> num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
    ...     selected_experts, routing_weights, num_experts)
    >>>
    >>> # Perform expert projection (e.g., gate_proj)
    >>> gate_output = ttnn.projection_to_intermediate(
    ...     hidden_states,           # (T, H) replicated
    ...     routed_tokens,           # (E, max_tokens) replicated
    ...     num_routed,              # (1, E) replicated
    ...     gate_weights,            # (E/D, H, H') sharded
    ...     device_expert_mapping,   # (E/D,) sharded
    ...     top_k=8                  # Number of experts per token
    ... )
)doc";

    using OperationType = decltype(ttnn::projection_to_intermediate);
    ttnn::bind_registered_operation(
        module,
        ttnn::projection_to_intermediate,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& hidden_states,
               const ttnn::Tensor& routed_tokens,
               const ttnn::Tensor& num_routed_tokens,
               const ttnn::Tensor& expert_weights,
               const ttnn::Tensor& device_expert_mapping,
               uint32_t top_k,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, hidden_states, routed_tokens, num_routed_tokens, expert_weights, device_expert_mapping, top_k, memory_config);
            },
            py::arg("hidden_states").noconvert(),
            py::arg("routed_tokens").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::arg("expert_weights").noconvert(),
            py::arg("device_expert_mapping").noconvert(),
            py::arg("top_k"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::projection_to_intermediate::detail
