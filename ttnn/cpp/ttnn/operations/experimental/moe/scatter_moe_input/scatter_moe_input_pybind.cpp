// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_moe_input_pybind.hpp"
#include "scatter_moe_input.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::scatter_moe_input::detail {
namespace py = pybind11;

void bind_scatter_moe_input(py::module& module) {
    const auto doc = R"doc(
scatter_moe_input(input_hidden_state: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, routed_tokens: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Rearranges input tokens based on expert assignments for MoE V2 pipeline.

This operation gathers tokens assigned to each local expert into contiguous memory,
enabling efficient batched matrix multiplication in subsequent projection operations.

For each local expert e in [0, E/D-1):
1. Read t_e = num_routed_tokens[e, 0] (number of tokens for this expert)
2. For each position i in [0, t_e):
   - Read global token index: t_{e,i} = routed_tokens[e, i]
   - Gather from input: output[e, i, :] = input_hidden_state[t_{e,i}, :]
3. For remaining positions i in [t_e, T):
   - Zero-pad: output[e, i, :] = 0

Args:
    * :attr:`input_hidden_state`: (T, H) bfloat16 tensor, ROW_MAJOR, replicated across devices
        Input token embeddings where T = number of tokens, H = hidden dimension
    * :attr:`num_routed_tokens`: (E/D, 1) uint32 tensor, ROW_MAJOR, sharded across devices
        Device-local token counts from prepare_moe_routing_tensors
        num_routed_tokens[e, 0] = number of tokens assigned to local expert e
    * :attr:`routed_tokens`: (E/D, T) uint32 tensor, ROW_MAJOR, sharded across devices
        Device-local token indices from prepare_moe_routing_tensors
        routed_tokens[e, i] = global token index for i-th token of expert e
        Valid entries: routed_tokens[e, 0:num_routed_tokens[e, 0]]

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    (E/D, T, H) bfloat16 tensor, ROW_MAJOR layout
    Scattered input organized by expert assignment with zero-padding

Example:
    >>> # After preparing routing tensors
    >>> num_routed, routed_tokens, routed_weights, token_idx_map = \\
    ...     ttnn.prepare_moe_routing_tensors(
    ...         selected_experts, routing_weights, device_expert_mapping, num_experts)
    >>>
    >>> # Scatter input by expert
    >>> scattered_input = ttnn.scatter_moe_input(
    ...     hidden_states,        # (T, H) replicated
    ...     num_routed,           # (E/D, 1) sharded
    ...     routed_tokens         # (E/D, T) sharded
    ... )  # Returns: (E/D, T, H) sharded
    >>>
    >>> # Convert to TILE layout for projections
    >>> scattered_tile = ttnn.to_layout(scattered_input, ttnn.TILE_LAYOUT)
    >>>
    >>> # Now ready for efficient BMM operations
    >>> gate_output = ttnn.experimental.moe_bmm(scattered_tile, gate_weights, num_routed)
)doc";

    using OperationType = decltype(ttnn::scatter_moe_input);
    ttnn::bind_registered_operation(
        module,
        ttnn::scatter_moe_input,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_hidden_state,
               const ttnn::Tensor& num_routed_tokens,
               const ttnn::Tensor& routed_tokens,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, input_hidden_state, num_routed_tokens, routed_tokens, memory_config);
            },
            py::arg("input_hidden_state").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::arg("routed_tokens").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::scatter_moe_input::detail
