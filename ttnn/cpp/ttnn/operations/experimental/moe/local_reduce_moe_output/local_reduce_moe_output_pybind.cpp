// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "local_reduce_moe_output_pybind.hpp"
#include "local_reduce_moe_output.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::detail {
namespace py = pybind11;

void bind_local_reduce_moe_output(py::module& module) {
    const auto doc = R"doc(
local_reduce_moe_output(input_hidden_state: ttnn.Tensor, token_idx_map: ttnn.Tensor, routed_token_weights: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, num_tokens: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Performs intra-device reduction by gathering expert outputs back to token order and applying routing weights.

This operation is part of the MoE V2 pipeline and performs the final accumulation step,
converting from expert-organized output to token-organized output.

For each global token index t in [0, T):
1. Initialize: output[t, :] = 0
2. For each local expert e in [0, E/D-1):
   - Read t_e = num_routed_tokens[e, 0]
   - For each expert-local position i in [0, t_e):
     - If token_idx_map[e, i] == t:
       - Read hidden state: hidden = input_hidden_state[e, i, :]
       - Read routing weight: weight = routed_token_weights[e, i]
       - Accumulate: output[t, :] += hidden * weight

Args:
    * :attr:`input_hidden_state`: (E/D, T, H) bfloat16 tensor, ROW_MAJOR, sharded across devices
        Expert outputs from moe_bmm (after layout conversion)
        For expert e, only first num_routed_tokens[e, 0] rows contain valid data
    * :attr:`token_idx_map`: (E/D, T) uint32 tensor, ROW_MAJOR, sharded across devices
        Mapping from expert-local position to global token index
        From prepare_moe_routing_tensors
        token_idx_map[e, i] = global token index for i-th position of expert e
    * :attr:`routed_token_weights`: (E/D, T) bfloat16 tensor, ROW_MAJOR, sharded across devices
        Routing weights for each expert-token assignment
        From prepare_moe_routing_tensors
        routed_token_weights[e, i] = routing weight for i-th token of expert e
    * :attr:`num_routed_tokens`: (E/D, 1) uint32 tensor, ROW_MAJOR, sharded across devices
        Device-local token counts
        num_routed_tokens[e, 0] = number of valid entries for expert e
    * :attr:`num_tokens`: int
        Total number of tokens (T)

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    (T, H) bfloat16 tensor, ROW_MAJOR layout
    Final output for all tokens on this device (contains weighted sum of all expert contributions per token)
    Note: Still needs inter-device allreduce to combine experts from all devices

Example:
    >>> # After down projection
    >>> down_output = ttnn.experimental.moe_bmm(combined, down_weights, num_routed)
    >>>
    >>> # Convert to ROW_MAJOR for reduce
    >>> down_output_rm = ttnn.to_layout(down_output, ttnn.ROW_MAJOR_LAYOUT)
    >>>
    >>> # Local reduce (intra-device)
    >>> local_output = ttnn.local_reduce_moe_output(
    ...     down_output_rm,       # (E/D, T, H) ROW_MAJOR
    ...     token_idx_map,        # (E/D, T)
    ...     routed_token_weights, # (E/D, T)
    ...     num_routed_tokens,    # (E/D, 1)
    ...     num_tokens            # scalar T
    ... )  # Returns: (T, H) ROW_MAJOR - per device
    >>>
    >>> # Inter-device reduce (allreduce)
    >>> final_output = ttnn.all_reduce(
    ...     local_output,
    ...     mesh_device,
    ...     math_op=ttnn.ReduceType.Sum
    ... )  # Returns: (T, H) ROW_MAJOR - complete result
)doc";

    using OperationType = decltype(ttnn::local_reduce_moe_output);
    ttnn::bind_registered_operation(
        module,
        ttnn::local_reduce_moe_output,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_hidden_state,
               const ttnn::Tensor& token_idx_map,
               const ttnn::Tensor& routed_token_weights,
               const ttnn::Tensor& num_routed_tokens,
               uint32_t num_tokens,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, input_hidden_state, token_idx_map, routed_token_weights,
                           num_routed_tokens, num_tokens, memory_config);
            },
            py::arg("input_hidden_state").noconvert(),
            py::arg("token_idx_map").noconvert(),
            py::arg("routed_token_weights").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::arg("num_tokens"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::detail
