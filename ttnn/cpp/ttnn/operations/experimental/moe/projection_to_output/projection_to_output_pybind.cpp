// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "projection_to_output_pybind.hpp"
#include "projection_to_output.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::projection_to_output::detail {
namespace py = pybind11;

void bind_projection_to_output(py::module& module) {
    const auto doc = R"doc(
projection_to_output(combined_activations: ttnn.Tensor, routed_tokens: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, routed_token_weights: ttnn.Tensor, down_proj_weights: ttnn.Tensor, num_tokens: int, top_k: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Performs the down projection step in MoE layers with routing weight application and accumulation.

This operation is Step 4 of the MoE computation pipeline. It takes the combined activations
from Step 3 (after SiLU activation and elementwise multiplication) and performs the down
projection to produce the final MoE output.

Each expert processes its assigned tokens using device-local routing:
1. For each local expert (0 to E/D-1):
   - Performing matmul: (T_e × H') @ (H' × H) = T_e × H
   - Multiplying each result by the corresponding routing weight
   - Accumulating (not overwriting) results to the final output tensor

Routing tensors are device-local from prepare_moe_routing_tensors.

Args:
    * :attr:`combined_activations`: (E/D, T, H') bfloat16 tensor - combined gate*up activations from Step 3
    * :attr:`token_idx_map`: (E/D, max_tokens) uint32 tensor, mapping from expert-local token index to global token index
    * :attr:`routed_tokens`: (E/D, max_tokens) uint32 tensor - device-local token indices, sharded
    * :attr:`num_routed_tokens`: (E/D, 1) uint32 2D tensor - device-local token counts, sharded (access as [e, 0])
    * :attr:`routed_token_weights`: (E/D, max_tokens) bfloat16 tensor - device-local routing weights, sharded
    * :attr:`down_proj_weights`: (E/D, H', H) bfloat16 tensor - down projection weight matrices, sharded
    * :attr:`num_tokens`: Total number of tokens (T)
    * :attr:`top_k`: Number of experts selected per token (K)

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor
    * :attr:`queue_id`: Command queue ID

Returns:
    * :attr:`output`: (E/D, T, H) bfloat16 tensor - partial MoE output (requires allreduce across devices)

Example:
    >>> # Assuming we have combined activations from Step 3
    >>> output = ttnn.projection_to_output(
    ...     combined_activations,
    ...     token_idx_map,
    ...     routed_tokens,
    ...     num_routed_tokens,
    ...     routed_token_weights,
    ...     down_proj_weights,
    ...     num_tokens=4096,
    ...     top_k=8
    ... )
)doc";

    using OperationType = decltype(ttnn::projection_to_output);
    ttnn::bind_registered_operation(
        module,
        ttnn::projection_to_output,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& combined_activations,
               const ttnn::Tensor& token_idx_map,
               const ttnn::Tensor& routed_tokens,
               const ttnn::Tensor& num_routed_tokens,
               const ttnn::Tensor& routed_token_weights,
               const ttnn::Tensor& down_proj_weights,
               uint32_t num_tokens,
               uint32_t top_k,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    combined_activations,
                    token_idx_map,
                    routed_tokens,
                    num_routed_tokens,
                    routed_token_weights,
                    down_proj_weights,
                    num_tokens,
                    top_k,
                    memory_config
                );
            },
            py::arg("combined_activations").noconvert(),
            py::arg("token_idx_map").noconvert(),
            py::arg("routed_tokens").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::arg("routed_token_weights").noconvert(),
            py::arg("down_proj_weights").noconvert(),
            py::arg("num_tokens"),
            py::arg("top_k"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::projection_to_output::detail