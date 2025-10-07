// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_down_projection_pybind.hpp"
#include "moe_down_projection.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::moe_down_projection::detail {
namespace py = pybind11;

void bind_moe_down_projection(py::module& module) {
    const auto doc = R"doc(
moe_down_projection(combined_activations: ttnn.Tensor, routed_tokens: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, routed_token_weights: ttnn.Tensor, down_proj_weights: ttnn.Tensor, device_expert_mapping: ttnn.Tensor, num_tokens: int, top_k: int, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Performs the down projection step in MoE layers with routing weight application and accumulation.

This operation is Step 4 of the MoE computation pipeline. It takes the combined activations
from Step 3 (after SiLU activation and elementwise multiplication) and performs the down
projection to produce the final MoE output.

Each expert processes its assigned tokens by:
1. Performing matmul: (T_e × H') @ (H' × H) = T_e × H
2. Multiplying each result by the corresponding routing weight
3. Accumulating (not overwriting) results to the final output tensor

Args:
    * :attr:`combined_activations`: Combined gate*up activations (T_d × H') from Step 3
    * :attr:`routed_tokens`: Token indices routed to each expert (E × max_tokens)
    * :attr:`num_routed_tokens`: Number of tokens routed to each expert (1 × E)
    * :attr:`routed_token_weights`: Routing weights for each token-expert pair (E × max_tokens)
    * :attr:`down_proj_weights`: Down projection weight matrices (E/D × H' × H) per device
    * :attr:`device_expert_mapping`: Global expert indices assigned to this device (E/D)
    * :attr:`num_tokens`: Total number of tokens (T)
    * :attr:`top_k`: Number of experts selected per token (K)

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor
    * :attr:`queue_id`: Command queue ID

Returns:
    * :attr:`output`: Final MoE output tensor (T × H) with accumulated results

Example:
    >>> # Assuming we have combined activations from Step 3
    >>> output = ttnn.moe_down_projection(
    ...     combined_activations,
    ...     routed_tokens,
    ...     num_routed_tokens,
    ...     routed_token_weights,
    ...     down_proj_weights,
    ...     device_expert_mapping,
    ...     num_tokens=4096,
    ...     top_k=8
    ... )
)doc";

    using OperationType = decltype(ttnn::moe_down_projection);
    ttnn::bind_registered_operation(
        module,
        ttnn::moe_down_projection,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& combined_activations,
               const ttnn::Tensor& routed_tokens,
               const ttnn::Tensor& num_routed_tokens,
               const ttnn::Tensor& routed_token_weights,
               const ttnn::Tensor& down_proj_weights,
               const ttnn::Tensor& device_expert_mapping,
               uint32_t num_tokens,
               uint32_t top_k,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(
                    queue_id,
                    combined_activations,
                    routed_tokens,
                    num_routed_tokens,
                    routed_token_weights,
                    down_proj_weights,
                    device_expert_mapping,
                    num_tokens,
                    top_k,
                    memory_config
                );
            },
            py::arg("combined_activations").noconvert(),
            py::arg("routed_tokens").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::arg("routed_token_weights").noconvert(),
            py::arg("down_proj_weights").noconvert(),
            py::arg("device_expert_mapping").noconvert(),
            py::arg("num_tokens"),
            py::arg("top_k"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::moe_down_projection::detail