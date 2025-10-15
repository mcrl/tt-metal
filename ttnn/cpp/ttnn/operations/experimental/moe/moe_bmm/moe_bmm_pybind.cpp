// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_bmm_pybind.hpp"
#include "moe_bmm.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::moe_bmm::detail {
namespace py = pybind11;

void bind_moe_bmm(py::module& module) {
    const auto doc = R"doc(
moe_bmm(input: ttnn.Tensor, weights: ttnn.Tensor, num_routed_tokens: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Performs batched matrix multiplication for Mixture-of-Experts (MoE) processing.

For each expert e, computes:
    output[e, :, :] = input[e, :, :] @ weights[e, :, :]
    
This multiplies (T × H_in) @ (H_in × H_out) → (T × H_out) per expert.
Only the first num_routed_tokens[e, 0] rows produce non-zero results.

Args:
    * :attr:`input`: (E/D, T, H_in) bfloat16 tensor, TILE_LAYOUT, sharded across devices
                     E/D experts per device, T max tokens per expert, H_in input hidden size
    * :attr:`weights`: (E/D, H_in, H_out) bfloat16 tensor, TILE_LAYOUT, sharded across devices
                       Expert weight matrices (one per local expert)
    * :attr:`num_routed_tokens`: (E/D, 1) uint32 2D tensor, ROW_MAJOR layout, sharded
                                 Access as num_routed_tokens[e, 0] for local expert e

Keyword Args:
    * :attr:`memory_config`: Memory configuration for output tensor
    * :attr:`queue_id`: Command queue ID

Returns:
    (E/D, T, H_out) bfloat16 tensor containing batched matmul outputs (zero-padded after active tokens)

Notes:
    - Single-core implementation using TILE_LAYOUT for efficient computation
    - Each device processes E/D experts in parallel (expert parallelism)
    - Output rows beyond num_routed_tokens[e, 0] are zero for each expert

Example:
    >>> # Input from previous expert processing step
    >>> expert_input = ...  # (E/D, T, H_in) TILE_LAYOUT
    >>> expert_weights = ...  # (E/D, H_in, H_out) TILE_LAYOUT
    >>> num_routed = ...  # (E/D, 1) ROW_MAJOR
    >>>
    >>> # Perform batched matmul per expert
    >>> output = ttnn.experimental.moe_bmm(
    ...     expert_input,
    ...     expert_weights,
    ...     num_routed,
    ...     memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ...     queue_id=0
    >>> )
    >>> # output shape: (E/D, T, H_out) TILE_LAYOUT

)doc";

    using OperationType = decltype(ttnn::experimental::moe_bmm);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::moe_bmm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const ttnn::Tensor& weights,
               const ttnn::Tensor& num_routed_tokens,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, input, weights, num_routed_tokens, memory_config);
            },
            py::arg("input").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("num_routed_tokens").noconvert(),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::moe_bmm::detail
