// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_attention_input_decode_pybind.hpp"
#include "extract_attention_input_decode.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::extract_attention_input_decode::detail {
namespace py = pybind11;

void bind_extract_attention_input_decode(py::module& module) {
    const auto doc = R"doc(
extract_attention_input_decode(hidden_state: ttnn.Tensor, mesh_device: ttnn.MeshDevice, *, output_dtype: ttnn.DataType = None, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Extracts batch chunks per device for attention input in decode mode.

Replaces matrix multiplication-based approach with dedicated operation that
extracts consecutive batch chunks for each device along the data parallelism dimension.

For device with index dev_idx in DP dimension:
1. Calculate dp = mesh_device.shape[0]
2. Calculate batch_per_device = B // dp
3. Extract batch slice: output = hidden_state[:, :, dev_idx * batch_per_device : (dev_idx+1) * batch_per_device, :]

Args:
    * :attr:`hidden_state`: [1, 1, B, H] bfloat16 tensor, TILE_LAYOUT, replicated across devices
        Input token embeddings where B = batch size, H = hidden dimension
    * :attr:`mesh_device`: MeshDevice
        Device mesh to extract data parallelism degree from (dp = mesh_device.shape[0])

Keyword Args:
    * :attr:`output_dtype`: DataType (default: same as input)
        Output data type (BFLOAT16 or BFLOAT8_B)
    * :attr:`memory_config`: Memory configuration for output tensor (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    [1, 1, B//dp, H] tensor, TILE_LAYOUT, output dtype
    Extracted batch chunk for this device with optional dtype conversion

Example:
    >>> # Input: [1, 1, 512, 2048] replicated across 8 devices
    >>> hidden_state = ttnn.from_torch(
    ...     torch_input,
    ...     dtype=ttnn.bfloat16,
    ...     layout=ttnn.TILE_LAYOUT,
    ...     device=mesh_device,
    ...     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    ... )
    >>>
    >>> # Extract batch chunks: each device gets [1, 1, 64, 2048]
    >>> attention_input = ttnn.extract_attention_input_decode(
    ...     hidden_state,
    ...     mesh_device,
    ...     output_dtype=ttnn.bfloat8_b
    ... )
)doc";

    using OperationType = decltype(ttnn::extract_attention_input_decode);
    ttnn::bind_registered_operation(
        module,
        ttnn::extract_attention_input_decode,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& hidden_state,
               const MeshDevice& mesh_device,
               const std::optional<DataType>& output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, hidden_state, mesh_device, output_dtype, memory_config);
            },
            py::arg("hidden_state").noconvert(),
            py::arg("mesh_device"),
            py::kw_only(),
            py::arg("output_dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace ttnn::operations::experimental::extract_attention_input_decode::detail
