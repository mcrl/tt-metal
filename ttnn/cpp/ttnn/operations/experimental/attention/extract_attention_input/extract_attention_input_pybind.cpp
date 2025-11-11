#include "extract_attention_input_pybind.hpp"
#include "extract_attention_input.hpp"
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::experimental::extract_attention_input::detail {
namespace py = pybind11;

void bind_extract_attention_input(py::module& module) {
    const auto doc = R"doc(
extract_attention_input(hidden_state: ttnn.Tensor, dp_degree: ttnn.Tensor, mesh_device: ttnn.MeshDevice, *, output_dtype: ttnn.DataType = None, memory_config: ttnn.MemoryConfig = None, queue_id: int = 0) -> ttnn.Tensor

Extracts batch chunks per device for attention input (unified prefill/decode operation).

Replaces matrix multiplication-based approach with dedicated operation that
extracts consecutive batch chunks for each device along the data parallelism dimension.

For device with index dev_idx in DP dimension:
1. Calculate dp = mesh_device.shape[0]
2. Calculate batch_per_device = B // dp
3. Extract batch slice based on mode

Args:
    * :attr:`hidden_state`: Input tensor, TILE_LAYOUT, replicated across devices
        - Prefill mode: [B, S, H] bfloat16 tensor
        - Decode mode: [1, 1, B, H] bfloat16 tensor
        where B = batch size, S = sequence length, H = hidden dimension
    * :attr:`dp_degree`: [1] integer tensor (UINT32 or INT32), per-device
        Data parallelism degree for this device (device's row index in mesh)
    * :attr:`mesh_device`: MeshDevice
        Device mesh to extract data parallelism degree from (dp = mesh_device.shape[0])

Keyword Args:
    * :attr:`output_dtype`: DataType (default: same as input)
        Output data type (BFLOAT16 or BFLOAT8_B)
    * :attr:`memory_config`: Memory configuration for output tensor (default: same as input)
    * :attr:`queue_id`: Command queue ID (default: 0)

Returns:
    Extracted batch chunk for this device with optional dtype conversion:
    - Prefill mode: [B//dp, 1, S, H] tensor, TILE_LAYOUT, output dtype
    - Decode mode: [1, 1, B//dp, H] tensor, TILE_LAYOUT, output dtype

Example:
    >>> # Prefill mode: Input [512, 128, 2048] replicated across 8 devices (dp=8)
    >>> hidden_state_prefill = ttnn.from_torch(
    ...     torch_input,
    ...     dtype=ttnn.bfloat16,
    ...     layout=ttnn.TILE_LAYOUT,
    ...     device=mesh_device,
    ...     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    ... )
    >>>
    >>> # Create dp_degree tensor: each device has its DP index (0, 1, 2, ..., 7)
    >>> dp_degree = create_dp_degree_tensor(mesh_device)  # User-provided function
    >>>
    >>> # Extract batch chunks: each device gets [64, 1, 128, 2048]
    >>> attention_input = ttnn.extract_attention_input(
    ...     hidden_state_prefill,
    ...     dp_degree,
    ...     mesh_device,
    ...     output_dtype=ttnn.bfloat8_b
    ... )
    >>>
    >>> # Decode mode: Input [1, 1, 512, 2048] replicated across 8 devices (dp=8)
    >>> hidden_state_decode = ttnn.from_torch(
    ...     torch_input,
    ...     dtype=ttnn.bfloat16,
    ...     layout=ttnn.TILE_LAYOUT,
    ...     device=mesh_device,
    ...     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    ... )
    >>>
    >>> # Extract batch chunks: each device gets [1, 1, 64, 2048]
    >>> attention_input = ttnn.extract_attention_input(
    ...     hidden_state_decode,
    ...     dp_degree,
    ...     mesh_device,
    ...     output_dtype=ttnn.bfloat8_b
    ... )
)doc";

    using OperationType = decltype(ttnn::extract_attention_input);
    ttnn::bind_registered_operation(
        module,
        ttnn::extract_attention_input,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& hidden_state,
               const ttnn::Tensor& dp_degree,
               const MeshDevice& mesh_device,
               const std::optional<DataType>& output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, hidden_state, dp_degree, mesh_device, output_dtype, memory_config);
            },
            py::arg("hidden_state").noconvert(),
            py::arg("dp_degree").noconvert(),
            py::arg("mesh_device"),
            py::kw_only(),
            py::arg("output_dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}