"""
Test suite for extract_attention_input operations (prefill and decode modes).

These operations replace the matrix multiplication-based approach with
dedicated operations that extract consecutive batch chunks per device.
"""

import pytest
import torch
import ttnn


def torch_random(shape, low=-1, high=1, dtype=torch.float32):
    """Generate random tensor with values in [low, high]."""
    return torch.rand(shape, dtype=dtype) * (high - low) + low


def create_dp_degree_tensor(mesh_device):
    """
    Create dp_degree tensor for mesh device.

    Each device at position (row, col) gets scalar value 'row'.
    This represents the device's index in the data parallelism dimension.

    Args:
        mesh_device: MeshDevice

    Returns:
        Mesh tensor where each device has its row index as a scalar [1] tensor
    """
    dp, tp = mesh_device.shape
    num_devices = dp * tp

    dp_degree_host = torch.tensor(
        [row for row in range(dp) for _ in range(tp)],
        dtype=torch.int32
    )

    dp_degree = ttnn.from_torch(
        dp_degree_host,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, None),
            mesh_shape=(num_devices, 1)
        ),
    )

    return dp_degree


@pytest.fixture(
    params=[
        (1, 8),
        (2, 4),
        (4, 2),
        (8, 1),
    ],
    ids=["1x8", "2x4", "4x2", "8x1"],
)
def mesh_device_with_shape(request):
    """
    Create mesh device with specific shape for testing.

    Yields mesh devices with shapes: 1x8, 2x4, 4x2, 8x1
    Skips if the requested shape is not available on the hardware.
    """
    dp, tp = request.param
    num_devices = dp * tp

    num_available_devices = ttnn.GetNumAvailableDevices()
    if num_available_devices < num_devices:
        pytest.skip(
            f"Test requires {num_devices} devices ({dp}x{tp} mesh), "
            f"but only {num_available_devices} available"
        )

    device_params = {
        "trace_region_size": 128 * 1024 * 1024,
        "num_command_queues": 2,
        "mesh_shape": ttnn.MeshShape(dp, tp),
    }

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    mesh_device = ttnn.open_mesh_device(**device_params)

    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def torch_extract_attention_input_prefill(hidden_state: torch.Tensor, dp: int, device_idx: int) -> torch.Tensor:
    """
    PyTorch reference for extract_attention_input_prefill.

    Args:
        hidden_state: [B, S, H] tensor
        dp: Data parallelism degree
        device_idx: Device index (0 to dp-1)

    Returns:
        [B//dp, 1, S, H] tensor for the specified device
    """
    B, S, H = hidden_state.shape
    batch_per_device = B // dp

    start_batch = device_idx * batch_per_device
    end_batch = (device_idx + 1) * batch_per_device

    output = hidden_state[start_batch:end_batch, :, :]  # [B//dp, S, H]
    output = output.unsqueeze(1)  # [B//dp, 1, S, H]

    return output


def torch_extract_attention_input_decode(hidden_state: torch.Tensor, dp: int, device_idx: int) -> torch.Tensor:
    """
    PyTorch reference for extract_attention_input_decode.

    Args:
        hidden_state: [1, 1, B, H] tensor
        dp: Data parallelism degree
        device_idx: Device index (0 to dp-1)

    Returns:
        [1, 1, B//dp, H] tensor for the specified device
    """
    _, _, B, H = hidden_state.shape
    batch_per_device = B // dp

    start_batch = device_idx * batch_per_device
    end_batch = (device_idx + 1) * batch_per_device

    output = hidden_state[:, :, start_batch:end_batch, :]  # [1, 1, B//dp, H]

    return output


@pytest.mark.parametrize("batch_size", [512])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_extract_attention_input_prefill(
    mesh_device_with_shape,
    batch_size,
    seq_len,
    hidden_size,
    output_dtype,
):
    """
    Test extract_attention_input_prefill operation.

    Validates that:
    1. Input is correctly extracted for each device along DP dimension
    2. Output shape is [batch_per_device, 1, seq_len, hidden_size]
    3. Output dtype matches requested dtype
    4. Values match PyTorch reference implementation
    """
    dp = mesh_device_with_shape.shape[0]
    if batch_size % dp != 0:
        pytest.skip(f"Batch size {batch_size} not divisible by dp={dp}")

    batch_per_device = batch_size // dp
    if batch_per_device % 32 != 0:
        pytest.skip(f"Batch per device {batch_per_device} not tile-aligned")

    assert (batch_size * seq_len) % 32 == 0, "Input rows (B * S) must be tile-aligned"
    assert hidden_size % 32 == 0, "Hidden dimension must be tile-aligned"

    torch_input = torch_random((batch_size, seq_len, hidden_size), -1, 1, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device_with_shape,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device_with_shape),
    )

    dp_degree = create_dp_degree_tensor(mesh_device_with_shape)

    ttnn_output = ttnn.extract_attention_input(
        ttnn_input,
        dp_degree,
        mesh_device_with_shape,
        output_dtype=output_dtype,
    )

    device_tensors = ttnn.get_device_tensors(ttnn_output)
    num_devices_per_row = mesh_device_with_shape.shape[1]

    device_idx = 0
    for row_idx in range(dp):
        row_outputs = []

        for col_idx in range(num_devices_per_row):
            device_output_ttnn = device_tensors[device_idx]
            device_output_torch = ttnn.to_torch(device_output_ttnn, dtype=torch.bfloat16)

            expected_shape = (batch_per_device, 1, seq_len, hidden_size)
            assert device_output_torch.shape == expected_shape, (
                f"Device {device_idx} (row={row_idx}, col={col_idx}) output shape "
                f"{device_output_torch.shape} != expected {expected_shape}"
            )

            row_outputs.append(device_output_torch)

            torch_reference = torch_extract_attention_input_prefill(torch_input, dp, row_idx)

            if output_dtype == ttnn.bfloat8_b:
                torch.testing.assert_close(
                    device_output_torch.to(torch.float32),
                    torch_reference.to(torch.float32),
                    rtol=1e-1,
                    atol=1e-1,
                    msg=f"Device {device_idx} (row={row_idx}, col={col_idx}) output doesn't match reference"
                )
            else: 
                torch.testing.assert_close(
                    device_output_torch,
                    torch_reference,
                    rtol=1e-2,
                    atol=1e-2,
                    msg=f"Device {device_idx} (row={row_idx}, col={col_idx}) output doesn't match reference"
                )

            print(f"Device {device_idx} (row={row_idx}, col={col_idx}) output matches reference")

            device_idx += 1

        if num_devices_per_row > 1:
            for col_idx in range(1, num_devices_per_row):
                torch.testing.assert_close(
                    row_outputs[col_idx],
                    row_outputs[0],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"DP row {row_idx}: device at col {col_idx} doesn't match col 0 (TP replication failed)"
                )


@pytest.mark.parametrize("batch_size", [512])
@pytest.mark.parametrize("hidden_size", [2048])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_extract_attention_input_decode(
    mesh_device_with_shape,
    batch_size,
    hidden_size,
    output_dtype,
):
    """
    Test extract_attention_input_decode operation.

    Validates that:
    1. Input is correctly extracted for each device along DP dimension (dimension 2)
    2. Output shape is [1, 1, batch_per_device, hidden_size]
    3. Output dtype matches requested dtype
    4. Values match PyTorch reference implementation
    """
    dp = mesh_device_with_shape.shape[0]
    if batch_size % dp != 0:
        pytest.skip(f"Batch size {batch_size} not divisible by dp={dp}")

    batch_per_device = batch_size // dp
    if batch_per_device % 32 != 0:
        pytest.skip(f"Batch per device {batch_per_device} not tile-aligned")

    assert batch_size % 32 == 0, "Batch size must be tile-aligned"
    assert hidden_size % 32 == 0, "Hidden dimension must be tile-aligned"

    torch_input = torch_random((1, 1, batch_size, hidden_size), -1, 1, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device_with_shape,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device_with_shape),
    )

    dp_degree = create_dp_degree_tensor(mesh_device_with_shape)

    ttnn_output = ttnn.extract_attention_input(
        ttnn_input,
        dp_degree,
        mesh_device_with_shape,
        output_dtype=output_dtype,
    )

    device_tensors = ttnn.get_device_tensors(ttnn_output)
    num_devices_per_row = mesh_device_with_shape.shape[1]

    device_idx = 0
    for row_idx in range(dp):
        row_outputs = []

        for col_idx in range(num_devices_per_row):
            device_output_ttnn = device_tensors[device_idx]
            device_output_torch = ttnn.to_torch(device_output_ttnn, dtype=torch.bfloat16)

            expected_shape = (1, 1, batch_per_device, hidden_size)
            assert device_output_torch.shape == expected_shape, (
                f"Device {device_idx} (row={row_idx}, col={col_idx}) output shape "
                f"{device_output_torch.shape} != expected {expected_shape}"
            )

            row_outputs.append(device_output_torch)

            torch_reference = torch_extract_attention_input_decode(torch_input, dp, row_idx)

            if output_dtype == ttnn.bfloat8_b:
                torch.testing.assert_close(
                    device_output_torch.to(torch.float32),
                    torch_reference.to(torch.float32),
                    rtol=1e-1,
                    atol=1e-1,
                )
            else:
                torch.testing.assert_close(
                    device_output_torch,
                    torch_reference,
                    rtol=1e-2,
                    atol=1e-2,
                )

            device_idx += 1

        if num_devices_per_row > 1:
            for col_idx in range(1, num_devices_per_row):
                torch.testing.assert_close(
                    row_outputs[col_idx],
                    row_outputs[0],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"DP row {row_idx}: device at col {col_idx} doesn't match col 0 (TP replication failed)"
                )
