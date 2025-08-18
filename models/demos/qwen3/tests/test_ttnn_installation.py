import pytest
import torch
import ttnn


def test_ttnn_import():
    """Test that TTNN can be imported successfully"""
    assert ttnn is not None


def test_device_detection():
    """Test that TTNN can detect available devices"""
    device_ids = ttnn.get_device_ids()
    assert len(device_ids) > 0


def test_mesh_creation():
    """Test that we can create a simple mesh device"""
    device_ids = ttnn.get_device_ids()
    if len(device_ids) >= 1:
        # Create a simple 1x1 mesh
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        assert mesh_device is not None
        # Clean up
        ttnn.close_mesh_device(mesh_device)
    else:
        pytest.skip("No devices available for mesh creation")


def test_basic_tensor_creation():
    """Test basic tensor operations"""
    # Create a simple tensor
    tensor = ttnn.from_torch(torch.randn(2, 2))
    assert tensor is not None

    # Convert back to torch
    torch_tensor = ttnn.to_torch(tensor)
    assert torch_tensor.shape == (2, 2)


@pytest.mark.parametrize("shape,dtype", [
    ((1, 1), torch.float32),
    ((2, 2), torch.float32),
    ((4, 4), torch.float32),
    ((1, 1), torch.bfloat16),
    ((2, 2), torch.bfloat16),
    ((4, 4), torch.bfloat16),
])
def test_tensor_shapes_and_types(shape, dtype):
    """Test tensor creation with different shapes and data types"""
    # Create torch tensor with specified shape and dtype
    torch_tensor = torch.randn(*shape, dtype=dtype)

    # Convert to TTNN tensor
    ttnn_tensor = ttnn.from_torch(torch_tensor)
    assert ttnn_tensor is not None

    # Convert back to torch and verify
    converted_tensor = ttnn.to_torch(ttnn_tensor)
    assert converted_tensor.shape == shape
    assert converted_tensor.dtype == dtype


@pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
    (1, 128, 512),
    (2, 64, 1024),
    (4, 32, 2048),
])
def test_transformer_like_tensors(batch_size, seq_len, hidden_size):
    """Test tensor operations with transformer-like dimensions"""
    # Create input tensor similar to transformer hidden states
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Convert to TTNN
    ttnn_tensor = ttnn.from_torch(input_tensor)
    assert ttnn_tensor is not None

    # Convert back and verify
    output_tensor = ttnn.to_torch(ttnn_tensor)
    assert output_tensor.shape == (batch_size, seq_len, hidden_size)
    assert output_tensor.dtype == torch.bfloat16
