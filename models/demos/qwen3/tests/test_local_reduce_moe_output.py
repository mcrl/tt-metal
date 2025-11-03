# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
from loguru import logger


def torch_local_reduce_moe_output(
    input_hidden_state: torch.Tensor,  # (E/D, T, H)
    token_idx_map: torch.Tensor,  # (E/D, T)
    routed_token_weights: torch.Tensor,  # (E/D, T)
    num_routed_tokens: torch.Tensor,  # (E/D,) 1D or (E/D, 1) 2D
    num_tokens: int,
) -> torch.Tensor:
    """
    PyTorch reference implementation of local_reduce_moe_output.

    For each global token index t:
    1. Initialize output[t, :] = 0
    2. For each local expert e:
       - For each expert-local position i with valid data:
         - If token_idx_map[e, i] == t:
           - Accumulate: output[t, :] += input[e, i, :] * weight[e, i]
    """
    num_local_experts = input_hidden_state.shape[0]
    hidden_dim = input_hidden_state.shape[2]

    # Initialize output
    output = torch.zeros(num_tokens, hidden_dim, dtype=input_hidden_state.dtype)

    # For each token
    for t in range(num_tokens):
        # Accumulate contributions from all experts
        for e in range(num_local_experts):
            if num_routed_tokens.ndim == 2:
                t_e = num_routed_tokens[e, 0].item()
            else:
                t_e = num_routed_tokens[e].item()

            # For each expert-local position
            for i in range(t_e):
                if token_idx_map[e, i] == t:
                    # Get hidden state and weight
                    hidden = input_hidden_state[e, i, :]
                    weight = routed_token_weights[e, i]
                    # Accumulate
                    output[t, :] += hidden * weight
                    # Each token appears at most once per expert
                    break

    return output


@pytest.mark.parametrize(
    "num_local_experts, num_tokens, hidden_dim, top_k",
    [
        # (4, 32, 128, 1),   # Small test case
        # (4, 32, 128, 2),   # Small test case
        # (8, 64, 256, 4),   # Medium test case
        # (16, 128, 512, 4),  # Larger test case
        # (4, 4, 1024, 2),  # Larger test case
        # (16, 512, 2048, 1),  # Larger test case
        (16, 128, 2048, 8),  # (real case)
        # (16, 1024, 2048, 8),  # (real case)
    ],
)
def test_local_reduce_moe_output(device, num_local_experts, num_tokens, hidden_dim, top_k):
    """
    Test local_reduce_moe_output operation.

    Args:
        num_local_experts: Number of local experts (E/D)
        num_tokens: Number of tokens (T)
        hidden_dim: Hidden dimension (H)
        top_k: Number of experts per token
    """
    logger.info(f"Testing local_reduce_moe_output with:")
    logger.info(f"  num_local_experts={num_local_experts}, num_tokens={num_tokens}")
    logger.info(f"  hidden_dim={hidden_dim}, top_k={top_k}")

    # Generate random input data
    torch.manual_seed(42)

    # Input hidden state: (E/D, T, H) - expert outputs
    input_hidden_state = torch.randn(num_local_experts, num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Create routing information
    # Simulate that each expert gets approximately num_tokens * top_k / num_local_experts tokens
    num_routed_tokens = torch.zeros(num_local_experts, 1, dtype=torch.int32)
    token_idx_map = torch.zeros(num_local_experts, num_tokens, dtype=torch.int32)
    routed_token_weights = torch.zeros(num_local_experts, num_tokens, dtype=torch.bfloat16)

    # For simplicity, assign tokens to experts in a round-robin fashion
    # Each token goes to top_k experts
    for t in range(num_tokens):
        # Select top_k experts for this token (round-robin)
        for k in range(top_k):
            expert_idx = (t * top_k + k) % num_local_experts

            # Get current position for this expert
            pos = num_routed_tokens[expert_idx, 0].item()

            # Assign this token to this expert
            token_idx_map[expert_idx, pos] = t
            routed_token_weights[expert_idx, pos] = torch.randn(1, dtype=torch.bfloat16).item()

            # Increment count for this expert
            num_routed_tokens[expert_idx, 0] += 1

    # Zero out unused entries in input_hidden_state
    for e in range(num_local_experts):
        t_e = num_routed_tokens[e, 0].item()
        if t_e < num_tokens:
            input_hidden_state[e, t_e:, :] = 0

    logger.info(f"  Token distribution across experts: {num_routed_tokens.squeeze().tolist()}")

    # Compute PyTorch reference
    output_torch = torch_local_reduce_moe_output(
        input_hidden_state,
        token_idx_map,
        routed_token_weights,
        num_routed_tokens,
        num_tokens,
    )

    logger.info(f"  PyTorch output shape: {output_torch.shape}")

    # Convert to TTNN tensors
    input_tt = ttnn.from_torch(
        input_hidden_state,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    token_idx_map_tt = ttnn.from_torch(
        token_idx_map,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    routed_token_weights_tt = ttnn.from_torch(
        routed_token_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Reshape to 1D for compatibility with new API
    num_routed_tokens_1d = num_routed_tokens.squeeze()

    num_routed_tokens_tt = ttnn.from_torch(
        num_routed_tokens_1d,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN operation
    output_tt = ttnn.local_reduce_moe_output(
        input_tt,
        token_idx_map_tt,
        routed_token_weights_tt,
        num_routed_tokens_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert back to torch
    output_tt_torch = ttnn.to_torch(output_tt)

    logger.info(f"  TTNN output shape: {output_tt_torch.shape}")

    # Validate shapes
    assert output_tt_torch.shape == output_torch.shape, \
        f"Shape mismatch: {output_tt_torch.shape} vs {output_torch.shape}"

    # Compute PCC (Pearson Correlation Coefficient)
    output_torch_flat = output_torch.flatten().float()
    output_tt_flat = output_tt_torch.flatten().float()

    pcc = torch.corrcoef(torch.stack([output_torch_flat, output_tt_flat]))[0, 1].item()
    logger.info(f"  PCC: {pcc:.6f}")

    # Compute max absolute error
    max_abs_error = torch.max(torch.abs(output_torch - output_tt_torch)).item()
    logger.info(f"  Max absolute error: {max_abs_error:.6f}")

    # Compute mean absolute error
    mean_abs_error = torch.mean(torch.abs(output_torch - output_tt_torch)).item()
    logger.info(f"  Mean absolute error: {mean_abs_error:.6f}")

    # Check if results match
    # For bfloat16, we expect PCC > 0.99
    assert pcc > 0.99, f"PCC too low: {pcc:.6f} (expected > 0.99)"

    logger.info("  ✓ Test passed!")


@pytest.mark.parametrize(
    "num_local_experts, num_tokens, hidden_dim",
    [
        (2, 8, 2048),    # Minimal test case
    ],
)
def test_local_reduce_moe_output_basic(device, num_local_experts, num_tokens, hidden_dim):
    """
    Basic test with simple, predictable values.
    """
    logger.info(f"Testing local_reduce_moe_output (basic) with:")
    logger.info(f"  num_local_experts={num_local_experts}, num_tokens={num_tokens}, hidden_dim={hidden_dim}")

    # Create simple input data
    # Expert 0 processes tokens [0, 2, 4, 6]
    # Expert 1 processes tokens [1, 3, 5, 7]

    input_hidden_state = torch.ones(num_local_experts, num_tokens, hidden_dim, dtype=torch.bfloat16)

    # Set values: expert e, token i -> value = (e+1) * 10 + i
    num_routed_tokens = torch.tensor([[4], [4]], dtype=torch.int32)
    token_idx_map = torch.zeros(num_local_experts, num_tokens, dtype=torch.int32)
    routed_token_weights = torch.zeros(num_local_experts, num_tokens, dtype=torch.bfloat16)

    # Expert 0: tokens [0, 2, 4, 6] with weight 0.5
    for i, t in enumerate([0, 2, 4, 6]):
        token_idx_map[0, i] = t
        routed_token_weights[0, i] = 1.0
        for h in range(hidden_dim):
            input_hidden_state[0, i, h] = 10.0 * i + h

    # Expert 1: tokens [1, 3, 5, 7] with weight 0.3
    for i, t in enumerate([1, 3, 5, 7]):
        token_idx_map[1, i] = t
        routed_token_weights[1, i] = 0.5
        for h in range(hidden_dim):
            input_hidden_state[1, i, h] = 20.0 * i + h

    # Zero out unused entries
    for e in range(num_local_experts):
        t_e = num_routed_tokens[e, 0].item()
        input_hidden_state[e, t_e:, :] = 0

    # Compute reference
    output_torch = torch_local_reduce_moe_output(
        input_hidden_state,
        token_idx_map,
        routed_token_weights,
        num_routed_tokens,
        num_tokens,
    )

    # Convert to TTNN tensors
    input_tt = ttnn.from_torch(input_hidden_state, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    token_idx_map_tt = ttnn.from_torch(token_idx_map, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    routed_token_weights_tt = ttnn.from_torch(routed_token_weights, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    num_routed_tokens_tt = ttnn.from_torch(num_routed_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run TTNN operation
    output_tt = ttnn.local_reduce_moe_output(
        input_tt, token_idx_map_tt, routed_token_weights_tt,
        num_routed_tokens_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tt_torch = ttnn.to_torch(output_tt)

    # Debug: print first few values
    for t in range(min(8, num_tokens)):
        logger.info(f"    Token {t}: {output_torch[t, :10].tolist()} ... {output_torch[t, -10:].tolist()}")
        logger.info(f"    Token {t}: {output_tt_torch[t, :10].tolist()} ... {output_tt_torch[t, -10:].tolist()}")

    # Validate
    pcc = torch.corrcoef(torch.stack([output_torch.flatten().float(), output_tt_torch.flatten().float()]))[0, 1].item()
    logger.info(f"  PCC: {pcc:.6f}")

    # Check max/mean abs error
    max_abs_error = torch.max(torch.abs(output_torch - output_tt_torch)).item()
    mean_abs_error = torch.mean(torch.abs(output_torch - output_tt_torch)).item()
    logger.info(f"  Max absolute error: {max_abs_error:.6f}")
    logger.info(f"  Mean absolute error: {mean_abs_error:.6f}")

    assert pcc > 0.99, f"PCC too low: {pcc:.6f}"
    logger.info("  ✓ Basic test passed!")


if __name__ == "__main__":
    # For manual testing
    import sys
    sys.path.append("models/demos/qwen3")
    from conftest import device

    with device() as dev:
        test_local_reduce_moe_output_basic(dev, 2, 8, 64)
        test_local_reduce_moe_output(dev, 4, 32, 128, 2)
