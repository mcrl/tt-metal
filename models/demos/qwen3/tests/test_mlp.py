# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeMLP
from models.demos.qwen3.tt.mlp import MLP1D
from models.demos.qwen3.utils.run_config import create_run_config
from models.demos.qwen3.utils.test_utils import (
    compare_tensor_pcc,
    get_model_config,
    run_module_forward,
)


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        ("prefill", 512),
        ("prefill", 2048),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    hf_config,
    tmp_path,
    mesh_device,
    ccl,
):
    num_module_layers, _ = mesh_device.shape

    # Reference torch MLP
    reference_model = Qwen3MoeMLP(hf_config, intermediate_size=hf_config.moe_intermediate_size).eval()
    state_dict = reference_model.to(torch.bfloat16).state_dict()
    torch_input = torch.randn(num_module_layers, 1, seq_len, hf_config.hidden_size)
    reference_model = reference_model.to(torch.float32)
    reference_output = reference_model(torch_input)

    # TTNN configs/state
    weight_config = MLP1D.convert_weights(hf_config, [state_dict] * num_module_layers, tmp_path, mesh_device)
    model_config = get_model_config(MLP1D, mode, hf_config, mesh_device)
    model_state = MLP1D.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Inputs
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Forward
    tt_output = run_module_forward(MLP1D, mode, tt_input, run_config)

    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    compare_tensor_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
