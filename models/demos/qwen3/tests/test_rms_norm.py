# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeRMSNorm
from models.demos.qwen3.tt.rms_norm import DistributedRMSNorm, RMSNorm
from models.demos.qwen3.utils.run_config import create_run_config
from models.demos.qwen3.utils.test_utils import (
    compare_tensor_pcc,
    get_model_config,
    load_state_dict,
    run_module_forward,
)


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
        ("prefill", 128),
        ("prefill", 512),
        ("prefill", 2048),
    ],
)
@pytest.mark.parametrize(
    "reference_layernorm_path, RMSNormClass, hf_config_size_attr",
    [
        (None, DistributedRMSNorm, "hidden_size"),
        ("model.layers.0.input_layernorm", DistributedRMSNorm, "hidden_size"),
        ("model.layers.0.post_attention_layernorm", DistributedRMSNorm, "hidden_size"),
        (None, RMSNorm, "head_dim"),  # q_norm / k_norm style per-head norm would pass head_dim; keep simple unit test
    ],
)
def test_forward_pass(
    RMSNormClass,
    hf_config_size_attr,
    mode,
    seq_len,
    reference_layernorm_path,
    model_path,
    hf_config,
    tmp_path,
    mesh_device,
    ccl,
):
    num_module_layers, _ = mesh_device.shape

    hidden_size = getattr(hf_config, hf_config_size_attr, hf_config.hidden_size)

    if reference_layernorm_path is None:
        reference_model = Qwen3MoeRMSNorm(hidden_size=hidden_size, eps=hf_config.rms_norm_eps).eval()
        state_dict = reference_model.to(torch.bfloat16).state_dict()

        torch_input = torch.randn(num_module_layers, 1, seq_len, hidden_size)
        reference_model = reference_model.to(torch.float32)
        reference_output = reference_model(torch_input)
    else:
        state_dict = load_state_dict(model_path, reference_layernorm_path)
        torch_input = torch.randn(num_module_layers, 1, seq_len, hidden_size)
        reference_model = Qwen3MoeRMSNorm(hidden_size=hidden_size, eps=hf_config.rms_norm_eps).eval().to(torch.float32)
        # Load weight
        with torch.no_grad():
            reference_model.weight.copy_(state_dict["weight"].to(torch.float32))
        reference_output = reference_model(torch_input)

    weight_config = RMSNormClass.convert_weights(hf_config, [state_dict] * num_module_layers, tmp_path, mesh_device)
    model_config = get_model_config(RMSNormClass, mode, hf_config, mesh_device)
    model_state = RMSNormClass.create_state(
        hf_config, mesh_device, *[ccl for _ in range(1) if RMSNormClass is DistributedRMSNorm]
    )
    run_config = create_run_config(model_config, weight_config, model_state)

    if RMSNormClass is not DistributedRMSNorm:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        memory_config = run_config["input_memory_config"]
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_device.shape, dims=(0, -1 if RMSNormClass is DistributedRMSNorm else None)
        ),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = run_module_forward(RMSNormClass, mode, tt_input, run_config)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -1)),
    )
    if RMSNormClass is RMSNorm:
        tt_output_torch = tt_output_torch[..., :hidden_size]

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    compare_tensor_pcc(tt_output_torch, reference_output, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
