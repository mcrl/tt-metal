# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeAttention
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.attention import Qwen3MoeAttentionTT
from models.demos.qwen3.utils.run_config import create_run_config
from models.demos.qwen3.utils.test_utils import assert_tensor_pcc, get_model_config, run_module_forward


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("prefill", 1),  # minimal case without rotary mismatch
        ("prefill", 16),
    ],
)
def test_attention_prefill_forward(mode, seq_len, hf_config, tmp_path, mesh_device):
    torch.manual_seed(0)

    # Reference module
    ref_attn = Qwen3MoeAttention(hf_config, layer_idx=0).eval()
    state_dict = {
        "q_proj.weight": ref_attn.q_proj.weight.detach().clone(),
        "k_proj.weight": ref_attn.k_proj.weight.detach().clone(),
        "v_proj.weight": ref_attn.v_proj.weight.detach().clone(),
        "o_proj.weight": ref_attn.o_proj.weight.detach().clone(),
        "q_norm.weight": ref_attn.q_norm.weight.detach().clone(),
        "k_norm.weight": ref_attn.k_norm.weight.detach().clone(),
    }

    torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size)
    with torch.no_grad():
        ref_out = ref_attn(
            hidden_states=torch_input.view(1, seq_len, hf_config.hidden_size),
            start_pos=0,
            position_embeddings=torch.zeros(seq_len, hf_config.head_dim),
            attention_mask=torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool),
        )
        ref_out = ref_out.view(1, 1, seq_len, hf_config.hidden_size)

    # Setup TT
    weight_config = Qwen3MoeAttentionTT.convert_weights(hf_config, [state_dict], tmp_path, mesh_device)
    model_config = get_model_config(Qwen3MoeAttentionTT, mode, hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, {"mesh_device": mesh_device})

    tt_in = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = run_module_forward(Qwen3MoeAttentionTT, mode, tt_in, run_config)

    tt_out_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1)))

    ttnn.deallocate(tt_in)
    ttnn.deallocate(tt_out)

    assert_tensor_pcc(tt_out_torch, ref_out, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])
