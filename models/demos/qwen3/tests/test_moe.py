import pytest
import torch
import json
import os
from torch import nn

import ttnn

from transformers import AutoConfig
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.tt.model_cache import get_model_path
from models.demos.qwen3.tt.ccl import TT_CCL

def create_test_config():
    model_path = get_model_path()
    config_path = os.path.join(model_path, "config.json")

    with open(config_path, "r") as f:
        data = json.load(f)
    return Qwen3MoeConfig.from_dict(data)

def load_reference_layer(layer_idx=0, seq_len=32):
    model_path = get_model_path()
    config = AutoConfig.from_pretrained(model_path)

    config.max_batch_size = 32
    config.max_seq_len = seq_len
    config._attn_implementation = "sdpa"

    layer = Qwen3MoeDecoderLayer(config, layer_idx)

    weight_path = os.path.join(model_path, f"layer_{layer_idx}.pt")
    if os.path.exists(weight_path):
        layer.load_state_dict(torch.load(weight_path))
    else:
        print(f"Warning: Weight file {weight_path} not found, using random weights")

    layer.to(torch.bfloat16)

    return layer

@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (64, 64),
        (1, 32),
        (10, 32),
        (40, 32),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_moe_prefill(batch_size, seq_len, mesh_device):
    """Compare TT Sparse MoE implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer = load_reference_layer(seq_len=seq_len)
    ref_mlp = ref_layer.mlp

    config = create_test_config()
    layer_idx = 0
    start_pos = 0
    ccl = TT_CCL(mesh_device)
    tt_mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device, ccl)

    tt_mlp.gate.weight.data = ref_mlp.gate.weight.data.clone()

    for i, tt_expert in enumerate(tt_mlp.experts):
        if i < len(ref_mlp.experts):
            ref_expert = ref_mlp.experts[i]
            tt_expert.gate_proj.weight.data = ref_expert.gate_proj.weight.data.clone()
            tt_expert.up_proj.weight.data = ref_expert.up_proj.weight.data.clone()
            tt_expert.down_proj.weight.data = ref_expert.down_proj.weight.data.clone()
    tt_mlp.setup_tt()

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    ref_output = ref_mlp(hidden_states)

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tt = tt_mlp(hidden_states_tt)
    tt_output = ttnn.to_torch(
        output_tt, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    compare_tensor_pcc(ref_output, tt_output[:batch_size, :, :])

@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 1),
        (20, 1),
        (40, 1),
        (64, 1)
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_moe_decode(batch_size, seq_len, mesh_device):
    """Compare TT Sparse MoE implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer = load_reference_layer(seq_len=seq_len)
    ref_mlp = ref_layer.mlp

    config = create_test_config()
    layer_idx = 0
    start_pos = 0
    ccl = TT_CCL(mesh_device)
    tt_mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device, ccl)

    tt_mlp.gate.weight.data = ref_mlp.gate.weight.data.clone()

    for i, tt_expert in enumerate(tt_mlp.experts):
        if i < len(ref_mlp.experts):
            ref_expert = ref_mlp.experts[i]
            tt_expert.gate_proj.weight.data = ref_expert.gate_proj.weight.data.clone()
            tt_expert.up_proj.weight.data = ref_expert.up_proj.weight.data.clone()
            tt_expert.down_proj.weight.data = ref_expert.down_proj.weight.data.clone()
    tt_mlp.setup_tt()

    hidden_states = torch.randn(1, seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16)
    ref_output = ref_mlp(hidden_states.reshape(batch_size, seq_len, config.hidden_size))

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tt = tt_mlp(hidden_states_tt, mode=InferenceMode.DECODE)
    tt_output = ttnn.to_torch(
        output_tt, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )

    compare_tensor_pcc(ref_output, tt_output.squeeze(1)[:1, :, :].reshape(batch_size, seq_len, config.hidden_size))

if __name__ == "__main__":
    pytest.main([__file__])
