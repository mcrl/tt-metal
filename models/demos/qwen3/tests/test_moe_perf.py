import pytest
import torch
import json
import os
from torch import nn

import ttnn
import tracy

from transformers import AutoConfig
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock
from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.demos.qwen3.utils.timer import set_and_get_device_cache


def create_test_config():
    config_path = "/shared/models/Qwen3-30B-A3B/config.json"

    with open(config_path, "r") as f:
        data = json.load(f)
    return Qwen3MoeConfig.from_dict(data)


def load_reference_layer(layer_idx=0, seq_len=32):
    config = AutoConfig.from_pretrained("/shared/models/Qwen3-30B-A3B/")

    config.max_batch_size = 32
    config.max_seq_len = seq_len
    config._attn_implementation = "sdpa"

    layer = Qwen3MoeDecoderLayer(config, layer_idx)

    weight_path = f"/shared/models/Qwen3-30B-A3B/layer_{layer_idx}.pt"
    if os.path.exists(weight_path):
        layer.load_state_dict(torch.load(weight_path)["state_dict"])
    else:
        print(f"Warning: Weight file {weight_path} not found, using random weights")

    layer.to(torch.bfloat16)

    return layer


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (32, 128)
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_moe_prefill(batch_size, seq_len, mesh_device):
    """Compare TT Sparse MoE implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)
    # Load reference layer
    ref_layer = load_reference_layer(seq_len=seq_len)
    ref_mlp = ref_layer.mlp
    # Create TT MoE
    config = create_test_config()
    layer_idx = 0
    start_pos = 0
    tt_mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device)

    # Copy gate weights from reference
    tt_mlp.gate.weight.data = ref_mlp.gate.weight.data.clone()

    # Copy expert weights from reference
    for i, tt_expert in enumerate(tt_mlp.experts):
        if i < len(ref_mlp.experts):
            ref_expert = ref_mlp.experts[i]
            tt_expert.gate_proj.weight.data = ref_expert.gate_proj.weight.data.clone()
            tt_expert.up_proj.weight.data = ref_expert.up_proj.weight.data.clone()
            tt_expert.down_proj.weight.data = ref_expert.down_proj.weight.data.clone()
    tt_mlp.setup_tt()

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tracy.signpost("Warmup")
    for _ in range(5):
        output_tt = tt_mlp(hidden_states_tt)

    tracy.signpost("Run")
    output_tt = tt_mlp(hidden_states_tt)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (32, 1)
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_moe_decode(batch_size, seq_len, mesh_device):
    """Compare TT Sparse MoE implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)
    # Load reference layer
    ref_layer = load_reference_layer(seq_len=seq_len)
    ref_mlp = ref_layer.mlp
    # Create TT MoE
    config = create_test_config()
    layer_idx = 0
    start_pos = 0
    tt_mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device)

    # Copy gate weights from reference
    tt_mlp.gate.weight.data = ref_mlp.gate.weight.data.clone()

    # Copy expert weights from reference
    for i, tt_expert in enumerate(tt_mlp.experts):
        if i < len(ref_mlp.experts):
            ref_expert = ref_mlp.experts[i]
            tt_expert.gate_proj.weight.data = ref_expert.gate_proj.weight.data.clone()
            tt_expert.up_proj.weight.data = ref_expert.up_proj.weight.data.clone()
            tt_expert.down_proj.weight.data = ref_expert.down_proj.weight.data.clone()
    tt_mlp.setup_tt()

    hidden_states = torch.randn(1, seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16)

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tracy.signpost("Warmup")
    for _ in range(5):
        output_tt = tt_mlp(hidden_states_tt, mode=InferenceMode.DECODE)

    tracy.signpost("Run")
    output_tt = tt_mlp(hidden_states_tt, mode=InferenceMode.DECODE)


if __name__ == "__main__":
    pytest.main([__file__])
