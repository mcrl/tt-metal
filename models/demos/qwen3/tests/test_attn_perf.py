import pytest
import torch
import json
import os
from torch import nn

import ttnn
import tracy

from transformers import AutoConfig
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

from models.demos.qwen3.tt.attention import Qwen3MoeAttention

from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.demos.qwen3.utils.timer import set_and_get_device_cache

from models.tt_transformers.tt.rope import RotarySetup

def create_test_config():
    config_path = "/shared/models/Qwen3-235B-A22B/config.json"

    with open(config_path, "r") as f:
        data = json.load(f)
    return Qwen3MoeConfig.from_dict(data)


def load_reference_layer(layer_idx=0):
    config = AutoConfig.from_pretrained("/shared/models/Qwen3-235B-A22B/")
    config._attn_implementation = "sdpa"

    layer = Qwen3MoeDecoderLayer(config, layer_idx)

    weight_path = f"/shared/models/Qwen3-235B-A22B/layer_{layer_idx}.pt"
    if os.path.exists(weight_path):
        layer.load_state_dict(torch.load(weight_path))
    else:
        print(f"Warning: Weight file {weight_path} not found, using random weights")
    layer.to(torch.bfloat16)

    rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    return layer, rotary_emb


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        # (8, 64),
        (128, 128),
        # (2, 64),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_prefill(batch_size, seq_len, mesh_device):
    """Compare TT Attention implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer()
    ref_attention = ref_layer.self_attn
    
    config = create_test_config()
    config.max_batch_size = batch_size

    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0
    start_pos = 0

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

    tt_attention = Qwen3MoeAttention(config, layer_idx, mesh_device)

    tt_attention.q_proj.weight.data = ref_attention.q_proj.weight.data.clone()
    tt_attention.k_proj.weight.data = ref_attention.k_proj.weight.data.clone()
    tt_attention.v_proj.weight.data = ref_attention.v_proj.weight.data.clone()
    tt_attention.o_proj.weight.data = ref_attention.o_proj.weight.data.clone()
    tt_attention.q_norm.weight.data = ref_attention.q_norm.weight.data.clone()
    tt_attention.k_norm.weight.data = ref_attention.k_norm.weight.data.clone()
    tt_attention.setup_tt()

    rope = RotarySetup(
        device=mesh_device,
        batch_size=batch_size // mesh_device.shape[0],
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )

    rot_mats = rope.cos_matrix, rope.sin_matrix
    trans_mat = rope.transformation_mat_prefill

    permutation = torch.randperm(config.max_num_blocks, device="cpu")
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(config.max_batch_size, config.max_num_blocks // config.max_batch_size)
    page_table_tt = ttnn.as_tensor(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    start_pos_tt = ttnn.as_tensor(
        torch.full((batch_size // mesh_device.shape[0],), start_pos),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

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
        output_tt = tt_attention(
            hidden_states=hidden_states_tt,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos_tt,
            page_table=page_table_tt,
            mode=InferenceMode.PREFILL,
        )
    
    tracy.signpost("Run")
    output_tt = tt_attention(
        hidden_states=hidden_states_tt,
        rot_mats=rot_mats,
        trans_mat=trans_mat,
        start_pos=start_pos_tt,
        page_table=page_table_tt,
        mode=InferenceMode.PREFILL,
    )

@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_decode(batch_size, seq_len, mesh_device):
    """Compare TT Attention implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer()
    ref_attention = ref_layer.self_attn

    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024
    
    layer_idx = 0
    start_pos = seq_len

    hidden_states = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16)

    tt_attention = Qwen3MoeAttention(config, layer_idx, mesh_device)

    tt_attention.q_proj.weight.data = ref_attention.q_proj.weight.data.clone()
    tt_attention.k_proj.weight.data = ref_attention.k_proj.weight.data.clone()
    tt_attention.v_proj.weight.data = ref_attention.v_proj.weight.data.clone()
    tt_attention.o_proj.weight.data = ref_attention.o_proj.weight.data.clone()
    tt_attention.q_norm.weight.data = ref_attention.q_norm.weight.data.clone()
    tt_attention.k_norm.weight.data = ref_attention.k_norm.weight.data.clone()
    tt_attention.setup_tt()

    rope = RotarySetup(
        device=mesh_device,
        batch_size=batch_size // mesh_device.shape[0],
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )

    position_idxs = torch.full((batch_size // mesh_device.shape[0],), start_pos, dtype=torch.long)
    rot_mats = rope.get_rot_mats(position_idxs)
    trans_mat = rope.transformation_mat

    permutation = torch.randperm(config.max_num_blocks, device="cpu")
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(config.max_batch_size, config.max_num_blocks // config.max_batch_size)
    page_table_tt = ttnn.as_tensor(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None)),
    )
    start_pos_tt = ttnn.as_tensor(
        torch.full((batch_size // mesh_device.shape[0],), start_pos),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    hidden_states = hidden_states.reshape((1, 1, batch_size, config.hidden_size))
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
        output_tt = tt_attention(
            hidden_states=hidden_states_tt,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos_tt,
            page_table=page_table_tt,
            mode=InferenceMode.DECODE,
        )
    
    tracy.signpost("Run")
    output_tt = tt_attention(
        hidden_states=hidden_states_tt,
        rot_mats=rot_mats,
        trans_mat=trans_mat,
        start_pos=start_pos_tt,
        page_table=page_table_tt,
        mode=InferenceMode.DECODE,
    )

if __name__ == "__main__":
    pytest.main([__file__])
