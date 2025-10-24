import pytest
import torch
import json
import os
from torch import nn

import ttnn

from transformers import AutoConfig
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode

# from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeDecoderLayer
# from models.demos.qwen3.reference.rope import precompute_freqs_cis

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

from models.demos.qwen3.tt.attention import Qwen3MoeAttention
# from models.demos.qwen3.tt.rope import precompute_freqs_cis_v2

from models.demos.qwen3.utils.test_utils import compare_tensor_pcc
from models.demos.qwen3.utils.timer import set_and_get_device_cache

from models.tt_transformers.tt.rope import RotarySetup

def create_test_config():
    config_path = "/home/chundoong/qwen3-235b-test/config.json"

    with open(config_path, "r") as f:
        data = json.load(f)
    return Qwen3MoeConfig.from_dict(data)


def load_reference_layer(layer_idx=0, seq_len=32):
    config = AutoConfig.from_pretrained("/home/chundoong/qwen3-235b-test/")

    config.max_batch_size = 32
    config.max_seq_len = seq_len
    config._attn_implementation = "sdpa"

    layer = Qwen3MoeDecoderLayer(config, layer_idx)

    weight_path = f"/home/chundoong/qwen3-235b-test/layer.pt"
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
        (32, 128),
        # (2, 64),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_prefill(batch_size, seq_len, mesh_device):
    """Compare TT Attention implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer(seq_len=seq_len)
    ref_attention = ref_layer.self_attn
    
    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0
    start_pos = 0

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    attention_mask = (
        torch.full(size=(1, 1, seq_len, start_pos + seq_len), fill_value=True, dtype=torch.bool)
        .triu_(diagonal=start_pos + 1)
        .logical_not_()
    )
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    ref_position_embeddings = ref_rope(hidden_states, position_ids)
    ref_output = ref_attention(
        hidden_states, position_embeddings=ref_position_embeddings, attention_mask=attention_mask
    )[0]

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
        batch_size=batch_size,
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
        torch.full((batch_size // 4,), start_pos),
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

    output_tt = tt_attention(
        hidden_states=hidden_states_tt,
        rot_mats=rot_mats,
        trans_mat=trans_mat,
        start_pos=start_pos_tt,
        page_table=page_table_tt,
        mode=InferenceMode.PREFILL,
    )
    output_tt = ttnn.to_layout(output_tt, ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.to_torch(
        output_tt, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:batch_size, :, :]

    compare_tensor_pcc(ref_output, tt_output)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (32, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_decode(batch_size, seq_len, mesh_device):
    """Compare TT Attention implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer(seq_len=seq_len)
    ref_attention = ref_layer.self_attn

    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0
    start_pos = 0

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    attention_mask = (
        torch.full(size=(1, 1, seq_len, start_pos + seq_len), fill_value=True, dtype=torch.bool)
        .triu_(diagonal=start_pos + 1)
        .logical_not_()
    )
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    ref_position_embeddings = ref_rope(hidden_states, position_ids)
    ref_output = ref_attention(
        hidden_states, position_embeddings=ref_position_embeddings, attention_mask=attention_mask
    )[0]

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
        batch_size=batch_size,
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )

    position_idxs = torch.full((batch_size // 4,), start_pos, dtype=torch.long)
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
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    start_pos_tt = ttnn.as_tensor(
        torch.full((batch_size,), start_pos),
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

    output_tt = tt_attention(
        hidden_states=hidden_states_tt,
        rot_mats=rot_mats,
        trans_mat=trans_mat,
        start_pos=start_pos_tt,
        page_table=page_table_tt,
        mode=InferenceMode.DECODE,
    )
    output_tt = ttnn.reshape(output_tt, (batch_size, 1, config.hidden_size))
    output_tt = ttnn.to_layout(output_tt, ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.to_torch(
        output_tt, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:batch_size, :, :]

    compare_tensor_pcc(ref_output, tt_output)

@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_prefill_and_decode(batch_size, seq_len, mesh_device):
    """Test prefill with paged cache followed by one decode step."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer()
    ref_attention = ref_layer.self_attn
    
    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0

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
        batch_size=batch_size,
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )

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

    hidden_states_full = torch.randn(batch_size, seq_len + 1, config.hidden_size, dtype=torch.bfloat16)
    
    print(f"\n=== Running REFERENCE for full sequence (seq_len={seq_len + 1}) ===")
    attention_mask_full = (
        torch.full(size=(1, 1, seq_len + 1, seq_len + 1), fill_value=True, dtype=torch.bool)
        .triu_(diagonal=1)
        .logical_not_()
    )
    position_ids_full = torch.arange(0, seq_len + 1, dtype=torch.long).unsqueeze(0)
    ref_position_embeddings_full = ref_rope(hidden_states_full, position_ids_full)
    ref_output_full = ref_attention(
        hidden_states_full, 
        position_embeddings=ref_position_embeddings_full, 
        attention_mask=attention_mask_full
    )[0]

    # ========== PREFILL PHASE ==========
    print(f"\n=== Running TT PREFILL phase (seq_len={seq_len}) ===")
    start_pos = 0
    
    hidden_states_prefill = hidden_states_full[:, :seq_len, :]

    # TT prefill
    rot_mats_prefill = rope.cos_matrix, rope.sin_matrix
    trans_mat_prefill = rope.transformation_mat_prefill

    start_pos_tt_prefill = ttnn.as_tensor(
        torch.full((batch_size,), start_pos),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    hidden_states_tt_prefill = ttnn.from_torch(
        hidden_states_prefill,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tt_prefill = tt_attention(
        hidden_states=hidden_states_tt_prefill,
        rot_mats=rot_mats_prefill,
        trans_mat=trans_mat_prefill,
        start_pos=start_pos_tt_prefill,
        page_table=page_table_tt,
        mode=InferenceMode.PREFILL,
    )
    output_tt_prefill = ttnn.to_layout(output_tt_prefill, ttnn.ROW_MAJOR_LAYOUT)

    tt_output_prefill = ttnn.to_torch(
        output_tt_prefill, 
        dtype=torch.bfloat16, 
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:batch_size, :, :]

    print("Comparing prefill outputs...")
    compare_tensor_pcc(ref_output_full[:, :seq_len, :], tt_output_prefill)

    # ========== DECODE PHASE ==========
    print(f"\n=== Running TT DECODE phase (1 step) ===")
    start_pos_decode = seq_len
    
    # Use the decode token input from full sequence
    hidden_states_decode = hidden_states_full[:, seq_len:seq_len + 1, :]

    # TT decode
    position_idxs = torch.full((batch_size,), start_pos_decode, dtype=torch.long)
    rot_mats_decode = rope.get_rot_mats(position_idxs)
    trans_mat_decode = rope.transformation_mat

    start_pos_tt_decode = ttnn.as_tensor(
        torch.full((batch_size,), start_pos_decode),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    hidden_states_decode_reshaped = hidden_states_decode.reshape((1, 1, batch_size, config.hidden_size))
    hidden_states_tt_decode = ttnn.from_torch(
        hidden_states_decode_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output_tt_decode = tt_attention(
        hidden_states=hidden_states_tt_decode,
        rot_mats=rot_mats_decode,
        trans_mat=trans_mat_decode,
        start_pos=start_pos_tt_decode,
        page_table=page_table_tt,
        mode=InferenceMode.DECODE,
    )
    output_tt_decode = ttnn.reshape(output_tt_decode, (batch_size, 1, config.hidden_size))
    output_tt_decode = ttnn.to_layout(output_tt_decode, ttnn.ROW_MAJOR_LAYOUT)

    tt_output_decode = ttnn.to_torch(
        output_tt_decode,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:batch_size, :, :]

    print("Comparing decode outputs...")
    compare_tensor_pcc(ref_output_full[:, seq_len:seq_len + 1, :], tt_output_decode)

    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    pytest.main([__file__])
