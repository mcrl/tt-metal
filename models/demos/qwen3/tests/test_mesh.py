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
    config_path = "/mnt/nvme0/models/qwen3-30b/config.json"

    with open(config_path, "r") as f:
        data = json.load(f)
    return Qwen3MoeConfig.from_dict(data)


def load_reference_layer(layer_idx=0, seq_len=32):
    config = AutoConfig.from_pretrained("/mnt/nvme0/models/qwen3-30b/")

    config.max_batch_size = 32
    config.max_seq_len = seq_len
    config._attn_implementation = "sdpa"

    layer = Qwen3MoeDecoderLayer(config, layer_idx)

    weight_path = f"/mnt/nvme0/models/qwen3-30b/layer_{layer_idx}.pt"
    if os.path.exists(weight_path):
        layer.load_state_dict(torch.load(weight_path)["state_dict"])
    else:
        print(f"Warning: Weight file {weight_path} not found, using random weights")
    layer.to(torch.bfloat16)

    rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    return layer, rotary_emb

@pytest.mark.parametrize(
    "bsz_per_device,seq_len",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_fullmesh(bsz_per_device, seq_len, mesh_device):
    """Test prefill with paged cache followed by one decode step."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    dp_degree = mesh_device.shape[0]
    batch_size = bsz_per_device * dp_degree

    ref_layer, ref_rope = load_reference_layer()
    ref_attention = ref_layer.self_attn
    
    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0

    permutation = torch.randperm(config.max_num_blocks, device="cpu")
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(config.max_batch_size, config.max_num_blocks // config.max_batch_size)

    hidden_states_full = torch.randn(batch_size, seq_len + 1, config.hidden_size, dtype=torch.bfloat16)

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
        batch_size=bsz_per_device,
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )
    
    page_table_tt = ttnn.as_tensor(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
    )

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
        torch.full((bsz_per_device,), start_pos),
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
    position_idxs = torch.full((bsz_per_device,), start_pos_decode, dtype=torch.long)
    rot_mats_decode = rope.get_rot_mats(position_idxs)
    trans_mat_decode = rope.transformation_mat

    start_pos_tt_decode = ttnn.as_tensor(
        torch.full((batch_size,), start_pos_decode),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
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
        save_file_name="fullmesh"
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

@pytest.mark.parametrize(
    "bsz_per_device,seq_len",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_attn_submesh(bsz_per_device, seq_len, mesh_device):
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    dp_degree = mesh_device.shape[0]
    batch_size = bsz_per_device * dp_degree
    num_submeshes = 2
    bsz_per_submesh = batch_size // num_submeshes

    ref_layer, ref_rope = load_reference_layer()
    ref_attention = ref_layer.self_attn
    
    config = create_test_config()
    config.max_batch_size = batch_size
    config.block_size = 32
    config.max_num_blocks = 1024

    layer_idx = 0

    permutation = torch.randperm(config.max_num_blocks, device="cpu")
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(config.max_batch_size, config.max_num_blocks // config.max_batch_size)
    page_table_submeshes = torch.chunk(page_table, 2, dim=0)

    hidden_states_full = torch.randn(batch_size, seq_len + 1, config.hidden_size, dtype=torch.bfloat16)
    hidden_state_submeshes = torch.chunk(hidden_states_full, 2, dim=0)

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

    for i, mesh_device in enumerate(mesh_device.get_submeshes()):    
        page_table = page_table_submeshes[i]
        hidden_states_full = hidden_state_submeshes[i]

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
            batch_size=bsz_per_device,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )

        page_table_tt = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
        )

        # ========== PREFILL PHASE ==========
        print(f"\n=== Running TT PREFILL phase (seq_len={seq_len}) ===")
        start_pos = 0
    
        hidden_states_prefill = hidden_states_full[:, :seq_len, :]

        # TT prefill
        rot_mats_prefill = rope.cos_matrix, rope.sin_matrix
        trans_mat_prefill = rope.transformation_mat_prefill

        start_pos_tt_prefill = ttnn.as_tensor(
            torch.full((bsz_per_device,), start_pos),
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
        )[:bsz_per_submesh, :, :]

        print("Comparing prefill outputs...")
        compare_tensor_pcc(ref_output_full[(bsz_per_submesh) * i:(bsz_per_submesh) * (i + 1), :seq_len, :], tt_output_prefill)

        # ========== DECODE PHASE ==========
        print(f"\n=== Running TT DECODE phase (1 step) ===")
        start_pos_decode = seq_len
    
        # Use the decode token input from full sequence
        hidden_states_decode = hidden_states_full[:, seq_len:seq_len + 1, :]

        # TT decode
        position_idxs = torch.full((bsz_per_device,), start_pos_decode, dtype=torch.long)
        rot_mats_decode = rope.get_rot_mats(position_idxs)
        trans_mat_decode = rope.transformation_mat

        start_pos_tt_decode = ttnn.as_tensor(
            torch.full((bsz_per_submesh,), start_pos_decode),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
        )

        hidden_states_decode_reshaped = hidden_states_decode.reshape((1, 1, bsz_per_submesh, config.hidden_size))
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
            save_file_name=f"submesh_{i}"
        )
        output_tt_decode = ttnn.reshape(output_tt_decode, (bsz_per_submesh, 1, config.hidden_size))
        output_tt_decode = ttnn.to_layout(output_tt_decode, ttnn.ROW_MAJOR_LAYOUT)

        tt_output_decode = ttnn.to_torch(
            output_tt_decode,
            dtype=torch.bfloat16,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )[:bsz_per_submesh, :, :]

        print("Comparing decode outputs...")
        compare_tensor_pcc(ref_output_full[(bsz_per_submesh) * i:(bsz_per_submesh) * (i + 1), seq_len:seq_len + 1, :], tt_output_decode)

    print("\n=== Test completed successfully! ===")

def test_attn_compare():
    layers = [
        "init_k_cache",
        "init_v_cache",
        "init_hidden_states",
        "qkv_states",
        "query_states_create_heads",
        "key_states_create_heads",
        "value_states_create_heads",
        "query_states_after_rmsnorm",
        "key_states_after_rmsnorm",
        "query_states_pre_rope",
        "rot_mat_cos",
        "rot_mat_sin",
        "trans_mat",
        "query_states",
        "key_states",
        "before_update_k_cache",
        "before_update_v_cache",
        "query_states_before_update",
        "after_update_k_cache",
        "after_update_v_cache",
        "query_states_before_sdpa",
        "attn_output",
    ]
    for layer in layers:
        fullmesh_tensor = torch.load(f"activations/fullmesh_{layer}.pt")
        fullmesh_tensor = torch.chunk(fullmesh_tensor, 32, dim=0)

        submesh_0_tensor = torch.load(f"activations/submesh_0_{layer}.pt")
        submesh_0_tensor = torch.chunk(submesh_0_tensor, 16, dim=0)
        submesh_1_tensor = torch.load(f"activations/submesh_1_{layer}.pt")
        submesh_1_tensor = torch.chunk(submesh_1_tensor, 16, dim=0)

        print(f"Comparing layer: {layer}")
        for i in range(32):
            if i < 16:
                submesh_tensor = submesh_0_tensor[i]
            else:
                submesh_tensor = submesh_1_tensor[i - 16]

            if not torch.allclose(fullmesh_tensor[i], submesh_tensor, atol=1e-3):
                diff = torch.abs(fullmesh_tensor[i] - submesh_tensor)
                max_diff = torch.max(diff).item()
                print(f"  Mismatch in chunk {i}: max difference = {max_diff}")
                break

if __name__ == "__main__":
    pytest.main([__file__])