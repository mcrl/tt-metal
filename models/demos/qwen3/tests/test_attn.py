import pytest
import torch
import json
import os
from torch import nn

import ttnn
import tt_lock

from transformers import AutoConfig
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode

# from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeDecoderLayer
# from models.demos.qwen3.reference.rope import precompute_freqs_cis

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

from models.demos.qwen3.tt.attention import Qwen3MoeAttention
from models.demos.qwen3.tt.rope import precompute_freqs_cis_v2

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

    rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    return layer, rotary_emb


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (8, 64),
        (8, 1),
        # (2, 64),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_tt_attn_matches_reference(batch_size, seq_len, mesh_device):
    """Compare TT Attention implementation with PyTorch reference."""
    torch.manual_seed(0)

    set_and_get_device_cache(mesh_device)

    ref_layer, ref_rope = load_reference_layer(seq_len=seq_len)
    ref_attention = ref_layer.self_attn

    config = create_test_config()
    layer_idx = 0
    start_pos = 0
    tt_attention = Qwen3MoeAttention(config, layer_idx, mesh_device)

    tt_attention.q_proj.weight.data = ref_attention.q_proj.weight.data.clone()
    tt_attention.k_proj.weight.data = ref_attention.k_proj.weight.data.clone()
    tt_attention.v_proj.weight.data = ref_attention.v_proj.weight.data.clone()
    tt_attention.o_proj.weight.data = ref_attention.o_proj.weight.data.clone()
    tt_attention.q_norm.weight.data = ref_attention.q_norm.weight.data.clone()
    tt_attention.k_norm.weight.data = ref_attention.k_norm.weight.data.clone()
    tt_attention.setup_tt()

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    attention_mask = (
        torch.full(size=(1, 1, seq_len, start_pos + seq_len), fill_value=True, dtype=torch.bool)
        .triu_(diagonal=start_pos + 1)
        .logical_not_()
    )
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long).unsqueeze(0)
    ref_position_embeddings = ref_rope(hidden_states, position_ids)
    ref_output = ref_attention(
        hidden_states, 
        position_embeddings=ref_position_embeddings, 
        attention_mask=attention_mask
    )[0]

    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    position_embeddings_tt = precompute_freqs_cis_v2(Qwen3MoeConfig(head_dim=config.head_dim, max_seq_len=seq_len))

    pos_embs_cos = position_embeddings_tt[0][start_pos: start_pos + seq_len]
    pos_embs_sin = position_embeddings_tt[1][start_pos: start_pos + seq_len]

    cos_tt = ttnn.from_torch(
        pos_embs_cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        pos_embs_sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    position_embeddings_tt = cos_tt, sin_tt

    output_tt = tt_attention(
        hidden_states=hidden_states_tt,
        start_pos=0,
        position_embeddings=position_embeddings_tt,
        mode=InferenceMode.PREFILL,
    )
    output_tt = ttnn.to_layout(output_tt, ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.to_torch(
        output_tt, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[:batch_size, :, :]

    compare_tensor_pcc(ref_output, tt_output)


if __name__ == "__main__":
    pytest.main([__file__])
