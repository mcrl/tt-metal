import torch
from torch import nn
from typing import Tuple

import ttnn
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.reference.rope import precompute_freqs_cis
from models.demos.qwen3.tt.rope import precompute_freqs_cis as precompute_freqs_cis_tt
from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm
from models.demos.qwen3.tt.attention import Qwen3MoeAttention
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.mesh_device = mesh_device
        self.is_tt_setup = False

        self.self_attn = Qwen3MoeAttention(config, layer_idx, mesh_device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)

    def setup_tt(self):
        if self.is_tt_setup:
            return
        self.self_attn.setup_tt()
        self.mlp.setup_tt()

        self.input_layernorm_weight_tt = ttnn.from_torch(self.input_layernorm.weight,
                                                         device=self.mesh_device,
                                                         mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                         dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.input_layernorm_weight_tt = ttnn.to_layout(self.input_layernorm_weight_tt, ttnn.TILE_LAYOUT,
                                                        dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.post_attention_layernorm_weight_tt = ttnn.from_torch(self.post_attention_layernorm.weight,
                                                                  device=self.mesh_device,
                                                                  mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                                  dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.post_attention_layernorm_weight_tt = ttnn.to_layout(self.post_attention_layernorm_weight_tt, ttnn.TILE_LAYOUT,
                                                                 dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.is_tt_setup = True

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor,
        mode: InferenceMode = InferenceMode.PREFILL
    ) -> ttnn.Tensor:

        hidden_states_0 = hidden_states
        attn_input = ttnn.rms_norm(hidden_states_0, epsilon=self.config.rms_norm_eps, weight=self.input_layernorm_weight_tt)
        attn_result = self.self_attn(
            hidden_states=attn_input,
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
            attention_mask=attention_mask,
            mode=mode,
        )

        hidden_states_1 = ttnn.add(attn_result, hidden_states_0)
        mlp_input = ttnn.rms_norm(hidden_states_1, epsilon=self.config.rms_norm_eps, weight=self.post_attention_layernorm_weight_tt)
        mlp_result = self.mlp(mlp_input)
        output = ttnn.add(hidden_states_1, mlp_result)

        return output


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.is_tt_setup = False

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx, mesh_device) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        position_embeddings = precompute_freqs_cis(config)
        self.position_embeddings_tt = precompute_freqs_cis_tt(config)
        self.register_buffer("position_embeddings", position_embeddings, persistent=False)

        assert config.sliding_window is None

    def setup_tt(self):
        if self.is_tt_setup:
            return
        for layer in self.layers:
            layer.setup_tt()
        self.embedding_weight_tt = ttnn.from_torch(self.embed_tokens.weight,
                                                   device=self.mesh_device,
                                                   mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                   dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.embedding_weight_tt = ttnn.to_layout(self.embedding_weight_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.norm_weight_tt = ttnn.from_torch(self.norm.weight,
                                              device=self.mesh_device,
                                              mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                              dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.norm_weight_tt = ttnn.to_layout(self.norm_weight_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.lm_head_weight_tt = ttnn.from_torch(self.lm_head.weight.transpose(0, 1),
                                                 device=self.mesh_device,
                                                 mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                                 dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.lm_head_weight_tt = ttnn.to_layout(self.lm_head_weight_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.is_tt_setup = True

    def forward(self, input_ids: torch.LongTensor, start_pos: int = 0, mode: InferenceMode = InferenceMode.PREFILL) -> torch.Tensor:
        # Normalize mode to InferenceMode enum (caller may pass a string)
        if isinstance(mode, str):
            mode = InferenceMode(mode)

        batch_size, sequence_length = input_ids.shape

        # CPU rope buffer exists for reference/compat but isn't used below

        pos_embs_cos = self.position_embeddings_tt[0][start_pos: start_pos + sequence_length]
        pos_embs_sin = self.position_embeddings_tt[1][start_pos: start_pos + sequence_length]

        cos_tt = ttnn.from_torch(pos_embs_cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
        sin_tt = ttnn.from_torch(pos_embs_sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))

        attention_mask = (
            torch.full(size=(1, 1, sequence_length, start_pos + sequence_length), fill_value=True, dtype=torch.bool)
            .triu_(diagonal=start_pos + 1)
            .logical_not_()
        )

        attention_mask_tt = ttnn.from_torch(
            attention_mask.repeat(batch_size, self.mesh_device.shape[1], 1, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        input_ids_tt = ttnn.from_torch(input_ids,
                                       device=self.mesh_device,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                       dtype=ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_ids_tt = ttnn.to_layout(input_ids_tt, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states_tt = ttnn.embedding(input_ids_tt, self.embedding_weight_tt, dtype=ttnn.bfloat16)

        for layer_idx, decoder_layer in enumerate(self.layers):
            position_embeddings = cos_tt, sin_tt
            hidden_states_tt = decoder_layer(
                hidden_states=hidden_states_tt,
                start_pos=start_pos,
                position_embeddings=position_embeddings,
                position_embeddings_tt=position_embeddings_tt,
                attention_mask=attention_mask,
                mode=mode,
            )

        hidden_states_tt = ttnn.rms_norm(hidden_states_tt, epsilon=self.config.rms_norm_eps, weight=self.norm_weight_tt)
        logits_tt = ttnn.linear(hidden_states_tt, self.lm_head_weight_tt, dtype=ttnn.bfloat16)
        logits_tt_cpu = ttnn.to_torch(logits_tt, dtype=self.config.dtype, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))

        return logits_tt_cpu


__all__ = ["Qwen3MoeModel"]
