import torch
from torch import nn
from typing import Tuple
from pathlib import Path

import ttnn
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.reference.rope import precompute_freqs_cis
from models.demos.qwen3.tt.rope import (
    precompute_freqs_cis as precompute_freqs_cis_tt,
    precompute_freqs_cis_v2 as precompute_freqs_cis_tt_v2,
)

from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm
from models.demos.qwen3.tt.attention import Qwen3MoeAttention
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock
from models.demos.qwen3.utils.timer import profile_time, start_timer, stop_timer
from models.demos.qwen3.utils.profiler import Profiler, profile_trace


class Qwen3MoeDecoderLayer(nn.Module):
    @profile_trace("create-layer", level=1, args={"class": "Qwen3MoeDecoderLayer"})
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
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device
        )

    @profile_trace("setup-tt", level=1, args={"class": "Qwen3MoeDecoderLayer"})
    def setup_tt(self):
        if self.is_tt_setup:
            return
        self.self_attn.setup_tt()
        self.mlp.setup_tt()

        self.input_layernorm_weight = ttnn.as_tensor(
            self.input_layernorm.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_input_ln_weight",
        )
        self.post_attention_layernorm_weight = ttnn.as_tensor(
            self.post_attention_layernorm.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_post_attn_ln_weight",
        )
        self.is_tt_setup = True

    @profile_trace("Qwen3MoeDecoderLayer", level=1)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor,
        mode: InferenceMode = InferenceMode.PREFILL,
    ) -> ttnn.Tensor:

        hidden_states_0 = hidden_states

        with Profiler().trace_with_timer("rmsnorm", level=3, args={"class": "Qwen3MoeDecoderLayer"}):
            attn_input = ttnn.rms_norm(
                hidden_states_0, epsilon=self.config.rms_norm_eps, weight=self.input_layernorm_weight
            )
        attn_result = self.self_attn(
            hidden_states=attn_input,
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
            attention_mask=attention_mask,
            mode=mode,
        )

        with Profiler().trace_with_timer("add", level=3, args={"class": "Qwen3MoeDecoderLayer"}):
            hidden_states_1 = ttnn.add(attn_result, hidden_states_0)

        with Profiler().trace_with_timer("rmsnorm", level=3, args={"class": "Qwen3MoeDecoderLayer"}):
            mlp_input = ttnn.rms_norm(
                hidden_states_1, epsilon=self.config.rms_norm_eps, weight=self.post_attention_layernorm_weight
            )
        mlp_result = self.mlp(mlp_input)

        with Profiler().trace_with_timer("add", level=3, args={"class": "Qwen3MoeDecoderLayer"}):
            output = ttnn.add(hidden_states_1, mlp_result)

        return output


class Qwen3MoeModel(nn.Module):
    @profile_trace("create-model", level=0, args={"class": "Qwen3MoeModel"})
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

        self.position_embeddings_v2_cpu = precompute_freqs_cis_tt_v2(config)

        assert config.sliding_window is None

    @profile_trace("setup-tt", level=0, args={"class": "Qwen3MoeModel"})
    def setup_tt(self):
        if self.is_tt_setup:
            return
        for layer in self.layers:
            layer.setup_tt()
        self.embedding_weight = ttnn.as_tensor(
            self.embed_tokens.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights/embedding_weight",
        )
        self.norm_weight = ttnn.as_tensor(
            self.norm.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights/final_norm_weight",
        )
        self.lm_head_weight = ttnn.as_tensor(
            self.lm_head.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights/lm_head_weight",
        )
        self.is_tt_setup = True

    @profile_trace("Qwen3MoeModel", level=0)
    def forward(
        self, input_ids: torch.LongTensor, start_pos: int = 0, mode: InferenceMode = InferenceMode.PREFILL
    ) -> torch.Tensor:
        # Normalize mode to InferenceMode enum (caller may pass a string)
        if isinstance(mode, str):
            mode = InferenceMode(mode)

        batch_size, sequence_length = input_ids.shape
        input_ids_cpu = input_ids

        pos_embs_cos_cpu = self.position_embeddings_v2_cpu[0][start_pos: start_pos + sequence_length]
        pos_embs_sin_cpu = self.position_embeddings_v2_cpu[1][start_pos: start_pos + sequence_length]

        with Profiler().trace_with_timer("input-transfer", level=3, args={"class": "Qwen3MoeModel"}):
            cos = ttnn.from_torch(
                pos_embs_cos_cpu,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            sin = ttnn.from_torch(
                pos_embs_sin_cpu,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            attention_mask_cpu = (
                torch.full(size=(1, 1, sequence_length, start_pos + sequence_length), fill_value=True, dtype=torch.bool)
                .triu_(diagonal=start_pos + 1)
                .logical_not_()
            )

            attention_mask = ttnn.from_torch(
                attention_mask_cpu.repeat(batch_size, self.mesh_device.shape[1], 1, 1),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ids = ttnn.from_torch(
                input_ids_cpu,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        with Profiler().trace_with_timer("embedding", level=3, args={"class": "Qwen3MoeModel"}):
            hidden_states = ttnn.embedding(ids, self.embedding_weight, dtype=ttnn.bfloat16)

        for layer_idx, decoder_layer in enumerate(self.layers):
            position_embeddings = cos, sin
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                start_pos=start_pos,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                mode=mode,
            )

        with Profiler().trace_with_timer("rmsnorm", level=3, args={"class": "Qwen3MoeModel"}):
            hidden_states = ttnn.rms_norm(
                hidden_states, epsilon=self.config.rms_norm_eps, weight=self.norm_weight
            )

        with Profiler().trace_with_timer("LMhead", level=3, args={"class": "Qwen3MoeModel"}):
            logits = ttnn.linear(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat16)

        with Profiler().trace_with_timer("output-transfer", level=3, args={"class": "Qwen3MoeModel"}):
            logits_cpu = ttnn.to_torch(
                logits, dtype=self.config.dtype, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2)
            )

        return logits_cpu


__all__ = ["Qwen3MoeModel"]
