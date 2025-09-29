from torch import nn
from typing import Tuple

import ttnn
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.tt.rope import (
    precompute_freqs_cis_v2 as precompute_freqs_cis_tt_v2,
)

from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm
from models.demos.qwen3.tt.attention import Qwen3MoeAttention
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock
from models.demos.qwen3.utils.profiler import Profiler, profile_trace
from models.demos.qwen3.tt.model_cache import ttnn_model_cache_path


class Qwen3MoeDecoderLayer(nn.Module):
    @profile_trace("create-layer", level=2, args={"class": "Qwen3MoeDecoderLayer"})
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

    @profile_trace("setup-tt", level=2, args={"class": "Qwen3MoeDecoderLayer"})
    def setup_tt(self):
        if self.is_tt_setup:
            return
        self.self_attn.setup_tt()
        self.mlp.setup_tt()
        self.input_layernorm.setup_tt()
        self.post_attention_layernorm.setup_tt()
        self.is_tt_setup = True

    def forward_prefill(
        self, hidden_states: ttnn.Tensor, start_pos: int, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        hidden_states_0 = hidden_states
        """Hidden states: [B, S, H]"""

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            attn_input = self.input_layernorm(hidden_states_0, mode=InferenceMode.PREFILL)

        attn_result = self.self_attn(
            hidden_states=attn_input,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos,
            mode=InferenceMode.PREFILL
        )

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            hidden_states_1 = ttnn.add(attn_result, hidden_states_0)

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            mlp_input = self.post_attention_layernorm(hidden_states_1, mode=InferenceMode.PREFILL)
        mlp_result = self.mlp(mlp_input, mode=InferenceMode.PREFILL)

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            output = ttnn.add(hidden_states_1, mlp_result)

        return output

    def forward_decode(
        self, hidden_states: ttnn.Tensor, start_pos: int, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Hidden states: [1, 1, B, H]"""

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            attn_input = self.input_layernorm(hidden_states, mode=InferenceMode.DECODE)

        attn_result = self.self_attn(
            hidden_states=attn_input,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos,
            mode=InferenceMode.DECODE
        )
        """attn result: [1, 1, B, H]"""

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            hidden_states = ttnn.add(attn_result, hidden_states)
        """Hidden states: [1, 1, B, H]"""

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            mlp_input = self.post_attention_layernorm(hidden_states, mode=InferenceMode.DECODE)
        mlp_result = self.mlp(mlp_input, mode=InferenceMode.DECODE)
        """mlp_result: [1, 1, B, H]"""

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            output = ttnn.add(hidden_states, mlp_result)
        """output: [1, 1, B, H]"""

        return output

    @profile_trace("Qwen3MoeDecoderLayer", level=2)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int,
        mode: InferenceMode,
        rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor],
        trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        if mode == InferenceMode.PREFILL:
            return self.forward_prefill(hidden_states, start_pos, rot_mats, trans_mat)
        elif mode == InferenceMode.DECODE:
            return self.forward_decode(hidden_states, start_pos, rot_mats, trans_mat)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class Qwen3MoeModel(nn.Module):
    @profile_trace("create-model", level=1, args={"class": "Qwen3MoeModel"})
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

        assert config.sliding_window is None

    @profile_trace("setup-tt", level=1, args={"class": "Qwen3MoeModel"})
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
            cache_file_name=ttnn_model_cache_path("embedding_weight"),
        )
        self.norm.setup_tt()
        self.lm_head_weight = ttnn.as_tensor(
            self.lm_head.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=ttnn_model_cache_path("lm_head_weight"),
        )

        self.is_tt_setup = True

    @profile_trace("Qwen3MoeModel", level=1)
    def forward(self, ids: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor,
                start_pos: int = 0, mode: InferenceMode = InferenceMode.PREFILL) -> ttnn.Tensor:
        if isinstance(mode, str):
            mode = InferenceMode(mode)

        batch_size, sequence_length = ids.shape

        with Profiler().trace_with_timer("embedding", level=4, args={"class": "Qwen3MoeModel"}):
            hidden_states = ttnn.embedding(ids, self.embedding_weight, dtype=ttnn.bfloat16)

        if mode == InferenceMode.PREFILL:
            pass
        elif mode == InferenceMode.DECODE:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, batch_size, self.config.hidden_size))

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                start_pos=start_pos,
                mode=mode,
                rot_mats=rot_mats,
                trans_mat=trans_mat,
            )

        if mode == InferenceMode.PREFILL:
            pass
        elif mode == InferenceMode.DECODE:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, sequence_length, self.config.hidden_size))

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeModel"}):
            hidden_states = self.norm(hidden_states, mode=InferenceMode.PREFILL)

        with Profiler().trace_with_timer("LMhead", level=4, args={"class": "Qwen3MoeModel"}):
            logits = ttnn.linear(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat16)

        return logits


__all__ = ["Qwen3MoeModel"]
