import torch
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
from models.demos.qwen3.tt.model_cache import ttnn_model_cache_path, get_model_cache_prefix
from models.tt_transformers.tt.ccl import TT_CCL

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
        
        # Load layer norm weights if using lazy loading
        from models.demos.qwen3.common.lazy_loader import get_lazy_loader, load_parameter_for_module, is_meta_tensor
        lazy_loader = get_lazy_loader()
        
        if lazy_loader is not None:
            param_prefix = f"layers.{self.layer_idx}"
            
            # Load layernorm weights before calling their setup_tt()
            if is_meta_tensor(self.input_layernorm.weight):
                load_parameter_for_module(self.input_layernorm, "weight", f"{param_prefix}.input_layernorm.weight")
            if is_meta_tensor(self.post_attention_layernorm.weight):
                load_parameter_for_module(self.post_attention_layernorm, "weight", f"{param_prefix}.post_attention_layernorm.weight")
        
        self.self_attn.setup_tt()
        self.mlp.setup_tt()
        self.input_layernorm.setup_tt()
        self.post_attention_layernorm.setup_tt()
        
        self.is_tt_setup = True
        
        # Force garbage collection after layer setup to free memory
        if lazy_loader is not None:
            import gc
            gc.collect()

    def forward_prefill(
        self, hidden_states: ttnn.Tensor, start_pos: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> ttnn.Tensor:
        hidden_states_0 = hidden_states

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            attn_input = self.input_layernorm(hidden_states_0, mode=InferenceMode.PREFILL)

        attn_result = self.self_attn(
            hidden_states=attn_input,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos,
            page_table=page_table,
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
        self, hidden_states: ttnn.Tensor, start_pos: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> ttnn.Tensor:
        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            attn_input = self.input_layernorm(hidden_states, mode=InferenceMode.DECODE)

        attn_result = self.self_attn(
            hidden_states=attn_input,
            rot_mats=rot_mats,
            trans_mat=trans_mat,
            start_pos=start_pos,
            page_table=page_table,
            mode=InferenceMode.DECODE
        )

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            hidden_states = ttnn.add(attn_result, hidden_states)

        with Profiler().trace_with_timer("rmsnorm", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            mlp_input = self.post_attention_layernorm(hidden_states, mode=InferenceMode.DECODE)
        mlp_result = self.mlp(mlp_input, mode=InferenceMode.DECODE)

        with Profiler().trace_with_timer("add", level=4, args={"class": "Qwen3MoeDecoderLayer"}):
            output = ttnn.add(hidden_states, mlp_result)

        return output

    @profile_trace("Qwen3MoeDecoderLayer", level=2)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: ttnn.Tensor,
        mode: InferenceMode,
        rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor],
        trans_mat: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if mode == InferenceMode.PREFILL:
            return self.forward_prefill(hidden_states, start_pos, rot_mats, trans_mat, page_table)
        elif mode == InferenceMode.DECODE:
            return self.forward_decode(hidden_states, start_pos, rot_mats, trans_mat, page_table)
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

        # Create cache prefix for this mesh configuration
        cache_prefix = get_model_cache_prefix(self.mesh_device)

        # Load embeddings and model-level weights if needed
        from models.demos.qwen3.common.lazy_loader import get_lazy_loader, load_parameter_for_module, is_meta_tensor, get_memory_usage_gb
        lazy_loader = get_lazy_loader()

        if lazy_loader is not None:
            print(f"Initial memory usage: {get_memory_usage_gb():.2f} GB")

            if is_meta_tensor(self.embed_tokens.weight):
                load_parameter_for_module(self.embed_tokens, "weight", "embed_tokens.weight")
            if is_meta_tensor(self.norm.weight):
                load_parameter_for_module(self.norm, "weight", "norm.weight")
            if is_meta_tensor(self.lm_head.weight):
                load_parameter_for_module(self.lm_head, "weight", "lm_head.weight")

        # Setup layers (will trigger lazy loading per layer)
        for layer_idx, layer in enumerate(self.layers):
            if lazy_loader is not None:
                mem_before = get_memory_usage_gb()

            print(f"Setting up layer {layer_idx}/{len(self.layers)}...")
            layer.setup_tt()

            if lazy_loader is not None:
                mem_after = get_memory_usage_gb()
                print(f"  Memory: {mem_after:.2f} GB (delta: {mem_after - mem_before:+.2f} GB)")

        # Upload embeddings to device
        self.embedding_weight = ttnn.as_tensor(
            self.embed_tokens.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=ttnn_model_cache_path(f"{cache_prefix}embedding_weight"),
        )

        self.norm.setup_tt()

        self.lm_head_weight = ttnn.as_tensor(
            self.lm_head.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=ttnn_model_cache_path(f"{cache_prefix}lm_head_weight"),
        )

        self.ccl = TT_CCL(self.mesh_device)

        self.is_tt_setup = True
        
        # Free CPU RAM after setup (if using lazy loader)
        if lazy_loader is not None:
            from models.demos.qwen3.common.lazy_loader import clear_module_weights
            clear_module_weights(self)
            print(f"All model weights cleared from CPU RAM")
            print(f"Final memory usage: {get_memory_usage_gb():.2f} GB")

    @profile_trace("Qwen3MoeModel", level=1)
    def forward(self, ids: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor,
                start_pos: ttnn.Tensor, mode: InferenceMode = InferenceMode.PREFILL) -> ttnn.Tensor:
        if isinstance(mode, str):
            mode = InferenceMode(mode)

        batch_size, sequence_length = ids.shape

        with Profiler().trace_with_timer("embedding", level=4, args={"class": "Qwen3MoeModel"}):
            hidden_states = ttnn.embedding(ids, self.embedding_weight, dtype=ttnn.bfloat16)
            hidden_states = ttnn.unsqueeze(hidden_states, 0)

            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                3,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
            )
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                3,
                cluster_axis=0,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
            )
            hidden_states = ttnn.squeeze(hidden_states, 0)

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
                page_table=page_table,
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
