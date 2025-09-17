import torch
from torch import nn
from typing import Tuple

import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward
from models.demos.qwen3.tt.rope import apply_rotary_emb as apply_rotary_emb_tt, apply_rotary_emb_v2 as apply_rotary_emb_tt_v2
from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm
from models.demos.qwen3.tt.timer import profile_time
from models.demos.qwen3.tt.timer import start_timer, stop_timer
from models.tt_transformers.tt.common import get_rot_transformation_mat


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.is_tt_setup = False
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = pow(self.head_dim, -0.5)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device)

        self.sliding_window = None

        self.KV_REPEAT_COEF = 2
        self.kv_heads_per_device = self.num_key_value_heads * self.KV_REPEAT_COEF // self.mesh_device.shape[1]

        cache_shape = (
            config.max_batch_size,
            self.num_key_value_heads * self.KV_REPEAT_COEF,
            config.max_seq_len,
            self.head_dim,
        )
        cache_k = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        cache_v = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None
        assert not config.attention_bias

    def setup_tt(self):
        if self.is_tt_setup:
            return
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        self.q_proj_weight = ttnn.from_torch(
            self.q_proj.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.q_proj_weight = ttnn.to_layout(
            self.q_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def reshape_weight(x, head_dim, repeats):
            x = x.transpose(0, 1)
            hidden_size, _ = x.shape
            x = x.view(hidden_size, -1, head_dim)
            x = x.repeat_interleave(repeats=repeats, dim=1)
            x = x.view(hidden_size, -1)
            return x.contiguous()

        self.cache_k_tt = ttnn.from_torch(
            self.cache_k,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.cache_k_tt = ttnn.to_layout(
            self.cache_k_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.cache_v_tt = ttnn.from_torch(
            self.cache_v,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.cache_v_tt = ttnn.to_layout(
            self.cache_v_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.k_proj_weight = ttnn.from_torch(
            reshape_weight(self.k_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_proj_weight = ttnn.to_layout(
            self.k_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.v_proj_weight = ttnn.from_torch(
            reshape_weight(self.v_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_proj_weight = ttnn.to_layout(
            self.v_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.query_rmsnorm_weight_tt = ttnn.from_torch(
            self.q_norm.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.query_rmsnorm_weight_tt = ttnn.to_layout(
            self.query_rmsnorm_weight_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.key_rmsnorm_weight_tt = ttnn.from_torch(
            self.k_norm.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.key_rmsnorm_weight_tt = ttnn.to_layout(
            self.key_rmsnorm_weight_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.o_proj_weight = ttnn.from_torch(
            self.o_proj.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.o_proj_weight = ttnn.to_layout(
            self.o_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.trans_mat_tt = ttnn.from_torch(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.is_tt_setup = True

    @profile_time()
    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor,
        mode: InferenceMode = InferenceMode.PREFILL,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_shape = (batch_size, sequence_length, -1, self.head_dim)

        start_timer("attention-qkv-projection", device=self.mesh_device)
        hidden_states_tt = hidden_states
        query_states_tt = ttnn.linear(hidden_states_tt, self.q_proj_weight, dtype=ttnn.bfloat16)
        query_states_tt = ttnn.reshape(query_states_tt, hidden_shape)
        query_states_tt = ttnn.rms_norm(
            query_states_tt, epsilon=self.config.rms_norm_eps, weight=self.query_rmsnorm_weight_tt
        )

        key_states_tt = ttnn.linear(hidden_states_tt, self.k_proj_weight, dtype=ttnn.bfloat16)
        key_states_tt = ttnn.reshape(key_states_tt, hidden_shape)

        key_states_tt = ttnn.rms_norm(
            key_states_tt, epsilon=self.config.rms_norm_eps, weight=self.key_rmsnorm_weight_tt
        )
        value_states_tt = ttnn.linear(hidden_states_tt, self.v_proj_weight, dtype=ttnn.bfloat16)
        value_states_tt = ttnn.reshape(value_states_tt, hidden_shape)
        stop_timer("attention-qkv-projection", device=self.mesh_device)

        start_timer("attention-rope", device=self.mesh_device)
        # query_states_tt, key_states_tt = apply_rotary_emb_tt(query_states_tt, key_states_tt, position_embeddings)
        query_states_tt, key_states_tt = apply_rotary_emb_tt_v2(query_states_tt, key_states_tt, position_embeddings, self.trans_mat_tt)
        stop_timer("attention-rope", device=self.mesh_device)

        start_timer("attention-post-rope-permute", device=self.mesh_device)
        query_states_tt = ttnn.permute(query_states_tt, dims=(0, 2, 1, 3))
        key_states_tt = ttnn.permute(key_states_tt, dims=(0, 2, 1, 3))
        value_states_tt = ttnn.permute(value_states_tt, dims=(0, 2, 1, 3))
        stop_timer("attention-post-rope-permute", device=self.mesh_device)

        start_timer("attention-kv-cache-store", device=self.mesh_device)
        if mode == InferenceMode.PREFILL:
            for b in range(batch_size):
                ttnn.kv_cache.fill_cache_for_user_(self.cache_k_tt, key_states_tt[b: b + 1], b)
                ttnn.kv_cache.fill_cache_for_user_(self.cache_v_tt, value_states_tt[b: b + 1], b)
        elif mode == InferenceMode.DECODE:
            for b in range(batch_size):
                ttnn.kv_cache.update_cache_for_token_(
                    self.cache_k_tt,
                    key_states_tt[b: b + 1],
                    start_pos + b * self.kv_heads_per_device * self.config.max_seq_len,
                )
                ttnn.kv_cache.update_cache_for_token_(
                    self.cache_v_tt,
                    value_states_tt[b: b + 1],
                    start_pos + b * self.kv_heads_per_device * self.config.max_seq_len,
                )
        stop_timer("attention-kv-cache-store", device=self.mesh_device)

        start_timer("attention-kv-cache-load", device=self.mesh_device)
        key_states_tt = self.cache_k_tt[:batch_size, :, : start_pos + sequence_length, :]
        value_states_tt = self.cache_v_tt[:batch_size, :, : start_pos + sequence_length, :]
        stop_timer("attention-kv-cache-load", device=self.mesh_device)

        start_timer("attention-sdpa", device=self.mesh_device)
        tt_out = tt_sdpa_forward(
            query_states_tt,
            key_states_tt,
            value_states_tt,
            attention_mask=attention_mask if mode == InferenceMode.DECODE else None,
            dropout=0.0,
            scaling=self.scaling,
            mode=mode,
        )
        stop_timer("attention-sdpa", device=self.mesh_device)

        tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[1], tt_out.shape[2] * tt_out.shape[3]])

        tt_out = ttnn.to_layout(tt_out, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        start_timer("attention-linear", device=self.mesh_device)
        linear_output_ttnn = ttnn.linear(
            tt_out,
            self.o_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stop_timer("attention-linear", device=self.mesh_device)

        start_timer("attention-allreduce", device=self.mesh_device)
        B, S, H = linear_output_ttnn.shape
        linear_output_ttnn = linear_output_ttnn.reshape(B, S, 1, H)
        linear_output_ttnn_reduced = ttnn.reduce_scatter(
            linear_output_ttnn,
            dim=-1,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.mesh_device)
        linear_output_ttnn_gathered = ttnn.all_gather(
            linear_output_ttnn_reduced,
            dim=-1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            mesh_device=self.mesh_device,
        )
        ttnn.synchronize_device(self.mesh_device)
        stop_timer("attention-allreduce", device=self.mesh_device)

        output = ttnn.reshape(linear_output_ttnn_gathered, (batch_size, sequence_length, hidden_size))
        return output


__all__ = ["Qwen3MoeAttention"]
