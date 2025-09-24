from models.demos.qwen3.utils.profiler import profile_trace, Profiler
from models.tt_transformers.tt.common import get_rot_transformation_mat
import torch
from torch import nn
from typing import Tuple
from pathlib import Path

import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward
from models.demos.qwen3.tt.rope import apply_rotary_emb_v2
from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm


class Qwen3MoeAttention(nn.Module):
    @profile_trace("create-layer", level=3, args={"class": "Qwen3MoeAttention"})
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.is_tt_setup = False
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
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

        self.cache_shape = (
            config.max_batch_size,
            self.num_key_value_heads * self.KV_REPEAT_COEF // self.mesh_device.shape[1],
            config.max_seq_len,
            self.head_dim,
        )

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None

    @profile_trace("setup-tt", level=3, args={"class": "Qwen3MoeAttention"})
    def setup_tt(self):
        if self.is_tt_setup:
            return

        self.q_proj_weight = ttnn.as_tensor(
            self.q_proj.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_q_proj",
        )

        def reshape_weight(x, head_dim, repeats):
            x = x.transpose(0, 1)
            hidden_size, _ = x.shape
            x = x.view(hidden_size, -1, head_dim)
            x = x.repeat_interleave(repeats=repeats, dim=1)
            x = x.view(hidden_size, -1)
            return x.contiguous()

        self.cache_k = ttnn.zeros(
            self.cache_shape,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.cache_v = ttnn.zeros(
            self.cache_shape,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.k_proj_weight = ttnn.as_tensor(
            reshape_weight(self.k_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_k_proj",
        )
        self.v_proj_weight = ttnn.as_tensor(
            reshape_weight(self.v_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_v_proj",
        )

        self.q_norm.setup_tt()
        self.k_norm.setup_tt()

        self.o_proj_weight = ttnn.as_tensor(
            self.o_proj.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_o_proj",
        )
        self.trans_mat = ttnn.as_tensor(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_rot_trans_mat",
        )
        self.is_tt_setup = True

    @profile_trace("Qwen3MoeAttention", level=3)
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

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            query_states = ttnn.linear(
                hidden_states, self.q_proj_weight, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            key_states = ttnn.linear(
                hidden_states, self.k_proj_weight, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            value_states = ttnn.linear(
                hidden_states, self.v_proj_weight, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        with Profiler().trace_with_timer("qkv-proj-reshape", level=4):
            key_states = ttnn.reshape(key_states, hidden_shape, memory_config=ttnn.L1_MEMORY_CONFIG)
            query_states = ttnn.reshape(query_states, hidden_shape, memory_config=ttnn.L1_MEMORY_CONFIG)
            value_states = ttnn.reshape(value_states, hidden_shape, memory_config=ttnn.L1_MEMORY_CONFIG)

        with Profiler().trace_with_timer("rmsnorm", level=4):
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        with Profiler().trace_with_timer("rope", level=4):
            query_states, key_states = apply_rotary_emb_v2(
                query_states, key_states, position_embeddings, self.trans_mat
            )

        with Profiler().trace_with_timer("permute", level=4):
            value_states = ttnn.permute(value_states, dims=(0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Q, K, V: [B n S H]

        with Profiler().trace_with_timer("kv-cache-store", level=4):
            if mode == InferenceMode.PREFILL:
                for b in range(batch_size):
                    ttnn.kv_cache.fill_cache_for_user_(self.cache_k, key_states[b : b + 1], b)
                    ttnn.kv_cache.fill_cache_for_user_(self.cache_v, value_states[b : b + 1], b)
            elif mode == InferenceMode.DECODE:
                key_states = ttnn.permute(key_states, dims=(2, 1, 0, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
                value_states = ttnn.permute(value_states, dims=(2, 1, 0, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.kv_cache.update_cache_for_token_(self.cache_k, key_states, update_index=start_pos, batch_offset=0)
                ttnn.kv_cache.update_cache_for_token_(
                    self.cache_v, value_states, update_index=start_pos, batch_offset=0
                )

        with Profiler().trace_with_timer("kv-cache-load", level=4):
            start_index = (0, 0, 0, 0)
            end_index = (batch_size, self.kv_heads_per_device, start_pos + sequence_length, self.head_dim)

            ttnn.deallocate(key_states)
            ttnn.deallocate(value_states)
            key_states = ttnn.slice(
                self.cache_k, slice_start=start_index, slice_end=end_index, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            value_states = ttnn.slice(
                self.cache_v, slice_start=start_index, slice_end=end_index, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        attn_out = tt_sdpa_forward(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask if mode == InferenceMode.DECODE else None,
            dropout=0.0,
            scaling=self.scaling,
            mode=mode,
        )

        with Profiler().trace_with_timer("reshape", level=4):
            # [B, n, S, h] -> [B, S, n * h]
            attn_out = ttnn.transformer.concatenate_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        with Profiler().trace_with_timer("output-proj", level=4):
            linear_output = ttnn.linear(
                attn_out,
                self.o_proj_weight,
                transpose_a=False,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        with Profiler().trace_with_timer("all-reduce", level=4):
            B, S, H = linear_output.shape
            linear_output = ttnn.reshape(
                linear_output, shape=(1, 1, B * S * H // 256, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            linear_output = ttnn.experimental.all_reduce(
                linear_output,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
            )
            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.reshape(linear_output, shape=(B, S, H), memory_config=ttnn.L1_MEMORY_CONFIG)

        with Profiler().trace_with_timer("reshape", level=4):
            output = ttnn.reshape(
                linear_output,
                (batch_size, sequence_length, hidden_size),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        return output


__all__ = ["Qwen3MoeAttention"]
