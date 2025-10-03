from typing import Tuple

import torch
from torch import nn

import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.utils.profiler import profile_trace, Profiler

from models.demos.qwen3.tt.model_cache import ttnn_model_cache_path
from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward
from models.demos.qwen3.tt.rms_norm import Qwen3MoeRMSNorm
from models.tt_transformers.tt.rope import RotarySetup

def reshape_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_half1, x_half2 = x.chunk(2, dim=-1)
    stacked = torch.stack([x_half1, x_half2], dim=-1)
    return stacked.flatten(start_dim=-2)

def reshape_weight(x, head_dim, repeats):
    x = x.transpose(0, 1)
    hidden_size, _ = x.shape
    x = x.view(hidden_size, -1, head_dim)
    x = x.repeat_interleave(repeats=repeats, dim=1)
    x = x.view(hidden_size, -1)
    return x.contiguous()

class Qwen3MoeAttention(nn.Module):
    @profile_trace("create-layer", level=3, args={"class": "Qwen3MoeAttention"})
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.is_tt_setup = False
        self.hidden_size = config.hidden_size
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

        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, interleaved=True)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device, interleaved=True)

        self.sliding_window = None

        self.q_heads_per_device = self.num_attention_heads // self.mesh_device.shape[1]

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

        self.ccl = CCL1D(self.mesh_device)

        weight_shape = (self.hidden_size, -1)

        q_weight = self.q_proj.weight.transpose(0, 1)
        q_weight = q_weight.reshape(self.hidden_size, -1, self.head_dim)
        q_weight = reshape_to_interleaved(q_weight).reshape(weight_shape)

        k_weight = reshape_weight(self.k_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF)
        k_weight = k_weight.reshape(self.hidden_size, -1, self.head_dim)
        k_weight = reshape_to_interleaved(k_weight).reshape(weight_shape)

        v_weight = reshape_weight(self.v_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF).reshape(weight_shape)
        v_weight = v_weight.reshape(self.hidden_size, -1, self.head_dim)
        v_weight = reshape_to_interleaved(v_weight).reshape(weight_shape)

        qkv_list = []
        wq = torch.chunk(q_weight, self.mesh_device.shape[1], dim=1)
        wk = torch.chunk(k_weight, self.mesh_device.shape[1], dim=1)
        wv = torch.chunk(v_weight, self.mesh_device.shape[1], dim=1)

        for i in range(self.mesh_device.shape[1]):
            qkv_list.append(torch.cat([wq[i], wk[i], wv[i]], dim=1))

        self.qkv_proj_weight = ttnn.as_tensor(
            torch.cat(qkv_list, dim=1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=ttnn_model_cache_path(f"decoder_{self.layer_idx}_qkv_proj"),
        )

        self.q_norm.setup_tt()
        self.k_norm.setup_tt()

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

        o_weight = self.o_proj.weight
        o_weight = o_weight.reshape(self.hidden_size, -1, self.head_dim)
        o_weight = reshape_to_interleaved(o_weight).reshape(weight_shape)

        self.o_proj_weight = ttnn.as_tensor(
            o_weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=ttnn_model_cache_path(f"decoder_{self.layer_idx}_o_proj"),
        )
        self.is_tt_setup = True

    def forward_prefill(
        self, hidden_states: ttnn.Tensor, start_pos: int, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Prefill starts. Hidden stats: [1, B=1, S, H]"""

        batch_size, sequence_length, hidden_size = hidden_states.shape
        mem_cfg = ttnn.L1_MEMORY_CONFIG if sequence_length == 1 else ttnn.DRAM_MEMORY_CONFIG
        hidden_shape = (batch_size, 1, sequence_length, -1)

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            qkv_states = ttnn.linear(hidden_states, self.qkv_proj_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)
            # ttnn.deallocate(hidden_states)
            qkv_states = ttnn.view(qkv_states, hidden_shape)

        with Profiler().trace_with_timer("qkv-split", level=4):
            query_states_pre_rot, key_states_pre_rot, value_states = ttnn.experimental.nlp_create_qkv_heads(
                qkv_states,
                num_heads=self.q_heads_per_device,
                num_kv_heads=self.kv_heads_per_device,
                transpose_k_heads=False,
                memory_config=mem_cfg
            )
            ttnn.deallocate(qkv_states)
            # B n S H

        with Profiler().trace_with_timer("rmsnorm", level=4):
            query_states_pre_rot = self.q_norm(query_states_pre_rot, mode=InferenceMode.PREFILL)
            key_states_pre_rot = self.k_norm(key_states_pre_rot, mode=InferenceMode.PREFILL)

        with Profiler().trace_with_timer("rope", level=4):
            query_states = ttnn.experimental.rotary_embedding_llama(
                query_states_pre_rot,
                rot_mats[0],
                rot_mats[1],
                trans_mat,
                is_decode_mode=False
            )
            ttnn.deallocate(query_states_pre_rot)
            key_states = ttnn.experimental.rotary_embedding_llama(
                key_states_pre_rot,
                rot_mats[0],
                rot_mats[1],
                trans_mat,
                is_decode_mode=False
            )
            ttnn.deallocate(key_states_pre_rot)

        # Q, K, V: [B n S H]
        with Profiler().trace_with_timer("kv-cache-store", level=4):
            for b in range(batch_size):
                ttnn.fill_cache(self.cache_k, key_states[b: b + 1], b)
                ttnn.fill_cache(self.cache_v, value_states[b: b + 1], b)

        with Profiler().trace_with_timer("kv-cache-load", level=4):
            start_index = (0, 0, 0, 0)
            end_index = (batch_size, self.kv_heads_per_device, start_pos + sequence_length, self.head_dim)

            ttnn.deallocate(key_states)
            ttnn.deallocate(value_states)
            key_states = ttnn.slice(self.cache_k, slice_start=start_index, slice_end=end_index, memory_config=mem_cfg)
            value_states = ttnn.slice(self.cache_v, slice_start=start_index, slice_end=end_index, memory_config=mem_cfg)

        attn_out = tt_sdpa_forward(
            query_states,
            key_states,
            value_states,
            dropout=0.0,
            scaling=self.scaling,
            mode=InferenceMode.PREFILL,
        )
        ttnn.deallocate(query_states)
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        with Profiler().trace_with_timer("reshape", level=4):
            # [B, n, S, h] -> [B, S, n * h]
            attn_out_cat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=mem_cfg)
            ttnn.deallocate(attn_out)

        with Profiler().trace_with_timer("output-proj", level=4):
            linear_output = ttnn.linear(
                attn_out_cat,
                self.o_proj_weight,
                transpose_a=False,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_out_cat)

        with Profiler().trace_with_timer("all-reduce", level=4):
            B, _, S, H = linear_output.shape
            linear_output = ttnn.reshape(
                linear_output, shape=(1, 1, B * S * H // 256, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            linear_output = ttnn.experimental.all_reduce_async(
                linear_output,
                math_op=ttnn.ReduceType.Sum,
                memory_config=mem_cfg,
                topology=ttnn.Topology.Linear,
                from_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                to_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                gather_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                num_links=1,
            )
            ttnn.synchronize_device(self.mesh_device)

            linear_output = ttnn.reshape(linear_output, shape=(B, S, H), memory_config=mem_cfg)

        return linear_output

    def forward_decode(
        self, hidden_states: ttnn.Tensor, start_pos: int, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        """Hidden state: [1, S=1, B, H]"""
        _, sequence_length, batch_size, hidden_size = hidden_states.shape

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            qkv_states = ttnn.linear(hidden_states, self.qkv_proj_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)
            # ttnn.deallocate(hidden_states)
        """ QKV: [1, 1, B, H] """

        with Profiler().trace_with_timer("qkv-split", level=4):
            query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_states,
                num_heads=self.q_heads_per_device,
                num_kv_heads=self.kv_heads_per_device,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
        """ QKV: [S=1, B, n, H] """

        with Profiler().trace_with_timer("rmsnorm", level=4):
            query_states = self.q_norm(query_states, mode=InferenceMode.DECODE)
            key_states = self.k_norm(key_states, mode=InferenceMode.DECODE)
        """ QKV: [1, B, n, H] """

        with Profiler().trace_with_timer("rope", level=4):
            query_states = ttnn.experimental.rotary_embedding_llama(
                query_states, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )
            key_states = ttnn.experimental.rotary_embedding_llama(
                key_states, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )

            query_states = ttnn.sharded_to_interleaved(query_states, memory_config=mem_cfg)
            key_states = ttnn.sharded_to_interleaved(key_states, memory_config=mem_cfg)
            value_states = ttnn.sharded_to_interleaved(value_states, memory_config=mem_cfg)
        """ Q: [S=1, B, n, H], K: [S=1, B, n, H], V: [S=1, B, n, H] """

        with Profiler().trace_with_timer("kv-cache-store", level=4):
            key_states = ttnn.permute(key_states, dims=(0, 2, 1, 3), memory_config=mem_cfg)
            value_states = ttnn.permute(value_states, dims=(0, 2, 1, 3), memory_config=mem_cfg)

            """ KV: [S=1, n, B, H] """
            ttnn.kv_cache.update_cache_for_token_(self.cache_k, key_states, update_index=start_pos, batch_offset=0)
            ttnn.kv_cache.update_cache_for_token_(self.cache_v, value_states, update_index=start_pos, batch_offset=0)

        with Profiler().trace_with_timer("kv-cache-load", level=4):
            start_index = (0, 0, 0, 0)
            end_index = (batch_size, self.kv_heads_per_device, start_pos + sequence_length, self.head_dim)

            ttnn.deallocate(key_states)
            ttnn.deallocate(value_states)
            key_states = ttnn.slice(self.cache_k, slice_start=start_index, slice_end=end_index, memory_config=mem_cfg)
            value_states = ttnn.slice(self.cache_v, slice_start=start_index, slice_end=end_index, memory_config=mem_cfg)
        """ Q: [S=1, B, n, H], KV: [B, n, S=1, H]"""

        attn_output = tt_sdpa_forward(
            query_states,
            key_states,
            value_states,
            cur_pos=[start_pos for _ in range(batch_size)],
            dropout=0.0,
            scaling=self.scaling,
            mode=InferenceMode.DECODE,
        )
        """ Output O: [S=1, B, n, H]"""

        # FIXME: Maybe we can try ttnn.experimental.nlp_concat_heads_decode to reshape.
        with Profiler().trace_with_timer("reshape", level=4):
            batch_size, hidden_dim_per_device = attn_output.shape[1], attn_output.shape[2] * attn_output.shape[3]
            attn_output = ttnn.reshape(
                attn_output, shape=(1, 1, batch_size, hidden_dim_per_device), memory_config=mem_cfg
            )
        """ O: [1, 1, B, H] """

        with Profiler().trace_with_timer("output-proj", level=4):
            linear_output = ttnn.linear(
                attn_output,
                self.o_proj_weight,
                transpose_a=False,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=mem_cfg,
            )
        """ O: [1, 1, B, H] """

        with Profiler().trace_with_timer("all-reduce", level=4):
            _, _, B, H = linear_output.shape
            linear_output = ttnn.reshape(linear_output, shape=(1, 1, B * H // 256, 256), memory_config=mem_cfg)
            linear_output = ttnn.experimental.all_reduce_async(
                linear_output,
                math_op=ttnn.ReduceType.Sum,
                memory_config=mem_cfg,
                topology=ttnn.Topology.Linear,
                from_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                to_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                gather_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                num_links=1,
            )
            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.reshape(linear_output, shape=(1, 1, B, H), memory_config=mem_cfg)
        """ O: [1, 1, B, H] """

        return linear_output

    @profile_trace("Qwen3MoeAttention", level=3)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor],
        trans_mat: ttnn.Tensor,
        start_pos: int,
        mode: InferenceMode = InferenceMode.PREFILL
    ) -> ttnn.Tensor:

        if mode == InferenceMode.PREFILL:
            return self.forward_prefill(hidden_states, start_pos, rot_mats, trans_mat)
        elif mode == InferenceMode.DECODE:
            return self.forward_decode(hidden_states, start_pos, rot_mats, trans_mat)
        else:
            raise ValueError(f"Unknown mode: {mode}")


__all__ = ["Qwen3MoeAttention"]
