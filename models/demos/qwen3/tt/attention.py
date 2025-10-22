from typing import Tuple
import math

import torch
from torch import nn

import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.utils.profiler import profile_trace, Profiler

from models.demos.qwen3.tt.model_cache import ttnn_model_cache_path
from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.tt_transformers.tt.ccl import TT_CCL

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
        self.num_devices = mesh_device.shape[0] * mesh_device.shape[1]
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

        self.q_heads_per_device = self.num_attention_heads // self.num_devices

        self.KV_REPEAT_COEF = 8
        self.kv_heads_per_device = self.num_key_value_heads * self.KV_REPEAT_COEF // self.num_devices

        self.cache_shape = (
            config.max_num_blocks,
            self.num_key_value_heads * self.KV_REPEAT_COEF // self.num_devices,
            config.block_size,
            self.head_dim,
        )

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None

    @profile_trace("setup-tt", level=3, args={"class": "Qwen3MoeAttention"})
    def setup_tt(self):
        if self.is_tt_setup:
            return

        self.ccl = TT_CCL(self.mesh_device) # CCL1D(self.mesh_device)

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
        wq = torch.chunk(q_weight, self.num_devices, dim=1)
        wk = torch.chunk(k_weight, self.num_devices, dim=1)
        wv = torch.chunk(v_weight, self.num_devices, dim=1)

        for i in range(self.num_devices):
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

        cache_k = torch.zeros(self.cache_shape)
        cache_v = torch.zeros(self.cache_shape)

        self.cache_k = ttnn.as_tensor(
            cache_k,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=ttnn_model_cache_path(f"kvcache_{self.cache_shape}")
        )
        self.cache_v = ttnn.as_tensor(
            cache_v,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=ttnn_model_cache_path(f"kvcache_{self.cache_shape}")
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
        self, hidden_states: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Prefill starts. Hidden stats: [1, B, S, H]"""

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
        with Profiler().trace_with_timer("fill-cache", level=4):
            for b in range(batch_size):
                ttnn.experimental.paged_fill_cache(self.cache_k, key_states[b:b + 1], page_table, batch_idx=b)
                ttnn.experimental.paged_fill_cache(self.cache_v, value_states[b:b + 1], page_table, batch_idx=b)

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
            linear_output = ttnn.view(linear_output, (1, 1, B * S, H))

            for cluster_axis in [1, 0]:
                linear_output = ttnn.experimental.reduce_scatter_minimal_async(
                    linear_output,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    cluster_axis=cluster_axis,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
                linear_output = ttnn.experimental.all_gather_async(
                    linear_output,
                    3,
                    cluster_axis=cluster_axis,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    num_links=1,
                )

            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.view(linear_output, (B, S, H))

        return linear_output

    def forward_decode(
        self, hidden_states: ttnn.Tensor, start_pos: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> ttnn.Tensor:
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        """Hidden state: [1, S=1, B, H]"""
        _, sequence_length, batch_size, hidden_size = hidden_states.shape

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            qkv_states = ttnn.linear(hidden_states, self.qkv_proj_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)
            # ttnn.deallocate(hidden_states)
        """ QKV: [1, 1, B, H] """

        with Profiler().trace_with_timer("qkv-split", level=4):
            query_states_pre_rot, key_states_pre_rot, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_states,
                num_heads=self.q_heads_per_device,
                num_kv_heads=self.kv_heads_per_device,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
            ttnn.deallocate(qkv_states)
        """ QKV: [S=1, B, n, H] """

        with Profiler().trace_with_timer("rmsnorm", level=4):
            query_states_pre_rot = self.q_norm(query_states_pre_rot, mode=InferenceMode.DECODE)
            key_states_pre_rot = self.k_norm(key_states_pre_rot, mode=InferenceMode.DECODE)
        """ QKV: [1, B, n, H] """

        with Profiler().trace_with_timer("rope", level=4):
            query_states = ttnn.experimental.rotary_embedding_llama(
                query_states_pre_rot, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )
            ttnn.deallocate(query_states_pre_rot)

            key_states = ttnn.experimental.rotary_embedding_llama(
                key_states_pre_rot, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )
            ttnn.deallocate(key_states_pre_rot)
        """ Q: [S=1, B, n, H], K: [S=1, B, n, H], V: [S=1, B, n, H] """

        ttnn.experimental.paged_update_cache(self.cache_k, key_states, update_idxs_tensor=start_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(self.cache_v, value_states, update_idxs_tensor=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.mesh_device) # If not, cause hanging

        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        attn_output = tt_sdpa_forward(
            query_states,
            self.cache_k,
            self.cache_v,
            cur_pos=start_pos,
            page_table=page_table,
            dropout=0.0,
            scaling=self.scaling,
            mode=InferenceMode.DECODE,
        )
        ttnn.deallocate(query_states)
        
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output, 
            num_heads=self.q_heads_per_device
        )
        ttnn.deallocate(attn_output)

        with Profiler().trace_with_timer("output-proj", level=4):
            linear_output = ttnn.linear(
                attn_output_cat,
                self.o_proj_weight,
                transpose_a=False,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=mem_cfg,
            )
            ttnn.deallocate(attn_output_cat)
        """ O: [1, 1, B, H] """

        with Profiler().trace_with_timer("all-reduce", level=4):
            _, _, B, H = linear_output.shape
            linear_output = ttnn.view(linear_output, (1, 1, B, H))

            for cluster_axis in [1, 0]:
                linear_output = ttnn.experimental.reduce_scatter_minimal_async(
                    linear_output,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    cluster_axis=cluster_axis,
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
                linear_output = ttnn.experimental.all_gather_async(
                    linear_output,
                    3,
                    cluster_axis=cluster_axis,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,
                    multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    num_links=1,
                )

            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.view(linear_output, shape=(1, 1, B, H))
        """ O: [1, 1, B, H] """

        return linear_output

    @profile_trace("Qwen3MoeAttention", level=3)
    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor],
        trans_mat: ttnn.Tensor,
        start_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        mode: InferenceMode = InferenceMode.PREFILL
    ) -> ttnn.Tensor:

        if mode == InferenceMode.PREFILL:
            return self.forward_prefill(hidden_states, rot_mats, trans_mat, page_table)
        elif mode == InferenceMode.DECODE:
            return self.forward_decode(hidden_states, start_pos, rot_mats, trans_mat, page_table)
        else:
            raise ValueError(f"Unknown mode: {mode}")


__all__ = ["Qwen3MoeAttention"]
