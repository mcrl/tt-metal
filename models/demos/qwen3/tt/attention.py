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

def save_activations(mesh_device, tensor: ttnn.Tensor, file_name: str, layer_name: str):
    if file_name == None:
        return

    sharded = False
    if ttnn.is_sharded(tensor):
        mem_cfg = tensor.memory_config()
        tensor = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        sharded = True

    tt_tensor = ttnn.clone(tensor)
    torch_tensor = ttnn.to_torch(
        tt_tensor,
        dtype=torch.bfloat16,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    torch.save(torch_tensor, f"activations/{file_name}_{layer_name}.pt")

    if sharded:
        tensor = ttnn.to_memory_config(tensor, mem_cfg)

class Qwen3MoeAttention(nn.Module):
    @profile_trace("create-layer", level=3, args={"class": "Qwen3MoeAttention"})
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.shape[0] * mesh_device.shape[1]

        self.dp = mesh_device.shape[0]
        self.tp = mesh_device.shape[1]

        # self.submeshes = mesh_device.create_submeshes(ttnn.MeshShape(mesh_device.shape[0] // self.dp, mesh_device.shape[1]))

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

        self.q_heads_per_device = self.num_attention_heads // self.tp

        self.KV_REPEAT_COEF = 2
        self.kv_heads_per_device = self.num_key_value_heads * self.KV_REPEAT_COEF // self.tp

        self.cache_shape = (
            config.max_num_blocks,
            self.num_key_value_heads * self.KV_REPEAT_COEF // self.tp,
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

        # Load weights lazily if they're still in meta state
        from models.demos.qwen3.common.lazy_loader import get_lazy_loader, load_parameter_for_module, is_meta_tensor
        lazy_loader = get_lazy_loader()
        
        if lazy_loader is not None:
            # Load weights on-demand if still in meta state
            param_prefix = f"layers.{self.layer_idx}.self_attn"
            
            if is_meta_tensor(self.q_proj.weight):
                load_parameter_for_module(self.q_proj, "weight", f"{param_prefix}.q_proj.weight")
            if is_meta_tensor(self.k_proj.weight):
                load_parameter_for_module(self.k_proj, "weight", f"{param_prefix}.k_proj.weight")
            if is_meta_tensor(self.v_proj.weight):
                load_parameter_for_module(self.v_proj, "weight", f"{param_prefix}.v_proj.weight")
            if is_meta_tensor(self.o_proj.weight):
                load_parameter_for_module(self.o_proj, "weight", f"{param_prefix}.o_proj.weight")
            
            # Load RMSNorm weights before calling their setup_tt()
            if is_meta_tensor(self.q_norm.weight):
                load_parameter_for_module(self.q_norm, "weight", f"{param_prefix}.q_norm.weight")
            if is_meta_tensor(self.k_norm.weight):
                load_parameter_for_module(self.k_norm, "weight", f"{param_prefix}.k_norm.weight")

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
        wq = torch.chunk(q_weight, self.tp, dim=1)
        wk = torch.chunk(k_weight, self.tp, dim=1)
        wv = torch.chunk(v_weight, self.tp, dim=1)

        for i in range(self.tp):
            qkv_list.append(torch.cat([wq[i], wk[i], wv[i]], dim=1))

        self.qkv_proj_weight = ttnn.as_tensor(
            torch.cat(qkv_list, dim=1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, dims=(None, 1)),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            # cache_file_name=ttnn_model_cache_path(f"235b_decoder_{self.layer_idx}_qkv_proj"),
        )
        
        # Free intermediate tensors
        del q_weight, k_weight, v_weight, qkv_list, wq, wk, wv

        self.q_norm.setup_tt()
        self.k_norm.setup_tt()

        o_weight = self.o_proj.weight
        o_weight = o_weight.reshape(self.hidden_size, -1, self.head_dim)
        o_weight = reshape_to_interleaved(o_weight).reshape(weight_shape)

        self.o_proj_weight = ttnn.as_tensor(
            o_weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, dims=(None, 1)),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            # cache_file_name=ttnn_model_cache_path(f"235b_decoder_{self.layer_idx}_o_proj"),
        )
        
        # Free intermediate tensor
        del o_weight

        self.init_kv_cache()

        weight = torch.zeros(1, self.num_devices, 32, 128) # 32 is batch size
        for i in range(self.num_devices):
            col = i // self.tp  # This determines which group of 32 to select
            weight[:, i, :, col * 32 : (col + 1) * 32] = torch.eye(32)

        self.slice_mat = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )

        # Create dp_degree tensor for extract_attention_input API
        # Each device gets its row index in the mesh (0 to dp-1)
        # Pattern: [0, 0, 0, 0, 1, 1, 1, 1] for dp=2, tp=4
        dp_degree_host = torch.tensor(
            [row for row in range(self.dp) for _ in range(self.tp)],
            dtype=torch.int32
        )
        self.dp_degree = ttnn.from_torch(
            dp_degree_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(0, None),  # Shard dim 0 across dp*tp devices
                mesh_shape=(self.num_devices, 1)  # Treat as 1D array
            ),
        )

        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.is_tt_setup = True
        
        # Free CPU RAM after uploading to device (if using lazy loader)
        if lazy_loader is not None:
            from models.demos.qwen3.common.lazy_loader import clear_module_weights
            clear_module_weights(self)
            print(f"✓ Layer {self.layer_idx} attention weights cleared from CPU RAM")

    
    def init_kv_cache(self):
        cache_k = torch.zeros(self.cache_shape, dtype=torch.bfloat16)
        cache_v = torch.zeros(self.cache_shape, dtype=torch.bfloat16)

        self.cache_k = ttnn.as_tensor(
            cache_k,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.cache_v = ttnn.as_tensor(
            cache_v,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def forward_prefill(
        self, hidden_states: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Prefill starts. Hidden stats: [1, B, S, H]"""

        batch_size, sequence_length, hidden_size = hidden_states.shape
        mem_cfg = ttnn.L1_MEMORY_CONFIG if sequence_length == 1 else ttnn.DRAM_MEMORY_CONFIG
        hidden_shape = (batch_size // self.dp, 1, sequence_length, -1)

        # Extract attention input for this device using new multi-core API
        # Replaces: reshape → matmul(slice_mat) → reshape
        hidden_states = ttnn.extract_attention_input(
            hidden_states,
            self.dp_degree,
            mesh_device=self.mesh_device,
            output_dtype=ttnn.bfloat16,
            memory_config=mem_cfg
        )

        # OLD CODE (replaced by extract_attention_input API):
        # hidden_states = ttnn.reshape(hidden_states, (batch_size, -1))
        # hidden_states = ttnn.matmul(self.slice_mat, hidden_states, dtype=ttnn.bfloat8_b, compute_kernel_config=self.compute_config, memory_config=mem_cfg)
        # hidden_states = ttnn.reshape(hidden_states, hidden_shape)

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            qkv_states = ttnn.linear(hidden_states, self.qkv_proj_weight, dtype=ttnn.bfloat8_b, compute_kernel_config=self.compute_config, memory_config=mem_cfg)
            # ttnn.deallocate(hidden_states)
            qkv_states = ttnn.typecast(qkv_states, ttnn.bfloat16)
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

        # print(f"query_states_pre_rot.shape: {query_states_pre_rot.shape}")
        

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
            for b in range(batch_size // self.dp):
                k_fill = ttnn.clone(key_states[b:b+1])
                v_fill = ttnn.clone(value_states[b:b+1])

                k_fill = ttnn.typecast(k_fill, ttnn.bfloat8_b)
                v_fill = ttnn.typecast(v_fill, ttnn.bfloat8_b)

                ttnn.experimental.paged_fill_cache(self.cache_k, k_fill, page_table, batch_idx=b)
                ttnn.experimental.paged_fill_cache(self.cache_v, v_fill, page_table, batch_idx=b)

            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

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
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_config, 
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_out_cat)

        with Profiler().trace_with_timer("all-reduce", level=4):
            B, _, S, H = linear_output.shape
            linear_output = ttnn.view(linear_output, (1, 1, B * S, H))

            linear_output = ttnn.experimental.reduce_scatter_minimal_async(
                linear_output,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                cluster_axis=1,
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
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
            )
            
            # linear_output = ttnn.experimental.all_gather_async(
            #     linear_output,
            #     2,
            #     cluster_axis=0,
            #     mesh_device=self.mesh_device,
            #     topology=ttnn.Topology.Linear,
            #     multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(0),
            #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
            #     num_links=1,
            # )

            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.view(linear_output, (-1, S, H))
            linear_output = ttnn.typecast(linear_output, ttnn.bfloat16)

        print(f"linear_output.shape: {linear_output.shape} {H=} {S=} {B=}")
        return linear_output

    def forward_decode(
        self, hidden_states: ttnn.Tensor, start_pos: ttnn.Tensor, rot_mats: Tuple[ttnn.Tensor, ttnn.Tensor], trans_mat: ttnn.Tensor, page_table: ttnn.Tensor,
        save_file_name: str = None
    ) -> ttnn.Tensor:
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        """Hidden state: [1, S=1, B, H]"""
        _, sequence_length, batch_size, hidden_size = hidden_states.shape

        # Extract attention input for this device using new multi-core API
        # Replaces: view → matmul(slice_mat) → view
        hidden_states = ttnn.extract_attention_input(
            hidden_states,
            self.dp_degree,
            mesh_device=self.mesh_device,
            output_dtype=ttnn.bfloat16,
            memory_config=mem_cfg,
        )

        save_activations(self.mesh_device, self.cache_k, save_file_name, "init_k_cache")
        save_activations(self.mesh_device, self.cache_v, save_file_name, "init_v_cache")
        save_activations(self.mesh_device, hidden_states, save_file_name, "init_hidden_states")

        # OLD CODE (replaced by extract_attention_input API):
        # hidden_states = ttnn.view(hidden_states, (batch_size, hidden_size))
        # hidden_states = ttnn.matmul(self.slice_mat, hidden_states, dtype=ttnn.bfloat8_b, compute_kernel_config=self.compute_config, memory_config=mem_cfg)
        # hidden_states = ttnn.view(hidden_states, (1, 1, batch_size // self.dp, hidden_size))

        with Profiler().trace_with_timer("qkv-proj-linear", level=4):
            qkv_states = ttnn.linear(hidden_states, self.qkv_proj_weight, dtype=ttnn.bfloat8_b, compute_kernel_config=self.compute_config, memory_config=mem_cfg)
            qkv_states = ttnn.typecast(qkv_states, ttnn.bfloat16)
            save_activations(self.mesh_device, qkv_states, save_file_name, "qkv_states")
            # ttnn.deallocate(hidden_states)
        """ QKV: [1, 1, B, H] """

        with Profiler().trace_with_timer("qkv-split", level=4):
            query_states_pre_rot, key_states_pre_rot, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv_states,
                num_heads=self.q_heads_per_device,
                num_kv_heads=self.kv_heads_per_device,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
            save_activations(self.mesh_device, query_states_pre_rot, save_file_name, "query_states_create_heads")
            save_activations(self.mesh_device, key_states_pre_rot, save_file_name, "key_states_create_heads")
            save_activations(self.mesh_device, value_states, save_file_name, "value_states_create_heads")
            ttnn.deallocate(qkv_states)
        """ QKV: [S=1, B, n, H] """

        with Profiler().trace_with_timer("rmsnorm", level=4):
            query_states_pre_rot = self.q_norm(query_states_pre_rot, mode=InferenceMode.DECODE)
            save_activations(self.mesh_device, query_states_pre_rot, save_file_name, "query_states_after_rmsnorm")
            key_states_pre_rot = self.k_norm(key_states_pre_rot, mode=InferenceMode.DECODE)
            save_activations(self.mesh_device, key_states_pre_rot, save_file_name, "key_states_after_rmsnorm")
        """ QKV: [1, B, n, H] """

        with Profiler().trace_with_timer("rope", level=4):
            save_activations(self.mesh_device, query_states_pre_rot, save_file_name, "query_states_pre_rope")
            save_activations(self.mesh_device, rot_mats[0], save_file_name, "rot_mat_cos")
            save_activations(self.mesh_device, rot_mats[1], save_file_name, "rot_mat_sin")
            save_activations(self.mesh_device, trans_mat, save_file_name, "trans_mat")
            
            query_states = ttnn.experimental.rotary_embedding_llama(
                query_states_pre_rot, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )
            save_activations(self.mesh_device, query_states, save_file_name, "query_states")
            ttnn.deallocate(query_states_pre_rot)

            key_states = ttnn.experimental.rotary_embedding_llama(
                key_states_pre_rot, rot_mats[0], rot_mats[1], trans_mat, is_decode_mode=True
            )
            save_activations(self.mesh_device, key_states, save_file_name, "key_states")
            ttnn.deallocate(key_states_pre_rot)
        """ Q: [S=1, B, n, H], K: [S=1, B, n, H], V: [S=1, B, n, H] """

        save_activations(self.mesh_device, self.cache_k, save_file_name, "before_update_k_cache")
        save_activations(self.mesh_device, self.cache_v, save_file_name, "before_update_v_cache")
        save_activations(self.mesh_device, query_states, save_file_name, "query_states_before_update")

        ttnn.experimental.paged_update_cache(self.cache_k, key_states, update_idxs_tensor=start_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(self.cache_v, value_states, update_idxs_tensor=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.mesh_device) # If not, cause hanging

        save_activations(self.mesh_device, self.cache_k, save_file_name, "after_update_k_cache")
        save_activations(self.mesh_device, self.cache_v, save_file_name, "after_update_v_cache")

        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        save_activations(self.mesh_device, query_states, save_file_name, "query_states_before_sdpa")
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
        save_activations(self.mesh_device, attn_output, save_file_name, "attn_output")
        ttnn.deallocate(attn_output)

        with Profiler().trace_with_timer("output-proj", level=4):
            linear_output = ttnn.linear(
                attn_output_cat,
                self.o_proj_weight,
                transpose_a=False,
                transpose_b=True,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_config, 
                memory_config=mem_cfg,
            )
            ttnn.deallocate(attn_output_cat)
        """ O: [1, 1, B, H] """

        with Profiler().trace_with_timer("all-reduce", level=4):
            _, _, B, H = linear_output.shape
            linear_output = ttnn.view(linear_output, (1, 1, B, H))

            linear_output = ttnn.experimental.reduce_scatter_minimal_async(
                linear_output,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
                cluster_axis=1,
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
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
            )
            # linear_output = ttnn.experimental.all_gather_async(
            #     linear_output,
            #     2,
            #     cluster_axis=0,
            #     mesh_device=self.mesh_device,
            #     topology=ttnn.Topology.Linear,
            #     multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(0),
            #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
            #     num_links=1,
            # )

            ttnn.synchronize_device(self.mesh_device)
            linear_output = ttnn.view(linear_output, shape=(1, 1, -1, H))
            linear_output = ttnn.typecast(linear_output, ttnn.bfloat16)
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
        mode: InferenceMode = InferenceMode.PREFILL,
        save_file_name: str = None,
    ) -> ttnn.Tensor:

        if mode == InferenceMode.PREFILL:
            return self.forward_prefill(hidden_states, rot_mats, trans_mat, page_table)
        elif mode == InferenceMode.DECODE:
            return self.forward_decode(hidden_states, start_pos, rot_mats, trans_mat, page_table, save_file_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

__all__ = ["Qwen3MoeAttention"]
