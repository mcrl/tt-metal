import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal, Tuple

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward, repeat_kv, repeat_kv_dim2
from models.demos.qwen3.reference.sdpa import sdpa_forward as cpu_sdpa_forward
from models.demos.qwen3.reference.rope import precompute_freqs_cis, apply_rotary_emb
import ttnn
from models.demos.qwen3.tt.lm_head import LMHead as LMHeadTT
from models.demos.qwen3.tt.rope import precompute_freqs_cis as precompute_freqs_cis_tt
from models.demos.qwen3.tt.rope import apply_rotary_emb as apply_rotary_emb_tt

from models.demos.qwen3.utils.test_utils import compare_tensor_pcc


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = pow(self.head_dim, -0.5)
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps, mesh_device=mesh_device)  # thus post q_norm does not need reshape

        self.sliding_window = None

        # FIXME: ad-hoc repeat
        self.KV_REPEAT_COEF = 2
        self.kv_heads_per_device = self.num_key_value_heads * self.KV_REPEAT_COEF // self.mesh_device.shape[1]

        # cache_shape = (config.max_batch_size, config.max_seq_len, self.num_key_value_heads, self.head_dim)
        # cache_shape = (config.max_batch_size, config.max_seq_len, self.num_key_value_heads * self.KV_REPEAT_COEF, self.head_dim)
        cache_shape = (config.max_batch_size, self.num_key_value_heads * self.KV_REPEAT_COEF, config.max_seq_len, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        cache_v = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None
        assert not config.attention_bias

    def setup_tt(self):
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        self.q_proj_weight = ttnn.from_torch(self.q_proj.weight.transpose(0, 1), device=self.mesh_device, mesh_mapper=mapper,
                                             dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.q_proj_weight = ttnn.to_layout(self.q_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def reshape_weight(x, head_dim, repeats):
            x = x.transpose(0, 1)  # [H, D*n_kv_heads]
            hidden_size, _ = x.shape
            x = x.view(hidden_size, -1, head_dim)  # [H n_kv_heads D]
            x = x.repeat_interleave(repeats=repeats, dim=1)  # [H, n_kv_heads*R, D]
            x = x.view(hidden_size, -1)  # [H, n_kv_heads*R*D]
            return x.contiguous()

        self.cache_k_tt = ttnn.from_torch(self.cache_k, device=self.mesh_device,
                                          mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                          dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.cache_k_tt = ttnn.to_layout(self.cache_k_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.cache_v_tt = ttnn.from_torch(self.cache_v, device=self.mesh_device,
                                          mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                          dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.cache_v_tt = ttnn.to_layout(self.cache_v_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.k_proj_weight = ttnn.from_torch(reshape_weight(self.k_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
                                             device=self.mesh_device, mesh_mapper=mapper,
                                             dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.k_proj_weight = ttnn.to_layout(self.k_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.v_proj_weight = ttnn.from_torch(reshape_weight(self.v_proj.weight, head_dim=self.head_dim, repeats=self.KV_REPEAT_COEF),
                                             device=self.mesh_device, mesh_mapper=mapper,
                                             dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.v_proj_weight = ttnn.to_layout(self.v_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.query_rmsnorm_weight_tt = ttnn.from_torch(self.q_norm.weight, device=self.mesh_device,
                                                       mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                       dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.query_rmsnorm_weight_tt = ttnn.to_layout(self.query_rmsnorm_weight_tt, ttnn.TILE_LAYOUT,
                                                      dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.key_rmsnorm_weight_tt = ttnn.from_torch(self.k_norm.weight, device=self.mesh_device,
                                                     mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                     dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.key_rmsnorm_weight_tt = ttnn.to_layout(self.key_rmsnorm_weight_tt, ttnn.TILE_LAYOUT,
                                                    dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.o_proj_weight = ttnn.from_torch(
            self.o_proj.weight,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.o_proj_weight = ttnn.to_layout(self.o_proj_weight, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        position_embeddings: torch.Tensor,
        position_embeddings_tt: Tuple[ttnn.Tensor, ttnn.Tensor],  # (cos, sin)
        attention_mask: torch.Tensor,
        mode: Literal["prefill", "decode"] = "prefill"
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)  # [batch_size, seq_len, n_head, head_dim]

        """ ############## QKV Projection ############## """
        hidden_states_tt = hidden_states
        # mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        # hidden_states_tt = ttnn.from_torch(hidden_states, device=self.mesh_device, mesh_mapper=mapper,
        #                                    dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # hidden_states_tt = ttnn.to_layout(hidden_states_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_states_tt = ttnn.linear(hidden_states_tt, self.q_proj_weight, dtype=ttnn.bfloat16)
        query_states_tt = ttnn.reshape(query_states_tt, hidden_shape)
        query_states_tt = ttnn.rms_norm(query_states_tt, epsilon=self.config.rms_norm_eps, weight=self.query_rmsnorm_weight_tt)

        key_states_tt = ttnn.linear(hidden_states_tt, self.k_proj_weight, dtype=ttnn.bfloat16)
        key_states_tt = ttnn.reshape(key_states_tt, hidden_shape)

        key_states_tt = ttnn.rms_norm(key_states_tt, epsilon=self.config.rms_norm_eps, weight=self.key_rmsnorm_weight_tt)
        value_states_tt = ttnn.linear(hidden_states_tt, self.v_proj_weight, dtype=ttnn.bfloat16)
        value_states_tt = ttnn.reshape(value_states_tt, hidden_shape)

        ttnn.deallocate(hidden_states_tt)

        """ ############## Rotary Embedding ############## """
        query_states_tt, key_states_tt = apply_rotary_emb_tt(query_states_tt, key_states_tt, position_embeddings_tt)
        # # [B n S H]
        query_states_tt = ttnn.permute(query_states_tt, dims=(0, 2, 1, 3))
        key_states_tt = ttnn.permute(key_states_tt, dims=(0, 2, 1, 3))
        value_states_tt = ttnn.permute(value_states_tt, dims=(0, 2, 1, 3))

        if mode == "prefill":
            for b in range(batch_size):
                ttnn.kv_cache.fill_cache_for_user_(self.cache_k_tt, key_states_tt[b:b+1], b)
                ttnn.kv_cache.fill_cache_for_user_(self.cache_v_tt, value_states_tt[b:b+1], b)
        elif mode == "decode":
            for b in range(batch_size):
                # (jinpyo) Manual batch offset to fill KV cache for B > 1
                # Fill Cache[b, :, start_pos, :] <- key_states_tt[b, :, 0, :]
                ttnn.kv_cache.update_cache_for_token_(self.cache_k_tt,
                                                      key_states_tt[b:b+1],
                                                      start_pos + b * self.kv_heads_per_device * self.config.max_seq_len)
                ttnn.kv_cache.update_cache_for_token_(self.cache_v_tt,
                                                      value_states_tt[b:b+1],
                                                      start_pos + b * self.kv_heads_per_device * self.config.max_seq_len)

        key_states_tt = self.cache_k_tt[:batch_size, :, : start_pos + seq_len, :]
        value_states_tt = self.cache_v_tt[:batch_size, :, : start_pos + seq_len, :]

        """ ############## SDPA ############## """
        tt_q = query_states_tt
        tt_k = key_states_tt
        tt_v = value_states_tt
        if mode == "prefill":
            tt_attention_mask = None
        else:
            tt_attention_mask = ttnn.from_torch(
                attention_mask.repeat(batch_size, self.mesh_device.shape[1], 1, 1),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tt_out = tt_sdpa_forward(
            query_states_tt,
            key_states_tt,
            value_states_tt,
            attention_mask=attention_mask if mode == "decode" else None,
            dropout=0.0,
            scaling=self.scaling,
            mesh_device=self.mesh_device,
            mode=mode,
        )  # [B S N H/N] split dim=2

        tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[1], tt_out.shape[2] * tt_out.shape[3]])

        attn_output = ttnn.to_torch(tt_out, dtype=self.config.dtype,
                                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))

        # Cleanup device tensors
        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)
        if tt_attention_mask is not None:
            ttnn.deallocate(tt_attention_mask)

        """ ############## Output Projection ############## """
        # tt_out: [B S H/8], split along dim=2. So [B S H/8](x8)
        # tt_weight: [H H], split along dim=1. So [H H/8](x8)
        tt_out = ttnn.to_layout(tt_out, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        linear_output_ttnn = ttnn.linear(
            tt_out,
            self.o_proj_weight,
            transpose_a=False,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # All-reduce across devices to sum the partial results
        B, S, H = linear_output_ttnn.shape
        linear_output_ttnn = linear_output_ttnn.reshape(B, S, 1, H)
        linear_output_ttnn_reduced = ttnn.reduce_scatter(
            linear_output_ttnn,
            dim=-1,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear
        )
        ttnn.synchronize_device(self.mesh_device)
        linear_output_ttnn_gathered = ttnn.all_gather(
            linear_output_ttnn_reduced,
            dim=-1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            mesh_device=self.mesh_device
        )
        # [B S 1 H]
        ttnn.synchronize_device(self.mesh_device)

        output = ttnn.reshape(linear_output_ttnn_gathered, (batch_size, seq_len, hidden_dim))
        return output


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int, mesh_device: ttnn.Device):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        assert config.hidden_act == "silu"

    def setup_tt(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size, mesh_device=mesh_device) for _ in range(self.num_experts)]
        )

        self.layer_idx = layer_idx

    def setup_tt(self):
        self.gate_weight_tt = ttnn.from_torch(
            self.gate.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.gate_weight_tt = ttnn.to_layout(self.gate_weight_tt,
                                             ttnn.TILE_LAYOUT,
                                             dtype=ttnn.bfloat16,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = ttnn.to_torch(hidden_states, dtype=self.config.dtype, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        hidden_states = hidden_states[:hidden_states.shape[0] // 8, :, :]

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # [BS H], S=1 at decode phase

        hidden_states_tt = ttnn.from_torch(hidden_states,
                                           device=self.mesh_device,
                                           mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                           dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states_tt = ttnn.to_layout(hidden_states_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        router_logits = self.gate(hidden_states)  # [BS N]
        router_logits_tt = ttnn.linear(hidden_states_tt, self.gate_weight_tt, dtype=ttnn.bfloat16)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # [BS N]

        routing_weights_tt = ttnn.softmax(router_logits_tt, dim=1)  # [BS N]

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [BS K] [BS K]
        routing_weights_tt, selected_experts_tt = ttnn.topk(routing_weights_tt, self.top_k, dim=1, largest=True)  # [BS K] [BS K]

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights_tt = ttnn.div(routing_weights_tt,
                                          ttnn.sum(routing_weights_tt, dim=1, keepdim=True))

        # compare_tensor_pcc(routing_weights, routing_weights_tt_cpu)

        routing_weights = routing_weights.to(hidden_states.dtype)  # [BS K]

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )  # [BS H]
        final_hidden_states_tt = ttnn.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=ttnn.bfloat16, device=self.mesh_device,
        )  # [BS H]

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_layer_tt = ttnn.from_torch(
                expert_layer.state_dict()["down_proj.weight"],
                device=self.mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            expert_layer_tt = ttnn.to_layout(expert_layer_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # E = num_experts = 128, top_k = 8
        # [BS 8 E] => [E 8 BS] = [128 8 BS]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # [E 8 BS] => [E] > 0 => [E] => [E_active, 1]
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # print(f"{expert_hitted=}")

        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            # print(f"{expert_idx=} {expert_mask[expert_idx]=} {expert_mask[expert_idx].squeeze(0)=}")
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # print(f"{expert_idx=} {idx=} {top_x=}")

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)  # [N H]
            current_hidden_states = torch.mul(expert_layer(current_state), routing_weights[top_x, idx, None])

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        final_hidden_states_tt = ttnn.from_torch(final_hidden_states,
                                                 device=self.mesh_device,
                                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                                                 dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        final_hidden_states_tt = ttnn.to_layout(final_hidden_states_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return final_hidden_states_tt


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, mesh_device: ttnn.Device = None):
        super().__init__()
        self.mesh_device = mesh_device
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = torch.mul(hidden_states, torch.rsqrt(variance + self.variance_epsilon))
        return torch.mul(self.weight, hidden_states.to(input_dtype))


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.mesh_device = mesh_device

        self.self_attn = Qwen3MoeAttention(config, layer_idx, mesh_device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)

    def setup_tt(self):
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
        self.post_attention_layernorm_weight_tt = ttnn.to_layout(
            self.post_attention_layernorm_weight_tt, ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        position_embeddings: torch.Tensor,
        position_embeddings_tt: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: torch.Tensor,
        mode: Literal["prefill", "decode"] = "prefill"
    ) -> torch.Tensor:

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

    def forward(self, input_ids: torch.LongTensor, start_pos: int = 0, mode: Literal["prefill", "decode"] = "prefill") -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        position_embeddings = self.position_embeddings[start_pos: start_pos + seq_len]

        pos_embs_cos = self.position_embeddings_tt[0][start_pos: start_pos + seq_len]
        pos_embs_sin = self.position_embeddings_tt[1][start_pos: start_pos + seq_len]

        cos_tt = ttnn.from_torch(pos_embs_cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
        sin_tt = ttnn.from_torch(pos_embs_sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))

        attention_mask = (
            torch.full(size=(1, 1, seq_len, start_pos + seq_len), fill_value=True, dtype=torch.bool)
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
            position_embeddings_tt = cos_tt, sin_tt

            hidden_states_tt = decoder_layer(
                hidden_states=hidden_states_tt,
                start_pos=start_pos,
                position_embeddings=position_embeddings,
                position_embeddings_tt=position_embeddings_tt,
                attention_mask=attention_mask,
                mode=mode,
            )
            print(f"layer {layer_idx} forward done")

        ttnn.deallocate(attention_mask_tt)

        hidden_states_tt = ttnn.rms_norm(hidden_states_tt, epsilon=self.config.rms_norm_eps, weight=self.norm_weight_tt)
        logits_tt = ttnn.linear(hidden_states_tt, self.lm_head_weight_tt, dtype=ttnn.bfloat16)
        logits_tt_cpu = ttnn.to_torch(logits_tt, dtype=self.config.dtype, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))

        return logits_tt_cpu


__all__ = ["Qwen3MoeModel"]
