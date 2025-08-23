import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.sdpa import sdpa_forward as tt_sdpa_forward, repeat_kv
from models.demos.qwen3.reference.sdpa import sdpa_forward as cpu_sdpa_forward
from models.demos.qwen3.reference.rope import precompute_freqs_cis, apply_rotary_emb
import ttnn
from models.demos.qwen3.tt.lm_head import LMHead as LMHeadTT


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

        cache_shape = (config.max_batch_size, config.max_seq_len, self.num_key_value_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        cache_v = torch.zeros(cache_shape, dtype=config.dtype, device=torch.device("cpu"), requires_grad=False)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None
        assert not config.attention_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: Literal["prefill", "decode"] = "prefill"
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [batch_size, seq_len, -1, head_dim]

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = key_states
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = value_states

        key_states = self.cache_k[:batch_size, : start_pos + seq_len]
        value_states = self.cache_v[:batch_size, : start_pos + seq_len]

        query_states = query_states.transpose(1, 2)  # [B n S H]
        key_states = key_states.transpose(1, 2)  # [B n_key S H]
        value_states = value_states.transpose(1, 2)  # [B n_key S H]

        if self.num_key_value_heads != self.config.num_attention_heads:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        if mode == "prefill":
            mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
            tt_q = ttnn.from_torch(
                query_states,
                device=self.mesh_device,
                mesh_mapper=mapper,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            tt_k = ttnn.from_torch(
                key_states,
                device=self.mesh_device,
                mesh_mapper=mapper,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            tt_v = ttnn.from_torch(
                value_states,
                device=self.mesh_device,
                mesh_mapper=mapper,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )

            tt_out = tt_sdpa_forward(
                tt_q,
                tt_k,
                tt_v,
                attention_mask=None,
                dropout=0.0,
                scaling=self.scaling,
                mesh_device=self.mesh_device,
                mode=mode,
            )
            attn_output = ttnn.to_torch(tt_out, dtype=self.config.dtype,
                                        mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))

            # from utils.test_utils import compare_tensor_pcc
            # compare_tensor_pcc(attn_output, attn_output_cpu, msg="sdpa", assert_mode=True)

            # Cleanup device tensors
            ttnn.deallocate(tt_q)
            ttnn.deallocate(tt_k)
            ttnn.deallocate(tt_v)
            ttnn.deallocate(tt_out)
        else:
            attn_output = cpu_sdpa_forward(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size, mesh_device=mesh_device) for _ in range(self.num_experts)]
        )

        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = torch.mul(expert_layer(current_state), routing_weights[top_x, idx, None])

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


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
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.mesh_device = mesh_device

        self.self_attn = Qwen3MoeAttention(config, layer_idx, mesh_device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, mesh_device)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps, mesh_device=mesh_device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: Literal["prefill", "decode"] = "prefill"
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            mode=mode,
        )

        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


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
        self.register_buffer("position_embeddings", position_embeddings, persistent=False)

        assert config.sliding_window is None

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, input_ids: torch.LongTensor, start_pos: int = 0, mode: Literal["prefill", "decode"] = "prefill") -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        position_embeddings = self.position_embeddings[start_pos: start_pos + seq_len]
        attention_mask = (
            torch.full(size=(1, 1, seq_len, start_pos + seq_len), fill_value=True, dtype=torch.bool)
            .triu_(diagonal=start_pos + 1)
            .logical_not_()
        )

        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                start_pos=start_pos,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                mode=mode,
            )
            print(f"layer {layer_idx} forward done")

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


__all__ = ["Qwen3MoeModel"]
