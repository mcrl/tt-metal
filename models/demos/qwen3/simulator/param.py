from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig


def num_params_linear(input_size, output_size, bias=False):
    return input_size * output_size + (output_size if bias else 0)


def num_params_embedding(vocab_size, embedding_size):
    return vocab_size * embedding_size


def num_params_rms_norm(hidden_size):
    return hidden_size


def num_params_moeblock(config):
    # Gate projection
    gate_params = num_params_linear(config.hidden_size, config.num_experts, bias=False)

    # Each expert has gate_proj, up_proj, down_proj
    single_expert_params = (
        num_params_linear(config.hidden_size, config.moe_intermediate_size, bias=False)  # gate_proj
        + num_params_linear(config.hidden_size, config.moe_intermediate_size, bias=False)  # up_proj
        + num_params_linear(config.moe_intermediate_size, config.hidden_size, bias=False)  # down_proj
    )

    total_experts_params = single_expert_params * config.num_experts

    return gate_params + total_experts_params


def num_params_attentionblock(config):
    # Q, K, V, O projections
    q_proj_params = num_params_linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
    k_proj_params = num_params_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
    v_proj_params = num_params_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
    o_proj_params = num_params_linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    # Q and K normalization
    q_norm_params = num_params_rms_norm(config.head_dim)
    k_norm_params = num_params_rms_norm(config.head_dim)

    return q_proj_params + k_proj_params + v_proj_params + o_proj_params + q_norm_params + k_norm_params


def num_params_qwen3moe(config):
    num_layers = config.num_hidden_layers

    # Embeddings and LM head
    embedding_params = num_params_embedding(config.vocab_size, config.hidden_size)
    lm_head_params = num_params_linear(config.hidden_size, config.vocab_size, bias=False)

    # Final norm layer
    final_norm_params = num_params_rms_norm(config.hidden_size)

    # Per-layer parameters
    layer_params = 0
    for layer_idx in range(num_layers):
        # Attention block
        layer_params += num_params_attentionblock(config)

        # MoE block (only on certain layers based on decoder_sparse_step)
        if (layer_idx + 1) % config.decoder_sparse_step == 0:
            layer_params += num_params_moeblock(config)

        # Layer norms
        layer_params += num_params_rms_norm(config.hidden_size)  # input_layernorm
        layer_params += num_params_rms_norm(config.hidden_size)  # post_attention_layernorm

    return embedding_params + lm_head_params + final_norm_params + layer_params


if __name__ == "__main__":
    config_30b = Qwen3MoeConfig.from_file("/shared/models/Qwen3-30B-A3B/config.json")
    num_params_30b = num_params_qwen3moe(config_30b)
    print(f"Qwen3-30B-A3B: {num_params_30b / 1e9:.2f}B params")

    moe_params = num_params_moeblock(config_30b)
    attention_params = num_params_attentionblock(config_30b)
    print(f"Qwen3-30B-A3B: {moe_params / 1e9:.2f}B moe params, {attention_params / 1e9:.2f}B attention params")

    config_235b = Qwen3MoeConfig.from_file("/shared/models/Qwen3-235B-A22B/config.json")
    num_params_235b = num_params_qwen3moe(config_235b)
    print(f"Qwen3-235B-A22B: {num_params_235b / 1e9:.2f}B params")
