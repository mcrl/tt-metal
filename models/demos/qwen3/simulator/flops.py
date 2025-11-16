from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode


def flops_matmul(M, N, K):
    return 2 * M * N * K


def flops_bmm(B, M, N, K):
    return 2 * B * M * N * K


def flops_linear(batch_size, input_size, output_size, bias=False):
    return flops_matmul(batch_size, input_size, output_size) + (output_size if bias else 0)


def flops_attention_block(config, batch_size, sequence_length, mode: InferenceMode):
    total_flops = 0

    if mode == InferenceMode.PREFILL:
        seq_len = sequence_length
        kv_seq_len = sequence_length
    else:  # DECODE
        seq_len = 1
        kv_seq_len = sequence_length

    # Q, K, V projections
    q_flops = flops_linear(batch_size * seq_len, config.hidden_size, config.num_attention_heads * config.head_dim)
    k_flops = flops_linear(batch_size * seq_len, config.hidden_size, config.num_key_value_heads * config.head_dim)
    v_flops = flops_linear(batch_size * seq_len, config.hidden_size, config.num_key_value_heads * config.head_dim)
    total_flops += q_flops + k_flops + v_flops

    # Attention scores: Q @ K^T
    # Shape: [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, kv_seq_len]
    # Note: K/V heads are repeated to match Q heads
    attention_scores_flops = flops_bmm(batch_size * config.num_attention_heads, seq_len, kv_seq_len, config.head_dim)
    total_flops += attention_scores_flops

    # Attention output: attention_weights @ V
    # Shape: [batch_size, num_heads, seq_len, kv_seq_len] @ [batch_size, num_heads, kv_seq_len, head_dim]
    attention_output_flops = flops_bmm(batch_size * config.num_attention_heads, seq_len, config.head_dim, kv_seq_len)
    total_flops += attention_output_flops

    # Output projection
    o_flops = flops_linear(batch_size * seq_len, config.num_attention_heads * config.head_dim, config.hidden_size)
    total_flops += o_flops

    return total_flops


def flops_expert(num_tokens, config):
    return (
        flops_linear(num_tokens, config.hidden_size, config.moe_intermediate_size)  # gate_proj
        + flops_linear(num_tokens, config.hidden_size, config.moe_intermediate_size)  # up_proj
        + flops_linear(num_tokens, config.moe_intermediate_size, config.hidden_size)  # down_proj
    )


def flops_moe_block(config, batch_size, sequence_length, mode: InferenceMode):
    total_flops = 0

    if mode == InferenceMode.PREFILL:
        seq_len = sequence_length
    else:  # DECODE
        seq_len = 1

    num_tokens = batch_size * seq_len

    # Gate computation
    gate_flops = flops_linear(num_tokens, config.hidden_size, config.num_experts)
    total_flops += gate_flops

    # Expert computation
    # Each token goes to top_k experts
    expert_flops = flops_expert(num_tokens, config) * config.num_experts_per_tok
    total_flops += expert_flops

    return total_flops


if __name__ == "__main__":
    config_30b = Qwen3MoeConfig.from_file("/shared/models/Qwen3-30B-A3B/config.json")
    config_235b = Qwen3MoeConfig.from_file("/shared/models/Qwen3-235B-A22B/config.json")

    batch_size = 1
    seq_length = 2048

    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    print()

    for config, model_name in [(config_30b, "Qwen3-30B-A3B"), (config_235b, "Qwen3-235B-A22B")]:
        print(f"{model_name}:")
        num_layers = config.num_hidden_layers

        # Prefill mode
        attention_flops_prefill = flops_attention_block(config, batch_size, seq_length, InferenceMode.PREFILL)
        moe_flops_prefill = flops_moe_block(config, batch_size, seq_length, InferenceMode.PREFILL)

        print(f"  Prefill mode:")
        print(f"    Attention FLOPs: {attention_flops_prefill / 1e9:.3f} GFLOPs")
        print(f"    MoE FLOPs: {moe_flops_prefill / 1e9:.3f} GFLOPs")
        print(f"    Total per layer: {(attention_flops_prefill + moe_flops_prefill) / 1e9:.3f} GFLOPs")
        print(f"    Total per model: {(attention_flops_prefill + moe_flops_prefill) / 1e9 * num_layers:.3f} GFLOPs")

        # Decode mode
        attention_flops_decode = flops_attention_block(config, batch_size, seq_length, InferenceMode.DECODE)
        moe_flops_decode = flops_moe_block(config, batch_size, seq_length, InferenceMode.DECODE)
        print(f"  Decode mode:")
        print(f"    Attention FLOPs: {attention_flops_decode / 1e9:.3f} GFLOPs")
        print(f"    MoE FLOPs: {moe_flops_decode / 1e9:.3f} GFLOPs")
        print(f"    Total per layer: {(attention_flops_decode + moe_flops_decode) / 1e9:.3f} GFLOPs")
        print(f"    Total per model: {(attention_flops_decode + moe_flops_decode) / 1e9 * num_layers:.3f} GFLOPs")
        print()
