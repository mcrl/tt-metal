# Reference Model

Model: [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)

The `modeling_qwen3_moe.py` file allow the creation and use of reference model objects without installing flash_attn and all its CUDA dependencies.

The other files here are experimental extractions / implementations by LLMs, so do not trust them blindly.

Loading the model structure without loading weights or CUDA dependencies:

```python
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from models.demos.qwen3_moe.reference.modeling_qwen3_moe import Qwen3MoeForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen3-235B-A22B", trust_remote_code=True)

with no_init_weights():
    model = Qwen3MoeForCausalLM._from_config(config)
```

Model structure:

```python
Qwen3MoeForCausalLM(
  (model): Qwen3MoeModel(
    (embed_tokens): Embedding(151936, 4096)
    (layers): ModuleList(
      (0-93): 94 x Qwen3MoeDecoderLayer(
        (self_attn): Qwen3MoeAttention(
          (q_proj): Linear(in_features=4096, out_features=8192, bias=False)
          (k_proj): Linear(in_features=4096, out_features=512, bias=False)
          (v_proj): Linear(in_features=4096, out_features=512, bias=False)
          (o_proj): Linear(in_features=8192, out_features=4096, bias=False)
          (q_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3MoeRMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MoeSparseMoeBlock(
          (gate): Linear(in_features=4096, out_features=128, bias=False)
          (experts): ModuleList(
            (0-127): 128 x Qwen3MoeMLP(
              (gate_proj): Linear(in_features=4096, out_features=1536, bias=False)
              (up_proj): Linear(in_features=4096, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): Qwen3MoeRMSNorm((4096,), eps=1e-06)
        (post_attention_layernorm): Qwen3MoeRMSNorm((4096,), eps=1e-06)
      )
    )
    (norm): Qwen3MoeRMSNorm((4096,), eps=1e-06)
    (rotary_emb): Qwen3MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
```
