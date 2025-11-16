from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum
import torch
import json


@dataclass(frozen=False)
class Qwen3MoeConfig:
    # Default: Qwen3-30B-A3B
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1000000.0
    rope_scaling: Dict = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int = None
    max_window_layers: int = 48
    attention_dropout: float = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: List = None
    head_dim: int = 128
    _attn_implementation: str = "sdpa"
    max_seq_len: int = 40960
    max_batch_size: int = 8
    bos_token_id: int = 151643
    pad_token_id: int = 151643
    eos_token_id: int = 151645
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def get_config(cls, name: str):
        return getattr(cls, name)()

    @classmethod
    def from_dict(cls, data: Optional[Dict] = None):
        if data is None:
            return cls()
        # Filter out unexpected keyword arguments
        import inspect

        valid_params = inspect.signature(cls.__init__).parameters
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered_data)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def dict(self):
        return asdict(self)


class InferenceMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


__all__ = ["Qwen3MoeConfig", "InferenceMode"]
