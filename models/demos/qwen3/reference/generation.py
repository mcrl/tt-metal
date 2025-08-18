import json
from typing import Optional, List
import torch

from tokenizers import Tokenizer

from models.demos.qwen3.reference.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.reference.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from models.demos.qwen3.reference.utils.loader import load, materialize


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = torch.gt(torch.sub(probs_sum, probs_sort), p)
    probs_sort.masked_fill_(mask, 0.0)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class Qwen3MoE:
    def __init__(self, ckpt_dir: str, tokenizer_path: str, config_path: Optional[str] = None) -> None:
        torch.manual_seed(42)
        torch.set_default_device(torch.device("cpu"))
        torch.set_default_dtype(torch.float16)

        data = None
        if config_path is not None:
            with open(config_path, "r") as f:
                data = json.load(f)

        self.config = Qwen3MoeConfig.from_dict(data)
        with torch.device("meta"):
            self.model = Qwen3MoeModel(self.config)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        materialize(self.model)
        load(ckpt_dir, self.model)
        self.model.eval()

    def generate(self, prompts: List[str], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9) -> List[List[str]]:
        prompt_tokens = [self.tokenizer.encode(prompt).ids for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.config.max_batch_size

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_seq_len

        total_len = min(self.config.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.config.pad_token_id
        tokens = torch.full(size=(batch_size, total_len), fill_value=pad_id, dtype=torch.int64)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.int64)

        prev_pos = 0
        eos_id = self.config.eos_token_id
        eos_reached = torch.tensor([False] * batch_size)
        input_text_mask = torch.ne(tokens, pad_id)
        for curr_pos in range(min_prompt_len, total_len):
            with torch.inference_mode():
                logits = self.model(tokens[:, prev_pos:curr_pos], start_pos=prev_pos)

            if temperature > 0:
                probs = torch.softmax(torch.div(logits[:, -1, :], temperature), dim=-1)
                next_tokens = sample_top_p(probs, top_p).reshape(-1)
            else:
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = torch.where(condition=input_text_mask[:, curr_pos], input=tokens[:, curr_pos], other=next_tokens)
            tokens[:, curr_pos] = next_tokens

            eos_reached = torch.logical_or(eos_reached, torch.logical_and(torch.logical_not(input_text_mask[:, curr_pos]), torch.eq(next_tokens, eos_id)))
            prev_pos = curr_pos
            if all(eos_reached):
                break

        tokens = tokens.tolist()
        prompt_lengths = [len(t) for t in prompt_tokens]
        split_tokens = [(output[:length], output[length:]) for output, length in zip(tokens, prompt_lengths)]
        return list(map(self.tokenizer.decode_batch, split_tokens))
