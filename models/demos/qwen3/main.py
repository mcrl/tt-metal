import fire
from typing import Optional

from models.demos.qwen3.generation import Qwen3MoE


def main(
        ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
        tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
        config_path: Optional[str] = None,
):
    qwen3_moe = Qwen3MoE(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path)
    prompts = [
        "Four score and seven years ago our fathers brought",
        "We hold these truths to be",
    ]
    responses = qwen3_moe.generate(prompts, max_gen_len=16, temperature=0.4, top_p=0.8)

    for prompt, completion in responses:
        print("\033[31m" + prompt + "\033[0m" + completion + "\n")


if __name__ == "__main__":
    fire.Fire(main)
