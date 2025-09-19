import fire
from typing import Optional, Dict

from models.demos.qwen3.generation import Qwen3MoETT, Qwen3MoEReference
import ttnn
from loguru import logger
import tt_lock
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.device import create_mesh_device


def perftest_tt(
    model: Qwen3MoETT,
    batch_size: int,
    prompt_len: int,
    gen_tokens: int,
):
    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = model.generate(prompts, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8)
    return prompt_and_responses, iter_times


def main(
    ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
    tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
    config_path: str = "/shared/models/Qwen3-30B-A3B/config.json",
):
    mesh_device = create_mesh_device()
    set_and_get_device_cache(mesh_device)

    qwen3_moe = Qwen3MoETT(
        mesh_device=mesh_device, ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path
    )

    perftest_tt(model=qwen3_moe, batch_size=8, prompt_len=64, gen_tokens=64)


if __name__ == "__main__":
    main()
