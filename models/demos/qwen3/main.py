import fire
from typing import Optional, Dict

from models.demos.qwen3.generation import Qwen3MoETT, Qwen3MoEReference
import ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
import tt_lock
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.profiler import profile_trace


def create_mesh_device(device_params: Optional[Dict] = None):
    params = dict(device_params or {})
    updated_device_params = get_updated_device_params(params)
    device_ids = ttnn.get_device_ids()

    # Default mesh shape: Galaxy (32) -> 4x8; otherwise 1 x num_devices
    default_mesh_shape = ttnn.MeshShape(4, 8) if len(device_ids) == 32 else ttnn.MeshShape(1, len(device_ids))

    fabric_config = params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params.setdefault("mesh_shape", default_mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)
    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")

    return mesh_device


def perftest_tt(batch_size: int, prompt_len: int, gen_tokens: int,
                ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
                tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
                config_path: str = "/shared/models/Qwen3-30B-A3B/config.json"):
    mesh_device = create_mesh_device()
    set_and_get_device_cache(mesh_device)

    qwen3_moe = Qwen3MoETT(
        mesh_device=mesh_device, ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path
    )

    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = qwen3_moe.generate(prompts, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8)

    return prompt_and_responses, iter_times


def perftest_reference(batch_size: int, prompt_len: int, gen_tokens: int,
                       ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
                       tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
                       config_path: str = "/shared/models/Qwen3-30B-A3B/config.json"):
    qwen3_moe_reference = Qwen3MoEReference(
        ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path
    )

    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = qwen3_moe_reference.generate(prompts, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8)

    return prompt_and_responses, iter_times


def main(
    ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
    tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
    config_path: Optional[str] = None,
):
    batch_size = 8
    prompt_len = 64
    gen_tokens = 64
    prompt_and_responses_tt, iter_times_tt = perftest_tt(batch_size, prompt_len, gen_tokens, ckpt_dir, tokenizer_path, config_path)
    prompt_and_responses_reference, iter_times_reference = perftest_reference(batch_size, prompt_len, gen_tokens, ckpt_dir, tokenizer_path, config_path)
    print(f"TT Time: {sum(iter_times_tt)}")
    print(f"Reference Time: {sum(iter_times_reference)}")

    print(f"TT Results:")
    for i in range(batch_size):
        print("\033[31m" + prompt_and_responses_tt[i][0] + "\033[0m" + prompt_and_responses_tt[i][1] + "\n")
    print(f"Reference Results:")
    for i in range(batch_size):
        print("\033[31m" + prompt_and_responses_reference[i][0] + "\033[0m" + prompt_and_responses_reference[i][1] + "\n")


if __name__ == "__main__":
    fire.Fire(main)
