import fire
from typing import Optional, Dict

from models.demos.qwen3.generation import Qwen3MoETT
import ttnn
from loguru import logger
from tests.scripts.common import get_updated_device_params
import tt_lock


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


def main(
        ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
        tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
        config_path: Optional[str] = None,
):
    mesh_device = create_mesh_device()
    qwen3_moe = Qwen3MoETT(mesh_device=mesh_device, ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path)
    prompts = [
        "Four score and seven years ago our fathers brought",
        "We hold these truths to be",
    ]
    responses = qwen3_moe.generate(prompts, max_gen_len=6, temperature=0.4, top_p=0.8)

    for prompt, completion in responses:
        print("\033[31m" + prompt + "\033[0m" + completion + "\n")


if __name__ == "__main__":
    fire.Fire(main)
