import fire
from typing import Optional, Dict, List

from models.demos.qwen3.generation import Qwen3MoETT, Qwen3MoEReference
import ttnn
from loguru import logger
import tt_lock
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.device import create_mesh_device


def trim_and_avg(times: List[float]) -> float:
    times = sorted(times)
    times = times[1:-1]
    return sum(times) / len(times)


def main(
    ckpt_dir: str = "/shared/models/Qwen3-30B-A3B/",
    tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
    config_path: str = "/shared/models/Qwen3-30B-A3B/config.json",
    log_file: str = "benchmark_result.csv",
    batch_sizes: Optional[List[int]] = None,
    input_lengths: Optional[List[int]] = None,
    output_lengths: Optional[List[int]] = None,
):
    if batch_sizes is None:
        batch_sizes = [32]
    if input_lengths is None:
        # input_lengths = [16, 32]
        input_lengths = [128]
    if output_lengths is None:
        output_lengths = [8]

    if type(batch_sizes) == int:
        batch_sizes = [batch_sizes]
    if type(input_lengths) == int:
        input_lengths = [input_lengths]
    if type(output_lengths) == int:
        output_lengths = [output_lengths]

    device_params = {"trace_region_size": 128 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D}  # 256MB
    mesh_device = create_mesh_device(device_params)
    set_and_get_device_cache(mesh_device)

    qwen3_moe = Qwen3MoETT(
        mesh_device=mesh_device,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        config_path=config_path,
        batch_size=max(batch_sizes),
    )

    for batch_size in batch_sizes:
        for input_length in input_lengths:
            for output_length in output_lengths:
                prefill_time = trim_and_avg(qwen3_moe.measure_prefill_time(batch_size, input_length))
                decode_time = trim_and_avg(qwen3_moe.measure_decode_time(batch_size, input_length, output_length))

                ttft_ms = prefill_time * 1000
                tpot_ms = decode_time * 1000
                output_token_per_s = batch_size / decode_time
                print(
                    f"{batch_size},{input_length},{output_length},{ttft_ms:.2f},{tpot_ms:.2f},{output_token_per_s:.2f}"
                )

                with open(log_file, "a") as f:
                    f.write(
                        f"{batch_size},{input_length},{output_length},{ttft_ms:.2f},{tpot_ms:.2f},{output_token_per_s:.2f}\n"
                    )


if __name__ == "__main__":
    fire.Fire(main)
