import tt_lock
import fire
from typing import Optional, Dict
import os
from models.demos.qwen3.generation import Qwen3MoETT, Qwen3MoEReference
import ttnn
from loguru import logger
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.profiler import profile_trace
from models.demos.qwen3.utils.timer import print_timer_all
from models.demos.qwen3.utils.device import create_mesh_device


ttnn.CONFIG.enable_model_cache = True


def perftest_tt(
    batch_size: int,
    prompt_len: int,
    gen_tokens: int,
    ckpt_dir: str,
    tokenizer_path: str,
    config_path: str,
):
    # Create device with trace region size for trace capture
    device_params = {"trace_region_size": 128 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D}  # 256MB
    mesh_device = create_mesh_device(device_params)
    set_and_get_device_cache(mesh_device)

    qwen3_moe = Qwen3MoETT(
        mesh_device=mesh_device,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        config_path=config_path,
    )

    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = qwen3_moe.generate(prompts, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8)

    return prompt_and_responses, iter_times


def perftest_reference(
    batch_size: int,
    prompt_len: int,
    gen_tokens: int,
    ckpt_dir: str,
    tokenizer_path: str,
    config_path: str,
):
    qwen3_moe_reference = Qwen3MoEReference(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path)

    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = qwen3_moe_reference.generate(
        prompts, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8
    )

    return prompt_and_responses, iter_times


def main(
    ckpt_dir: str = "/shared/models/Qwen3-30B-A3B",
    tokenizer_path: str = "/shared/models/Qwen3-30B-A3B/tokenizer.json",
    config_path: Optional[str] = "/shared/models/Qwen3-30B-A3B/config.json",
    batch_size: int = 128,
    prompt_len: int = 64,
    gen_tokens: int = 32,
    run_tt: bool = True,
    run_reference: bool = False,
):
    init_trace_file()

    ran_any = False

    if run_tt:
        ran_any = True
        prompt_and_responses_tt, iter_times_tt = perftest_tt(
            batch_size, prompt_len, gen_tokens, ckpt_dir, tokenizer_path, config_path
        )
        print(f"TT Generation Results:")
        for i in range(batch_size):
            print("\033[31m" + prompt_and_responses_tt[i][0] + "\033[0m" + prompt_and_responses_tt[i][1] + "\n")

    if run_tt:
        ttft_ms = iter_times_tt[0] * 1000
        decode_times = iter_times_tt[1:] if len(iter_times_tt) > 1 else []
        avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
        tokens_per_second_per_user = 1 / avg_decode_time if avg_decode_time > 0 else 0

        print(f"TT TTFT: {ttft_ms:.2f} ms")
        print(f"TT T/S/U: {tokens_per_second_per_user:.2f}")
        print(f"TT Output Tokens/s: {tokens_per_second_per_user * batch_size:.2f}")

    if run_reference:
        ran_any = True
        prompt_and_responses_reference, iter_times_reference = perftest_reference(
            batch_size, prompt_len, gen_tokens, ckpt_dir, tokenizer_path, config_path
        )
        print(f"Reference Time: {sum(iter_times_reference)}")

        print(f"Reference Results:")
        for i in range(batch_size):
            print(
                "\033[31m"
                + prompt_and_responses_reference[i][0]
                + "\033[0m"
                + prompt_and_responses_reference[i][1]
                + "\n"
            )

    if not ran_any:
        print("No runs selected. Set run_tt=True and/or run_reference=True.")

    print_timer_all()


if __name__ == "__main__":
    fire.Fire(main)
