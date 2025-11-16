import fire
from typing import Optional, Dict
import os
from models.demos.qwen3.generation import Qwen3MoETT
import ttnn
from loguru import logger
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.profiler import profile_trace
from models.demos.qwen3.utils.timer import print_timer_all
from models.demos.qwen3.utils.device import create_mesh_device
from models.demos.qwen3.utils.profiler import init_trace_file
from models.demos.qwen3.tt.model_cache import get_model_path

ttnn.CONFIG.enable_model_cache = True

def perftest_tt(
    batch_size: int,
    prompt_len: int,
    gen_tokens: int,
):
    # Get model paths from environment variables
    model_path = get_model_path()
    ckpt_dir = model_path
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    config_path = os.path.join(model_path, "config.json")

    # Create device with trace region size for trace capture
    device_params = {"trace_region_size": 128 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}  # 256MB
    mesh_device = create_mesh_device(device_params)
    set_and_get_device_cache(mesh_device)

    qwen3_moe = Qwen3MoETT(
        mesh_device=mesh_device,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        max_seq_len=prompt_len + gen_tokens,
        config_path=config_path,
    )

    prompts = load_prompts(batch_size, prompt_len)
    prompt_and_responses, iter_times = qwen3_moe.generate(prompts, prompt_len, max_gen_len=gen_tokens, temperature=0.7, top_p=0.8)

    return prompt_and_responses, iter_times

def main(
    batch_size: int = 256,
    prompt_len: int = 32,
    gen_tokens: int = 128
):
    init_trace_file()

    prompt_and_responses_tt, iter_times_tt = perftest_tt(
        batch_size, prompt_len, gen_tokens
    )
    print(f"TT Generation Results:")
    for i in range(batch_size):
        print("\033[31m" + prompt_and_responses_tt[i][0] + "\033[0m" + prompt_and_responses_tt[i][1] + "\n")

    ttft_ms = iter_times_tt[0] * 1000
    decode_times = iter_times_tt[1:] if len(iter_times_tt) > 1 else []
    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
    tokens_per_second_per_user = 1 / avg_decode_time if avg_decode_time > 0 else 0

    print(f"TT TTFT: {ttft_ms:.2f} ms")
    print(f"TT T/S/U: {tokens_per_second_per_user:.2f}")
    print(f"TT Output Tokens/s: {tokens_per_second_per_user * batch_size:.2f}")
    print(f"Decode times (s): {[f'{t:.4f}' for t in decode_times]}")


    print_timer_all()


if __name__ == "__main__":
    fire.Fire(main)
