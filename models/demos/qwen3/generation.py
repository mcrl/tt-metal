import json
from typing import Optional, List
import torch
import ttnn
import time
from tokenizers import Tokenizer
from utils.profiler import Profiler
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeModel as Qwen3MoeModelReference
from models.demos.qwen3.tt.qwen import Qwen3MoeModel as Qwen3MoeModelTT
from models.demos.qwen3.utils.timer import print_timer_all, reset_timer, profile_time, start_timer, stop_timer
from models.demos.qwen3.utils.profiler import disable_profiler, enable_profiler
from models.demos.qwen3.common.loader import load, materialize
from models.utility_functions import enable_persistent_kernel_cache
from utils.memory_state import print_memory_state

from models.tt_transformers.tt.rope import RotarySetup


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = torch.gt(torch.sub(probs_sum, probs_sort), p)
    probs_sort.masked_fill_(mask, 0.0)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class Qwen3MoEReference:
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
            self.model = Qwen3MoeModelReference(self.config)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        materialize(self.model)
        load(ckpt_dir, self.model)
        self.model.eval()

    def generate(
        self, prompts: List[str], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9
    ) -> List[List[str]]:
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

        warmup = True
        if warmup is True:
            for curr_pos in range(min_prompt_len, total_len):
                with torch.inference_mode():
                    mode = "prefill" if prev_pos == 0 else "decode"
                    logits = self.model(tokens[:, prev_pos:curr_pos], start_pos=prev_pos, mode=mode)
                prev_pos = curr_pos
        prev_pos = 0

        iter_times = []
        generate_start_time = time.time()
        for curr_pos in range(min_prompt_len, total_len):
            iter_start_time = time.time()
            with torch.inference_mode():
                logits = self.model(tokens[:, prev_pos:curr_pos], start_pos=prev_pos, mode="decode")
            iter_times.append(time.time() - iter_start_time)

            if temperature > 0:
                probs = torch.softmax(torch.div(logits[:, -1, :], temperature), dim=-1)
                next_tokens = sample_top_p(probs, top_p).reshape(-1)
            else:
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = torch.where(
                condition=input_text_mask[:, curr_pos], input=tokens[:, curr_pos], other=next_tokens
            )
            tokens[:, curr_pos] = next_tokens

            eos_reached = torch.logical_or(
                eos_reached,
                torch.logical_and(torch.logical_not(input_text_mask[:, curr_pos]), torch.eq(next_tokens, eos_id)),
            )
            prev_pos = curr_pos
            if all(eos_reached):
                break

        generate_end_time = time.time()
        print(
            f"Generation Time: {generate_end_time - generate_start_time:.3f}s, {batch_size=}, {min_prompt_len=}, {max_prompt_len=}, {max_gen_len=}"
        )

        tokens = tokens.tolist()
        prompt_lengths = [len(t) for t in prompt_tokens]
        split_tokens = [(output[:length], output[length:]) for output, length in zip(tokens, prompt_lengths)]
        return list(map(self.tokenizer.decode_batch, split_tokens)), iter_times


class Qwen3MoETT:
    def __init__(
        self, mesh_device: ttnn.Device, ckpt_dir: str, tokenizer_path: str, batch_size: int, config_path: Optional[str] = None
    ) -> None:
        torch.manual_seed(42)
        torch.set_default_device(torch.device("cpu"))
        torch.set_default_dtype(torch.bfloat16)

        self.mesh_device = mesh_device

        data = None
        if config_path is not None:
            with open(config_path, "r") as f:
                data = json.load(f)

        self.config = Qwen3MoeConfig.from_dict(data)

        # FIXME: ad-hoc for reducing KV cache memory
        self.config.max_batch_size = 32
        self.config.max_seq_len = 512

        with Profiler().trace_with_timer("Create-Model", level=0):
            with torch.device("meta"):
                self.model = Qwen3MoeModelTT(self.config, self.mesh_device)

            self.rope = RotarySetup(
                device=self.mesh_device,
                batch_size=batch_size,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                rope_theta=self.config.rope_theta,
            )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with Profiler().trace_with_timer("Load-Model", level=0):
            materialize(self.model)
            load(ckpt_dir, self.model)

        self.model.eval()

        with Profiler().trace_with_timer("Setup-TT", level=0):
            self.model.setup_tt()

        enable_persistent_kernel_cache()

    def measure_prefill_time(self, batch_size: int, input_length: int):
        input_tokens = torch.randint(0, self.config.vocab_size, (batch_size, input_length), dtype=torch.int64)
        input_tokens_tt = ttnn.from_torch(
            input_tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        rot_mats = self.rope.cos_matrix, self.rope.sin_matrix
        trans_mat = self.rope.transformation_mat_prefill

        for i in range(5):
            self.model(input_tokens_tt, start_pos=0, mode="prefill", rot_mats=rot_mats, trans_mat=trans_mat)
        ttnn.synchronize_device(self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.model(input_tokens_tt, start_pos=0, mode="prefill", rot_mats=rot_mats, trans_mat=trans_mat)
        ttnn.end_trace_capture(self.mesh_device, trace_id)

        trace_execute_start_time = time.time()
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        trace_execute_end_time = time.time()

        times = []
        for i in range(10):
            trace_execute_start_time = time.time()
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            trace_execute_end_time = time.time()
            times.append(trace_execute_end_time - trace_execute_start_time)

        ttnn.release_trace(self.mesh_device, trace_id)
        return times

    def measure_decode_time(self, batch_size: int, input_length: int, output_length: int):
        input_tokens = torch.randint(0, self.config.vocab_size, (batch_size, 1), dtype=torch.int64)
        input_tokens_tt = ttnn.from_torch(
            input_tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.synchronize_device(self.mesh_device)

        position_idxs = torch.full((batch_size,), input_length, dtype=torch.long)
        rot_mats = self.rope.get_rot_mats(position_idxs)
        trans_mat = self.rope.transformation_mat

        for i in range(5):
            self.model(input_tokens_tt, start_pos=input_length, mode="decode", rot_mats=rot_mats, trans_mat=trans_mat)
        ttnn.synchronize_device(self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.model(input_tokens_tt, start_pos=input_length, mode="decode", rot_mats=rot_mats, trans_mat=trans_mat)
        ttnn.end_trace_capture(self.mesh_device, trace_id)

        trace_execute_start_time = time.time()
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        trace_execute_end_time = time.time()

        times = []
        for i in range(10):
            trace_execute_start_time = time.time()
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            trace_execute_end_time = time.time()
            times.append(trace_execute_end_time - trace_execute_start_time)

        ttnn.release_trace(self.mesh_device, trace_id)
        return times

    def generate(
        self, prompts: List[str], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9
    ) -> List[List[str]]:

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

        ttnn.synchronize_device(self.mesh_device)
        prev_pos = 0

        iter_times = []
        generate_start_time = time.time()

        with Profiler().trace_with_timer("Generate", level=0):
            for curr_pos in range(min_prompt_len, total_len):
                print(f"curr_pos: {curr_pos}")
                iter_start_time = time.time()
                mode = "prefill" if prev_pos == 0 else "decode"

                ids = ttnn.from_torch(
                    tokens[:, prev_pos:curr_pos],
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    dtype=ttnn.uint32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                if mode == "prefill":
                    rot_mats = self.rope.cos_matrix, self.rope.sin_matrix
                    trans_mat = self.rope.transformation_mat_prefill
                else:
                    position_idxs = torch.full((batch_size,), prev_pos, dtype=torch.long)
                    rot_mats = self.rope.get_rot_mats(position_idxs)
                    trans_mat = self.rope.transformation_mat

                logits_tt = self.model(ids, rot_mats=rot_mats, trans_mat=trans_mat, start_pos=prev_pos, mode=mode)

                logits = ttnn.to_torch(
                    logits_tt,
                    dtype=self.config.dtype,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2),
                )
                iter_times.append(time.time() - iter_start_time)

                if temperature > 0:
                    probs = torch.softmax(torch.div(logits[:, -1, :], temperature), dim=-1)
                    next_tokens = sample_top_p(probs, top_p).reshape(-1)
                else:
                    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.where(
                    condition=input_text_mask[:, curr_pos], input=tokens[:, curr_pos], other=next_tokens
                )
                tokens[:, curr_pos] = next_tokens

                eos_reached = torch.logical_or(
                    eos_reached,
                    torch.logical_and(torch.logical_not(input_text_mask[:, curr_pos]), torch.eq(next_tokens, eos_id)),
                )
                prev_pos = curr_pos
                # print_memory_state(self.mesh_device)
                if all(eos_reached):
                    break

        tokens = tokens.tolist()
        prompt_lengths = [len(t) for t in prompt_tokens]
        split_tokens = [(output[:length], output[length:]) for output, length in zip(tokens, prompt_lengths)]
        generate_end_time = time.time()

        print(
            f"Generation Time: {generate_end_time - generate_start_time:.3f}s, {batch_size=}, {min_prompt_len=}, {max_prompt_len=}, {max_gen_len=}"
        )

        return list(map(self.tokenizer.decode_batch, split_tokens)), iter_times
