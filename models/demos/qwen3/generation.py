import json
import os
from typing import Optional, List
import torch
import ttnn
import time
from tokenizers import Tokenizer
from utils.profiler import Profiler
from loguru import logger
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeModel as Qwen3MoeModelReference
from models.demos.qwen3.tt.qwen import Qwen3MoeModel as Qwen3MoeModelTT
from models.demos.qwen3.utils.timer import print_timer_all, reset_timer, profile_time, start_timer, stop_timer
from models.demos.qwen3.utils.profiler import disable_profiler, enable_profiler
from models.demos.qwen3.common.loader import load, materialize
from models.common.utility_functions import enable_persistent_kernel_cache
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

        self.config.unit_batch_size = 32
        self.config.max_batch_size = 1024
        self.config.max_seq_len = 64

        self.config.block_size = self.config.max_seq_len * 2
        self.config.max_num_blocks = self.config.max_batch_size

        self.dp_degree = mesh_device.shape[0]
        self.bsz_per_device = batch_size // self.dp_degree

        with Profiler().trace_with_timer("Create-Model", level=0):
            with torch.device("meta"):
                self.model = Qwen3MoeModelTT(self.config, self.mesh_device)

            self.rope = RotarySetup(
                device=self.mesh_device,
                batch_size=self.config.unit_batch_size,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                rope_theta=self.config.rope_theta,
            )

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with Profiler().trace_with_timer("Load-Model", level=0):
            # Check environment variable to enable/disable materialize (default: disabled)
            enable_materialize = os.environ.get("TT_ENABLE_MATERIALIZE", "0").lower() in ("1", "true", "yes")

            if enable_materialize:
                print("Materialize enabled via TT_ENABLE_MATERIALIZE")
                from models.demos.qwen3.common.lazy_loader import materialize, load
                materialize(self.model)  # Keep as meta tensors
                load(ckpt_dir, self.model)
            else:
                print("Materialize disabled (default) - loading weights directly")
                from models.demos.qwen3.common.loader import load, materialize
                materialize(self.model)
                load(ckpt_dir, self.model)

        self.model.eval()

        with Profiler().trace_with_timer("Setup-TT", level=0):
            self.model.setup_tt()

        enable_persistent_kernel_cache()
        
        self.trace_id_decode = None
        self.trace_inputs_decode = None
        self.trace_output_decode = None

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

    def _capture_trace_decode(self, tokens, prev_pos, page_table, bsz_per_device):
        """
        Captures a trace for decode mode execution.

        """
        logger.info("Capturing decode trace...")
        
        # Allocate persistent input tensors for trace
        ids_host = tokens[:, prev_pos:prev_pos+1]
        self.trace_ids_persistent = ttnn.from_torch(
            ids_host,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        
        start_pos_host = torch.tensor([prev_pos for _ in range(self.config.unit_batch_size)])
        self.trace_start_pos_persistent = ttnn.as_tensor(
            start_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        
        position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
        rot_mats_initial = self.rope.get_rot_mats(position_idxs)
        trans_mat = self.rope.transformation_mat
        
        self.trace_rot_cos_spec = rot_mats_initial[0].shape
        self.trace_rot_sin_spec = rot_mats_initial[1].shape
        
        self.trace_rot_mats_persistent = rot_mats_initial
        
        # Compile run
        logits_tt = self.model(
            self.trace_ids_persistent, 
            rot_mats=self.trace_rot_mats_persistent, 
            trans_mat=trans_mat, 
            start_pos=self.trace_start_pos_persistent, 
            mode="decode", 
            page_table=page_table
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Done compiling model for trace")
        
        # Warmup runs
        for i in range(3):
            position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
            rot_mats_warmup = self.rope.get_rot_mats(position_idxs)
            
            self.trace_rot_mats_persistent[0].deallocate()
            self.trace_rot_mats_persistent[1].deallocate()
            
            self.trace_rot_mats_persistent = rot_mats_warmup
            
            logits_tt = self.model(
                self.trace_ids_persistent, 
                rot_mats=self.trace_rot_mats_persistent, 
                trans_mat=trans_mat, 
                start_pos=self.trace_start_pos_persistent, 
                mode="decode", 
                page_table=page_table
            )
        ttnn.synchronize_device(self.mesh_device)
        
        position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
        rot_mats_capture = self.rope.get_rot_mats(position_idxs)
        
        self.trace_rot_mats_persistent[0].deallocate()
        self.trace_rot_mats_persistent[1].deallocate()
        self.trace_rot_mats_persistent = rot_mats_capture
        
        self.trace_id_decode = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.trace_output_decode = self.model(
            self.trace_ids_persistent, 
            rot_mats=self.trace_rot_mats_persistent, 
            trans_mat=trans_mat, 
            start_pos=self.trace_start_pos_persistent, 
            mode="decode", 
            page_table=page_table
        )
        ttnn.end_trace_capture(self.mesh_device, self.trace_id_decode, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        
        logger.info("Done capturing decode trace")

    def _execute_trace_decode(self, tokens, prev_pos, page_table, bsz_per_device):
        """
        Executes the captured decode trace with updated inputs.

        """
        # Update persistent input token tensor with new data
        ids_host = tokens[:, prev_pos:prev_pos+1]
        ids_to_copy = ttnn.from_torch(
            ids_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(
            ids_to_copy, 
            self.trace_ids_persistent,
            cq_id=0
        )
        
        # Update persistent start_pos tensor with new position
        start_pos_host = torch.tensor([prev_pos for _ in range(self.config.unit_batch_size)])
        start_pos_to_copy = ttnn.from_torch(
            start_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(
            start_pos_to_copy,
            self.trace_start_pos_persistent,
            cq_id=0
        )
        
        position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
        rot_mats_new = self.rope.get_rot_mats(position_idxs)
        
        assert rot_mats_new[0].shape == self.trace_rot_cos_spec, \
            f"Rotation cos matrix shape mismatch: {rot_mats_new[0].shape} vs {self.trace_rot_cos_spec}"
        assert rot_mats_new[1].shape == self.trace_rot_sin_spec, \
            f"Rotation sin matrix shape mismatch: {rot_mats_new[1].shape} vs {self.trace_rot_sin_spec}"
        
        self.trace_rot_mats_persistent[0].deallocate()
        self.trace_rot_mats_persistent[1].deallocate()
        
        self.trace_rot_mats_persistent = rot_mats_new
        
        # Execute trace
        ttnn.execute_trace(self.mesh_device, self.trace_id_decode, cq_id=0, blocking=False)
        
        return self.trace_output_decode

    def generate(
        self, prompts: List[str], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9
    ) -> List[List[str]]:

        prompt_tokens = [self.tokenizer.encode(prompt).ids for prompt in prompts]

        batch_size = len(prompt_tokens)
        bsz_per_device = batch_size // self.dp_degree

        permutation = torch.randperm(self.config.max_num_blocks, device="cpu")
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, self.config.max_num_blocks // batch_size)
        page_table_tt = ttnn.as_tensor(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, dims=(0, None))
        )
        page_table_tt_list = []
        for i in range(bsz_per_device // self.config.unit_batch_size):
            page_table_tt_list.append(page_table_tt[i * self.config.unit_batch_size:(i + 1) * self.config.unit_batch_size, :])
        
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

        use_trace = os.environ.get("TT_TRACE", "1") == "1"
        if use_trace:
            logger.warning("Trace ENABLED")

        with Profiler().trace_with_timer("Warmup", level=0):
            logger.info("Warmup: Running prefill...")
            curr_pos = min_prompt_len
            
            ids = ttnn.from_torch(
                tokens[:, 0:curr_pos],
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            start_pos = ttnn.as_tensor(
                torch.tensor([0 for _ in range(bsz_per_device)]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
            )
            rot_mats = self.rope.cos_matrix, self.rope.sin_matrix
            trans_mat = self.rope.transformation_mat_prefill
            
            logits_tt = self.model(ids, rot_mats=rot_mats, trans_mat=trans_mat, start_pos=start_pos, mode="prefill", page_table=page_table_tt)
            ttnn.synchronize_device(self.mesh_device)
            logger.info("Warmup: Prefill complete")
            
            logger.info("Warmup: Running decode (1 token)...")
            prev_pos = min_prompt_len
            
            ids_decode = ttnn.from_torch(
                tokens[:, prev_pos:prev_pos+1],
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            start_pos_decode = ttnn.as_tensor(
                torch.tensor([prev_pos for _ in range(self.config.unit_batch_size)]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
            )
            position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
            rot_mats_decode = self.rope.get_rot_mats(position_idxs)
            trans_mat_decode = self.rope.transformation_mat
            
            logits_tt = self.model(ids_decode, rot_mats=rot_mats_decode, trans_mat=trans_mat_decode, 
                                 start_pos=start_pos_decode, mode="decode", page_table=page_table_tt_list)
            ttnn.synchronize_device(self.mesh_device)
            logger.info("Warmup: Decode compilation complete")
            
            if use_trace:
                logger.info("Capturing decode trace during warmup...")
                self._capture_trace_decode(tokens, prev_pos, page_table_tt_list, bsz_per_device)
                logger.info("Decode trace captured")

        ttnn.synchronize_device(self.mesh_device)

        prev_pos = 0
        iter_times = []
        generate_start_time = time.time()

        with Profiler().trace_with_timer("Generate", level=0):
            for curr_pos in range(min_prompt_len, total_len):
                print(f"curr_pos: {curr_pos}")
                iter_start_time = time.time()
                mode = "prefill" if prev_pos == 0 else "decode"
                page_table = page_table_tt

                if mode == "prefill":
                    ids = ttnn.from_torch(
                        tokens[:, prev_pos:curr_pos],
                        device=self.mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        dtype=ttnn.uint32,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    start_pos = ttnn.as_tensor(
                        torch.tensor([prev_pos for _ in range(bsz_per_device)]),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
                    )
                    rot_mats = self.rope.cos_matrix, self.rope.sin_matrix
                    trans_mat = self.rope.transformation_mat_prefill
                    logits_tt = self.model(ids, rot_mats=rot_mats, trans_mat=trans_mat, start_pos=start_pos, mode=mode, page_table=page_table)
                else:
                    if use_trace and self.trace_id_decode is not None:
                        logits_tt = self._execute_trace_decode(tokens, prev_pos, page_table_tt_list, bsz_per_device)
                    else:
                        ids = ttnn.from_torch(
                            tokens[:, prev_pos:curr_pos],
                            device=self.mesh_device,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                            dtype=ttnn.uint32,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                        start_pos = ttnn.as_tensor(
                            torch.tensor([prev_pos for _ in range(self.config.unit_batch_size)]),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
                        )
                        position_idxs = torch.full((self.config.unit_batch_size,), prev_pos, dtype=torch.long)
                        rot_mats = self.rope.get_rot_mats(position_idxs)
                        trans_mat = self.rope.transformation_mat
                        logits_tt = self.model(ids, rot_mats=rot_mats, trans_mat=trans_mat, start_pos=start_pos, mode=mode, page_table=page_table_tt_list)

                with Profiler().trace_with_timer("Sampling", level=2):        
                    _, next_tokens_tt = ttnn.topk(logits_tt[:, -1, :], k=1, dim=-1, largest=True)
                    next_tokens = ttnn.to_torch(
                        next_tokens_tt,
                        dtype=torch.int32,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1),
                    )[:, 0]
                    iter_times.append(time.time() - iter_start_time)

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

        tokens = tokens.tolist()
        prompt_lengths = [len(t) for t in prompt_tokens]
        split_tokens = [(output[:length], output[length:]) for output, length in zip(tokens, prompt_lengths)]
        generate_end_time = time.time()

        print(
            f"Generation Time: {generate_end_time - generate_start_time:.3f}s, {batch_size=}, {min_prompt_len=}, {max_prompt_len=}, {max_gen_len=}"
        )

        return list(map(self.tokenizer.decode_batch, split_tokens)), iter_times

    def __del__(self):
        """Destructor to release trace resources"""
        if self.trace_id_decode is not None:
            logger.info("Releasing decode trace")
            ttnn.release_trace(self.mesh_device, self.trace_id_decode)
