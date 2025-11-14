import ttnn
import torch
from loguru import logger
from transformers import AutoConfig
import os
import json
from models.demos.qwen3.reference.modeling_qwen3_moe import Qwen3MoeDecoderLayer
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig, InferenceMode
from models.demos.qwen3.benchmark.moe_benchmark import Qwen3MoeSparseMoeBlock
import tracy
from models.demos.qwen3.utils.timer import set_and_get_device_cache
import time
from enum import Enum
import shutil
from models.demos.qwen3.benchmark.gpt_oss_moe import MLP

class Mode(Enum):
  PREFILL = 1
  DECODE = 2

class Impl(Enum):
  MOE_SNU = 1
  MOE_TT_BY_SNU = 2
  MOE_GPT_OSS = 3

def open_device():
  ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
  device = ttnn.open_mesh_device(mesh_shape = ttnn.MeshShape(4, 8),
                                 trace_region_size = 128 * 1024 * 1024)
  logger.info(f"multidevice with {device.get_num_devices()} devices is created with shape {device.shape}")
  return device

def close_device(device):
  ttnn.close_mesh_device(device)
  ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

def bench_single_case(num_experts, num_experts_per_tok, norm_topk_prob, hidden_size, moe_intermediate_size,
                      batch_size, seq_len, mode, impl, device, memory_config, use_trace=False):
  if mode == Mode.PREFILL:
    infer_mode = InferenceMode.PREFILL
    hidden_states_shape = (batch_size, seq_len, hidden_size)
  elif mode == Mode.DECODE:
    infer_mode = InferenceMode.DECODE
    hidden_states_shape = (1, seq_len, batch_size, hidden_size)
  else:
    raise ValueError(f'Unknown mode {mode}')

  torch.manual_seed(0) 

  if impl == Impl.MOE_SNU:
    layer_idx = 0
    tt_mlp = Qwen3MoeSparseMoeBlock(layer_idx, device,
                                    num_experts, num_experts_per_tok,
                                  norm_topk_prob, hidden_size, moe_intermediate_size)
    forward_impl = tt_mlp.forward
    tt_mlp.setup_tt(infer_mode, memory_config, bs=batch_size * seq_len)
    forward_kwargs = {'mem_cfg': memory_config, 'mode': infer_mode}
  elif impl == Impl.MOE_TT_BY_SNU:
    layer_idx = 0
    tt_mlp = Qwen3MoeSparseMoeBlock(layer_idx, device,
                                    num_experts, num_experts_per_tok,
                                  norm_topk_prob, hidden_size, moe_intermediate_size)
    forward_impl = tt_mlp.forward_tt
    tt_mlp.setup_tt(infer_mode, memory_config, bs=batch_size * seq_len)
    forward_kwargs = {'mem_cfg': memory_config, 'mode': infer_mode}
  elif impl == Impl.MOE_GPT_OSS:
    tt_mlp = MLP(device, num_experts_per_tok, num_experts, hidden_size, moe_intermediate_size)
    forward_impl = tt_mlp
    forward_kwargs = {}
  else:
    raise ValueError(f'Unknown impl {impl}')

  hidden_states = torch.randn(hidden_states_shape, dtype=torch.bfloat16)
  hidden_states_tt = ttnn.from_torch(
      hidden_states,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=device,
      mesh_mapper=ttnn.ReplicateTensorToMesh(device),
      memory_config=memory_config,
  )

  logger.info('Running tt_mlp')

  witers, niters = 3, 10
  
  if use_trace:
    tracy.signpost("Trace Warmup")
    ttnn.synchronize_device(device)
    st = time.time()
    output_tt = forward_impl(hidden_states_tt, **forward_kwargs)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Trace 1 iter: {elapsed:.3f} us per iter, total {elapsed:.3f} us')

    tracy.signpost("Trace Run")
    ttnn.synchronize_device(device)
    st = time.time()
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    try:
      output_tt = forward_impl(hidden_states_tt, **forward_kwargs)
    except Exception as e:
      ttnn.end_trace_capture(device, trace, cq_id=0)
      ttnn.release_trace(device, trace)
      raise e
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Trace 1 iter: {elapsed:.3f} us per iter, total {elapsed:.3f} us')

    tracy.signpost("Warmup")
    ttnn.synchronize_device(device)
    st = time.time()
    for _ in range(witers):
      ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Warmup {witers} iters: {elapsed / witers:.3f} us per iter, total {elapsed:.3f} us')

    tracy.signpost("Run")
    ttnn.synchronize_device(device)
    st = time.time()
    for _ in range(niters):
      ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Main {niters} iters: {elapsed / niters:.3f} us per iter, total {elapsed:.3f} us')

    ttnn.release_trace(device, trace)

    return [elapsed / niters]

  else:
    tracy.signpost("Warmup")
    ttnn.synchronize_device(device)
    st = time.time()
    for _ in range(witers):
      output_tt = forward_impl(hidden_states_tt, **forward_kwargs)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Warmup {witers} iters: {elapsed / witers:.3f} us per iter, total {elapsed:.3f} us')

    tracy.signpost("Run")
    ttnn.synchronize_device(device)
    st = time.time()
    for _ in range(niters):
      output_tt = forward_impl(hidden_states_tt, **forward_kwargs)
    ttnn.synchronize_device(device)
    et = time.time()
    elapsed = (et - st) * 1e6
    logger.info(f'Main {niters} iters: {elapsed / niters:.3f} us per iter, total {elapsed:.3f} us')

    return [elapsed / niters]

#MODELS = ['Qwen3-235B-A22B', 'Qwen3-30B-A3B', 'DeepSeek-V3', 'GPT-OSS-120B', 'GPT-OSS-20B']
MODELS = ['GPT-OSS-20B']
#PREFILL_BS_PAIR = [(32, 32), (32, 64), (32, 128), (32, 256), (32, 512), (32, 1024)]
PREFILL_BS_PAIR = [(32, 32)]
#DECODE_BS_PAIR = [(32, 1), (64, 1), (128, 1), (256, 1), (512, 1), (1024, 1), (2048, 1)]
DECODE_BS_PAIR = [(1, 32), (1, 64), (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048)]
#MODES = [Mode.PREFILL, Mode.DECODE]
MODES = [Mode.DECODE]
#IMPLS = [Impl.MOE_SNU, Impl.MOE_TT_BY_SNU]
IMPLS = [Impl.MOE_GPT_OSS]

def bench_all_cases(mesh_device):
  for model in MODELS:
    # Empty cache, because shapes are changing
    shutil.rmtree('/root/.cache/ttnn-weights', ignore_errors=True)

    with open(f'benchmark/configs/{model}.json') as f:
      config = json.load(f)
    num_experts = config['num_experts']
    num_experts_per_tok = config['num_experts_per_tok']
    norm_topk_prob = config['norm_topk_prob']
    hidden_size = config['hidden_size']
    moe_intermediate_size = config['moe_intermediate_size']

    for mode in MODES:
      if mode == Mode.PREFILL:
        BS_PAIR = PREFILL_BS_PAIR
      elif mode == Mode.DECODE:
        BS_PAIR = DECODE_BS_PAIR
      else:
        raise ValueError(f'Unknown mode {mode}')
      for batch_size, seq_len in BS_PAIR:
        row = []
        for impl in IMPLS:
          if mode == Mode.PREFILL:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
          elif mode == Mode.DECODE:
            memory_config = ttnn.L1_MEMORY_CONFIG
          else:
            raise ValueError(f'Unknown mode {mode}')

          try:
            logger.info(f'{mode=}, {impl=}, {batch_size=}, {seq_len=}, {model=}, {num_experts=}, {num_experts_per_tok=}, {norm_topk_prob=}, {hidden_size=}, {moe_intermediate_size=}, {memory_config=}')
            data = bench_single_case(num_experts, num_experts_per_tok, norm_topk_prob,
                              hidden_size, moe_intermediate_size,
                              batch_size, seq_len, mode, impl, mesh_device, memory_config)
            row.extend(data)
          except Exception as e:
            logger.error(f'Error occurred: {e}')
            if memory_config == ttnn.L1_MEMORY_CONFIG:
              logger.info('Retrying with DRAM')
              memory_config = ttnn.DRAM_MEMORY_CONFIG
              logger.info(f'{mode=}, {impl=}, {batch_size=}, {seq_len=}, {model=}, {num_experts=}, {num_experts_per_tok=}, {norm_topk_prob=}, {hidden_size=}, {moe_intermediate_size=}, {memory_config=}')
              try:
                data = bench_single_case(num_experts, num_experts_per_tok, norm_topk_prob,
                                  hidden_size, moe_intermediate_size,
                                  batch_size, seq_len, mode, impl, mesh_device, memory_config)
                row.extend(data)
              except Exception as e2:
                logger.error(f'Error occurred again: {e2}')
                row.extend(['err'])
            else:
              row.extend(['err'])
        with open('result.csv', 'a') as f:
          f.write(','.join([str(x) for x in row]) + '\n')

if __name__ == "__main__":
  device = open_device()
  set_and_get_device_cache(device)
  bench_all_cases(device)
  close_device(device)

# MAIN BENCH HERE



