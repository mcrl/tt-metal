import fire
from typing import Optional, Dict
import os
import sys
import json
import ttnn
from loguru import logger
from test_dataset.dataset_loader import load_prompts
from models.demos.qwen3.utils.timer import set_and_get_device_cache
from models.demos.qwen3.utils.profiler import profile_trace
from models.demos.qwen3.utils.timer import print_timer_all
from models.demos.qwen3.utils.device import create_mesh_device
from models.demos.qwen3.utils.profiler import init_trace_file
from models.demos.qwen3.tt.model_cache import get_model_path
from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.attention import Qwen3MoeAttention
from models.demos.qwen3.tt.moe import Qwen3MoeSparseMoeBlock
from models.demos.qwen3.common.lazy_loader import init_lazy_loader

ttnn.CONFIG.enable_model_cache = True

model_path = get_model_path()
ckpt_dir = model_path
init_lazy_loader(ckpt_dir)

tokenizer_path = os.path.join(model_path, "tokenizer.json")
config_path = os.path.join(model_path, "config.json")

# Set environment variable to enable cache generation mode
os.environ["GENERATE_MODEL_CACHE"] = "1"

# Load config using Qwen3MoeConfig to ensure all fields match
with open(config_path, "r") as f:
    data = json.load(f)
config = Qwen3MoeConfig.from_dict(data)
config.max_batch_size = 32
config.max_seq_len = 512
config.block_size = 32
config.max_num_blocks = 512
config._attn_implementation = "sdpa"

import torch

# Set defaults to match generation.py
torch.manual_seed(42)
torch.set_default_device(torch.device("cpu"))
torch.set_default_dtype(torch.bfloat16)

device_params = {"trace_region_size": 128 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
mesh_device = create_mesh_device(device_params)
set_and_get_device_cache(mesh_device)

num_layers = config.num_hidden_layers

for i in range(num_layers):
    print(f"Generating cache for layer {i}/{num_layers}...")
    
    # Create layers with meta tensors, just like real execution
    with torch.device("meta"):
        attn = Qwen3MoeAttention(config, i, mesh_device)
        moe = Qwen3MoeSparseMoeBlock(config, i, mesh_device)
    
    # Setup will trigger lazy loading and cache generation
    attn.setup_tt()
    moe.setup_tt()
    
    print(f"Layer {i} cache generated successfully")