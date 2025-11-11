# Qwen3-MoE

## Clone & Build

```
$ git clone https://github.com/mcrl/tt-metal --recurse-submodules
$ cd tt-metal/
$ git checkout qwen3_bringup
$ ./build_metal.sh -c -p --release

export TT_METAL_HOME=${pwd}
export PYTHONPATH=${pwd}:${pwd}/ttnn
```

## Run

### Unit Tests

```bash
$ cd models/demos/qwen3

# Attention tests
$ pytest tests/test_attn.py

# MoE tests
$ pytest tests/test_moe.py
```

### Performance Tests

```bash
# Attention performance
$ ./run.sh attn_prefill
$ ./run.sh attn_decode

# MoE performance
$ ./run.sh moe_prefill
$ ./run.sh moe_decode
```

### Run Models
```bash
export TT_TRACE=1 # enable trace on decode
export TT_ENABLE_MATERIALIZE=1 # dynamic load
export TT_MESHDEVICE_SHAPE=4,2 # mesh device shape
export QWEN3_MODEL="" # folder that contains the model files
export QWEN3_MODEL_DIR="" # parent directory that contains the QWEN3_MODEL folder

python generate_model_cache.py # generate model cache
python main.py --batch_size=128 --prompt_len=64 --gen_tokens=64
```