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
$ pytest tests/test_attn.py -v

# MoE tests
$ pytest tests/test_moe.py -v

# MoE operation tests
$ pytest tests/test_moe_mapping.py -v
$ pytest tests/test_moe_routing_tensors.py -v
```

### Performance Tests

```bash
# Attention performance
$ python -m tracy -r -p -v -m pytest tests/test_attn_perf.py::test_attn_prefill
$ python -m tracy -r -p -v -m pytest tests/test_attn_perf.py::test_attn_decode

# MoE performance
$ python -m tracy -r -p -v -m pytest tests/test_moe_perf.py::test_moe_prefill
$ python -m tracy -r -p -v -m pytest tests/test_moe_perf.py::test_moe_decode
```
