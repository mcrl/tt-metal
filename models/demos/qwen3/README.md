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

```
$ cd models/demos/qwen3
$ pytest tests/test_attn.py
$ pytest tests/test_moe.py

$ python -m tracy -r -p -v -m pytest tests/test_attn_perf.py::test_attn_prefill
$ python -m tracy -r -p -v -m pytest tests/test_attn_perf.py::test_attn_decode

$ python -m tracy -r -p -v -m pytest tests/test_moe_perf.py::test_moe_prefill
$ python -m tracy -r -p -v -m pytest tests/test_moe_perf.py::test_moe_decode
```
