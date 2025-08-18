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
$ pytest tests/test_lm_head.py
$ pytest tests/test_embedding_1d.py
```