#!/bin/bash

export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": true,
    "enable_logging": false,
    "report_name": "qwen3",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'

python -m tracy -r -p -v -m pytest tests/test_moe_perf.py::test_moe_decode
