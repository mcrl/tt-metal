#!/bin/bash

unset TT_METAL_DPRINT_CORES

export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": true,
    "enable_logging": false,
    "report_name": "qwen3",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'

if [ $# -ne 1 ]; then
    echo "Usage: $0 {attn_prefill|attn_decode|moe_prefill|moe_decode}"
    exit 1
fi

mode=$1

case $mode in
    attn_prefill)
        test_cmd="tests/test_attn_perf.py::test_attn_prefill"
        ;;
    attn_decode)
        test_cmd="tests/test_attn_perf.py::test_attn_decode"
        ;;
    moe_prefill)
        test_cmd="tests/test_moe_perf.py::test_moe_prefill"
        ;;
    moe_decode)
        test_cmd="tests/test_moe_perf.py::test_moe_decode"
        ;;
    *)
        echo "Invalid argument: $mode"
        echo "Valid options: attn_prefill, attn_decode, moe_prefill, moe_decode"
        exit 1
        ;;
esac

tmp_log=$(mktemp /tmp/tracy_output_XXXXXX.log)

cleanup() {
    rm -f "$tmp_log"
}
trap cleanup EXIT

script -q /dev/null -c "bash -c 'python -m tracy -r -p -v -m pytest \"$test_cmd\" || exit 1'" 2>&1 | tee "$tmp_log"
csv_path=$(grep -oP '(?<=OPs csv generated at: ).*\.csv' "$tmp_log" | tail -n 1)

if [ -n "$csv_path" ]; then
    tt-perf-report "$csv_path"
else
    echo "Could not find CSV report path in output."
    exit 1
fi