#!/bin/bash

set -e

MATMUL_IMPLS=("moe_bmm" "ttnn.matmul")

declare -a PROMPTS=()

p=32
for b in 32 64 128 256; do
    PROMPTS+=("$p $b")
done

p=64
for b in 32 64 128; do
    PROMPTS+=("$p $b")
done

p=128
for b in 32 64; do
    PROMPTS+=("$p $b")
done

for impl in "${MATMUL_IMPLS[@]}"; do
    export MOE_MATMUL_IMPL="$impl"
    echo "======================================================"
    echo "Running with MOE_MATMUL_IMPL=$impl"
    echo "======================================================"

    for pair in "${PROMPTS[@]}"; do
        p=$(echo $pair | awk '{print $1}')
        b=$(echo $pair | awk '{print $2}')
        echo ">>> Running prompt_len=$p, batch_size=$b"
        python main.py --prompt_len="$p" --batch_size="$b"
        echo ""
    done
done

