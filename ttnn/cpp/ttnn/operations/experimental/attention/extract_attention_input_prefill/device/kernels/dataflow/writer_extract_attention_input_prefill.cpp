// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// Dummy writer kernel for extract_attention_input_prefill
// PURPOSE: Test kernel compilation and launch
// Does nothing - placeholder for real implementation

void kernel_main() {
    // Get compile-time args (test compilation)
    constexpr uint32_t batch_per_device = get_compile_time_arg_val(0);
    constexpr uint32_t seq_len = get_compile_time_arg_val(1);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(2);

    // Get runtime args (test runtime arg passing)
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    // Dummy implementation - do nothing
    // Real implementation will:
    // 1. Read tiles from circular buffer
    // 2. Write to output tensor in shape [B//dp, 1, S, H]
    // 3. Handle optional dtype conversion (bfloat16 -> bfloat8_b)
}
