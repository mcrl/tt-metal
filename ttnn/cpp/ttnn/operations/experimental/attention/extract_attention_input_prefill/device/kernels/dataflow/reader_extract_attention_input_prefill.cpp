// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// Dummy reader kernel for extract_attention_input_prefill
// PURPOSE: Test kernel compilation and launch
// Does nothing - placeholder for real implementation

void kernel_main() {
    // Get compile-time args (test compilation)
    constexpr uint32_t batch_size = get_compile_time_arg_val(0);
    constexpr uint32_t seq_len = get_compile_time_arg_val(1);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(2);
    constexpr uint32_t dp = get_compile_time_arg_val(3);

    // Get runtime args (test runtime arg passing)
    uint32_t input_addr = get_arg_val<uint32_t>(0);

    // Dummy implementation - do nothing
    // Real implementation will:
    // 1. Calculate device index from mesh
    // 2. Extract batch slice: [dev_idx * (B/dp) : (dev_idx+1) * (B/dp), :, :]
    // 3. Read tiles from DRAM to circular buffer
}
