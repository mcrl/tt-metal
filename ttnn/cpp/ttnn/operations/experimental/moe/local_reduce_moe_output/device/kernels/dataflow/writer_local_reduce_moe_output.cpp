// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * Writer Kernel for Local Reduce MoE Output (Multi-Core, Token-Parallel)
 *
 * This kernel reads accumulated results from the compute kernel and writes
 * them to the output tensor in DRAM.
 *
 * For each token assigned to this core:
 * 1. Wait for accumulated result from compute kernel
 * 2. Write result to output[token_idx, :] in DRAM
 *
 * Compile-time args:
 * - output_is_dram: Whether output buffer is in DRAM
 * - row_size_bytes: Byte size of one output row (H * element_size)
 *
 * Runtime args:
 * - output_buffer_addr: Address of output tensor (T, H)
 * - start_token_idx: First token index for this core
 * - end_token_idx: Last token index (exclusive) for this core
 */

void kernel_main() {
    // Runtime arguments
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t start_token_idx = get_arg_val<uint32_t>(1);
    uint32_t end_token_idx = get_arg_val<uint32_t>(2);

    // Circular buffer IDs (static)
    constexpr tt::CBIndex cb_id_output = tt::CBIndex::c_16;

    // Compile-time arguments
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(1);

    // Create address generator for output
    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };
    // Write each output row
    for (uint32_t token_idx = start_token_idx; token_idx < end_token_idx; token_idx++) {
        // Wait for accumulated result from compute kernel
        cb_wait_front(cb_id_output, 1);
        uint32_t output_l1_addr = get_read_ptr(cb_id_output);
        // Write to output[token_idx, :]
        uint64_t output_row_noc_addr = get_noc_addr(token_idx, output_addrgen);
        noc_async_write(output_l1_addr, output_row_noc_addr, row_size_bytes);
        noc_async_write_barrier();
        // Free the output buffer slot
        cb_pop_front(cb_id_output, 1);
    }
}
