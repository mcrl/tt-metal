// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// Writer kernel for extract_attention_input (unified)
// PURPOSE: Write tiles from circular buffer to output tensor
// APPROACH: Safety-first - one tile at a time, barrier after each write
// MODES: Works for both prefill and decode modes, handles format conversion

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);  // Output buffer address

    // Compile-time arguments
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(0);  // Output buffer type
    constexpr uint32_t tiles_per_device = get_compile_time_arg_val(1);  // Number of tiles to write
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);             // CB index (0 for 2-kernel, 16 for 3-kernel)
    constexpr uint32_t output_data_format = get_compile_time_arg_val(3);  // Output data format

    // Get actual tile size from circular buffer (handles both bfloat16 and bfp8_b correctly)
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    // Setup address generator for interleaved buffer
    const InterleavedAddrGenFast<output_is_dram> addrgen = {
        .bank_base_address = output_addr,
        .page_size = tile_size_bytes,
        .data_format = static_cast<DataFormat>(output_data_format)
    };

    // SAFE PATTERN: Write one tile at a time, barrier immediately after each write
    // No batching for maximum safety
    for (uint32_t i = 0; i < tiles_per_device; i++) {
        // Wait for tile to be available in circular buffer
        cb_wait_front(cb_id, 1);

        // Get L1 address to read from
        uint32_t l1_addr = get_read_ptr(cb_id);

        // Output tiles are written sequentially starting from tile 0
        uint32_t tile_idx = i;

        // Get NOC address for this tile
        uint64_t noc_addr = get_noc_addr(tile_idx, addrgen);

        // Issue async write from L1 to DRAM
        noc_async_write(l1_addr, noc_addr, tile_size_bytes);

        // CRITICAL: Barrier immediately after write for safety
        noc_async_write_barrier();

        // Pop tile from circular buffer (frees space for reader)
        cb_pop_front(cb_id, 1);
    }
}
