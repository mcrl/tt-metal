// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// Reader kernel for extract_attention_input (unified)
// PURPOSE: Read batch slice from replicated input for this device
// APPROACH: Safety-first - one tile at a time, barrier after each read
// MODES: Auto-detects prefill ([B,S,H]) vs decode ([1,1,B,H]) from program factory

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);        // Input buffer address
    uint32_t dp_degree_addr = get_arg_val<uint32_t>(1);    // dp_degree buffer address

    // Compile-time arguments
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(0);       // Input buffer type
    constexpr bool dp_degree_is_dram = (bool)get_compile_time_arg_val(1);   // dp_degree buffer type
    constexpr uint32_t tiles_per_device = get_compile_time_arg_val(2);      // Number of tiles to read
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(3);         // Number of tile columns
    constexpr uint32_t batch_size = get_compile_time_arg_val(4);            // Total batch size
    constexpr uint32_t seq_len = get_compile_time_arg_val(5);               // Sequence length

    // Constants
    constexpr uint32_t tile_size_bytes = 2048;  // 32 * 32 * 2 bytes (bfloat16)
    constexpr uint32_t cb_id = 0;               // Circular buffer index
    constexpr uint32_t cb_dp_degree = 1;        // Circular buffer for dp_degree

    // Read dp_degree scalar value from buffer
    // Reserve space in CB to read dp_degree
    cb_reserve_back(cb_dp_degree, 1);
    uint32_t dp_degree_l1_addr = get_write_ptr(cb_dp_degree);

    // Setup address generator for dp_degree buffer (ROW_MAJOR layout, single uint32 value)
    const InterleavedAddrGenFast<dp_degree_is_dram> dp_degree_addrgen = {
        .bank_base_address = dp_degree_addr,
        .page_size = sizeof(uint32_t),
        .data_format = DataFormat::UInt32
    };

    // Read dp_degree value (4 bytes)
    uint64_t dp_degree_noc_addr = get_noc_addr(0, dp_degree_addrgen);
    noc_async_read(dp_degree_noc_addr, dp_degree_l1_addr, sizeof(uint32_t));
    noc_async_read_barrier();

    // Get the dp_degree value
    volatile tt_l1_ptr uint32_t* dp_degree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dp_degree_l1_addr);
    uint32_t dp_degree_value = dp_degree_ptr[0];

    // Release the CB
    cb_push_back(cb_dp_degree, 1);
    cb_pop_front(cb_dp_degree, 1);

    // Calculate starting tile index based on dp_degree
    uint32_t start_tile_idx = dp_degree_value * tiles_per_device;

    // Setup address generator for interleaved buffer
    const InterleavedAddrGenFast<input_is_dram> addrgen = {
        .bank_base_address = input_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b  // bfloat16
    };

    DPRINT << "Reader: dp_degree=" << dp_degree_value << " tiles_per_device=" << tiles_per_device << " start_tile_idx=" << start_tile_idx << ENDL();
    // SAFE PATTERN: Read one tile at a time, barrier immediately after each read
    // No batching for maximum safety
    for (uint32_t i = 0; i < tiles_per_device; i++) {
        // Reserve space in circular buffer for one tile
        cb_reserve_back(cb_id, 1);

        // Get L1 address to write to
        uint32_t l1_addr = get_write_ptr(cb_id);

        // Calculate global tile index
        uint32_t tile_idx = start_tile_idx + i;

        // Get NOC address for this tile
        uint64_t noc_addr = get_noc_addr(tile_idx, addrgen);

        // Issue async read from DRAM to L1
        noc_async_read(noc_addr, l1_addr, tile_size_bytes);

        // CRITICAL: Barrier immediately after read for safety
        noc_async_read_barrier();

        // Push tile to circular buffer (makes it available to writer)
        cb_push_back(cb_id, 1);
    }
}
