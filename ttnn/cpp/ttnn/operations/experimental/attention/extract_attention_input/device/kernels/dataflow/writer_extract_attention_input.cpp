// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// Multi-core writer kernel for extract_attention_input
// PURPOSE: Write assigned tile range to output tensor
// PATTERN: Matches writer_unary_interleaved_start_id.cpp
// SUPPORTS: Both bfloat16 and bfp8_b output formats

void kernel_main() {
    // Runtime arguments (unique per core)
    uint32_t dst_addr = get_arg_val<uint32_t>(0);        // Output buffer address
    uint32_t num_tiles = get_arg_val<uint32_t>(1);       // Number of tiles for this core
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Starting tile index

    // Compile-time arguments
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);  // CB index (0 or 16)
    constexpr auto dst_args = TensorAccessorArgs<1>();       // TensorAccessor args
    constexpr uint32_t output_data_format = get_compile_time_arg_val(2);  // Output data format

    // Get tile size from CB (handles both bfloat16 and bfp8_b)
    const uint32_t tile_bytes = get_tile_size(cb_id);

    // Setup tensor accessor
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Write tiles from [start_tile_id, start_tile_id + num_tiles)
    uint32_t end_tile_id = start_tile_id + num_tiles;
    for (uint32_t tile_idx = start_tile_id; tile_idx < end_tile_id; ++tile_idx) {
        // Wait for tile from CB
        cb_wait_front(cb_id, 1);

        // Get L1 read address
        uint32_t l1_read_addr = get_read_ptr(cb_id);

        // Write tile from L1 to DRAM
        noc_async_write_tile(tile_idx, accessor, l1_read_addr);
        noc_async_write_barrier();

        // Pop tile from CB
        cb_pop_front(cb_id, 1);
    }
}
