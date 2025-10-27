// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Multi-core reader kernel for extract_attention_input
// PURPOSE: Read assigned tile range from input tensor
// Each core reads dp_degree once, calculates global offset, then reads its tile range

void kernel_main() {
    // Runtime arguments (unique per core)
    uint32_t input_addr = get_arg_val<uint32_t>(0);          // Input buffer address
    uint32_t dp_degree_addr = get_arg_val<uint32_t>(1);      // dp_degree buffer address
    uint32_t num_tiles = get_arg_val<uint32_t>(2);           // Number of tiles for this core
    uint32_t local_tile_offset = get_arg_val<uint32_t>(3);   // Offset within device's tiles
    uint32_t tiles_per_device = get_arg_val<uint32_t>(4);    // Total tiles per device

    // Compile-time arguments
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr bool dp_degree_is_dram = (bool)get_compile_time_arg_val(1);

    constexpr uint32_t cb_id = 0;       // Input circular buffer
    constexpr uint32_t cb_dp_degree = 1;  // dp_degree circular buffer

    // Get tile size from CB
    const uint32_t tile_bytes = get_tile_size(cb_id);

    // Read dp_degree value (4 bytes) to calculate global start offset
    // All cores on this device read the same dp_degree value
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

    // Release dp_degree CB (no longer needed)
    cb_push_back(cb_dp_degree, 1);
    cb_pop_front(cb_dp_degree, 1);

    // Calculate global start tile for this device
    // global_start_tile = dp_degree_value * tiles_per_device + local_tile_offset
    uint32_t global_start_tile = dp_degree_value * tiles_per_device + local_tile_offset;

    // Setup tensor accessor for input
    const auto accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // Read this core's assigned tiles from [global_start_tile, global_start_tile + num_tiles)
    uint32_t end_tile_id = global_start_tile + num_tiles;
    for (uint32_t tile_idx = global_start_tile; tile_idx < end_tile_id; ++tile_idx) {
        // Reserve space in CB
        cb_reserve_back(cb_id, 1);

        // Get L1 write address
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        // Read tile from DRAM to L1
        noc_async_read_tile(tile_idx, accessor, l1_write_addr);
        noc_async_read_barrier();

        // Push tile to CB (makes available to compute/writer)
        cb_push_back(cb_id, 1);
    }
}
