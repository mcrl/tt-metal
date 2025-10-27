// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"

// Multi-core compute kernel for extract_attention_input format conversion
// PURPOSE: Convert tiles from bfloat16 to bfp8_b using hardware SFPU
// APPROACH: Simple tile-by-tile processing (runtime arg for num_tiles)
//
// CB  0: Input tiles  (bfloat16)
// CB 16: Output tiles (bfp8_b)

namespace NAMESPACE {
void MAIN {
    // Runtime argument (unique per core)
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // Number of tiles for this core

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize SFPU and copy operations
    init_sfpu(cb_in, cb_out);
    copy_tile_init(cb_in);

    // Process assigned tiles one at a time
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Wait for input tile
        cb_wait_front(cb_in, 1);

        // Reserve output buffer
        cb_reserve_back(cb_out, 1);

        // Acquire tile registers
        tile_regs_acquire();

        // Copy tile from input CB to DST register 0
        copy_tile(cb_in, 0, 0);

        // Initialize typecast
        // typecast_tile_init();

        // Perform format conversion: bfloat16 → bfp8_b
        // TYPECAST_LLK(0);

        // Commit and wait
        tile_regs_commit();
        tile_regs_wait();

        // Pack result to output CB
        pack_tile(0, cb_out);

        // Release tile registers
        tile_regs_release();

        // Free buffers
        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
}  // namespace NAMESPACE
