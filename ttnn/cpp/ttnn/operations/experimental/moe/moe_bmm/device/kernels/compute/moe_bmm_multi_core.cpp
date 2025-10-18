// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

namespace NAMESPACE {

/**
 * @brief Multi-core compute kernel for MoE batched matrix multiplication.
 *
 * This kernel performs batched matrix multiplication for multiple experts in a multi-core setup.
 * Each core processes a subset of output tiles assigned by the work distribution logic.
 *
 * Compile-time arguments:
 *   - Kt: Number of tiles in K dimension (reduction dimension)
 *   - Nt: Number of tiles in N dimension (output columns)
 *
 * Runtime arguments:
 *   - work_per_core: Number of output tiles this core should process
 *
 * Circular buffers:
 *   - cb_in0: Input tiles from input tensor
 *   - cb_in1: Weight tiles from weights tensor
 *   - cb_out: Output tiles
 */
void MAIN {
    const uint32_t Kt = get_compile_time_arg_val(0);
    const uint32_t Nt = get_compile_time_arg_val(1);
    
    const uint32_t work_per_core = get_arg_val<uint32_t>(0);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Initialize matrix multiplication unit
    mm_init(cb_in0, cb_in1, cb_out);

    // Process work_per_core output tiles
    // The dataflow kernel provides tiles in the order: expert0 tiles, expert1 tiles, etc.
    // For each output tile, it provides Kt pairs of (input, weight) tiles
    
    for (uint32_t tile_idx = 0; tile_idx < work_per_core; tile_idx++) {
        // Acquire destination registers and zero them for accumulation
        tile_regs_acquire();
        
        // Accumulate over K dimension
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Wait for input tiles
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            
            // Perform tile matrix multiplication and accumulate
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
            
            // Pop input tiles
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        
        // Commit and wait for FPU operations to complete
        tile_regs_commit();
        tile_regs_wait();
        
        // Reserve output buffer and pack result
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        
        // Release destination registers
        tile_regs_release();
    }
}

}  // namespace NAMESPACE
