// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

namespace NAMESPACE {

/**
 * @brief Compute kernel for MoE batched matrix multiplication.
 *
 * This kernel performs batched matrix multiplication for multiple experts.
 * For each expert, it computes: output[expert, :, :] = input[expert, :, :] @ weights[expert, :, :]
 *
 * Compile-time arguments:
 *   - num_experts: Number of experts to process
 *   - Mt_max: Maximum number of output tile rows per expert
 *   - Kt: Number of tiles in K dimension (reduction dimension)
 *   - Nt: Number of tiles in N dimension (output columns)
 *
 * Circular buffers:
 *   - cb_in0: Input tiles from input tensor
 *   - cb_in1: Weight tiles from weights tensor
 *   - cb_out: Output tiles
 *
 * The dataflow kernel provides exactly Mt * Nt * Kt tile pairs per expert
 * where Mt is determined from num_routed_tokens.
 */
void MAIN {
    const uint32_t num_experts = get_compile_time_arg_val(0);
    const uint32_t Mt_max = get_compile_time_arg_val(1);
    const uint32_t Kt = get_compile_time_arg_val(2);
    const uint32_t Nt = get_compile_time_arg_val(3);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_num_rows = tt::CBIndex::c_3;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    volatile uint32_t* num_rows_addr_ptr;

    // Initialize matrix multiplication unit
    mm_init(cb_in0, cb_in1, cb_out);

    // Process all experts
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        // For each expert, process Mt_max * Nt output tiles
        // (dataflow only sends tiles for active Mt rows based on num_routed_tokens)

        cb_wait_front(cb_num_rows, 1);
        tensix_sync();
        cb_get_tile(cb_num_rows, 0, &num_rows_addr_ptr);
        uint32_t num_tokens = num_rows_addr_ptr[4];
        cb_release_tile(cb_num_rows);
        cb_pop_front(cb_num_rows, 1);

        uint32_t Mt = (num_tokens + 32 - 1) / 32;

        for (uint32_t mt = 0; mt < Mt; mt++) {
            for (uint32_t nt = 0; nt < Nt; nt++) {
                // Acquire destination register for accumulation
                acquire_dst();
                
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
                
                // Reserve output buffer and pack result
                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);
                
                // Release destination register
                release_dst();
            }
        }
    }
}

}  // namespace NAMESPACE

