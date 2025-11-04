// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {

void MAIN {
    uint32_t Mt_max = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t BMt = get_arg_val<uint32_t>(3);
    uint32_t BNt = get_arg_val<uint32_t>(4);
    uint32_t BKt = get_arg_val<uint32_t>(5);
    uint32_t SBMt = get_arg_val<uint32_t>(6);
    uint32_t SBNt = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_metadata = tt::CBIndex::c_3;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_buffer = tt::CBIndex::c_24;

    mm_block_init(cb_input, cb_weights, cb_out_buffer, false, BNt, BMt, BKt);

    cb_wait_front(cb_metadata, 1);
    tensix_sync();
    uint32_t* metadata_ptr;
    cb_get_tile(cb_metadata, 0, &metadata_ptr);
    metadata_ptr += 4; // Need this offset!
    uint32_t Mt = metadata_ptr[0];
    uint32_t row_bidx0 = metadata_ptr[2];
    uint32_t col_bidx0 = metadata_ptr[4];
    uint32_t row_nblocks_per_core = metadata_ptr[1];
    uint32_t col_nblocks_per_core = metadata_ptr[3];
    cb_release_tile(cb_metadata);

    uint32_t num_ki_iterations = Kt / BKt;
    uint32_t num_output_tiles = BMt * BNt;
    uint32_t subblock_size = SBMt * SBNt;

    // Main loop.
    for (uint32_t block_idx_row = row_bidx0; block_idx_row < row_bidx0 + row_nblocks_per_core; block_idx_row++) {
        for (uint32_t block_idx_col = col_bidx0; block_idx_col < col_bidx0 + col_nblocks_per_core; block_idx_col++) {
            for (uint32_t ki_iter = 0; ki_iter < num_ki_iterations; ki_iter++) {
                bool is_first_ki = (ki_iter == 0);
                bool is_last_ki = (ki_iter == num_ki_iterations - 1);

                // Compute: Process all BMt × BNt output blocks in order
                // Wait for BMt × BKt tiles from A and BKt × BNt tiles from B
                cb_wait_front(cb_input, BMt * BKt);
                cb_wait_front(cb_weights, BKt * BNt);

                // Process one block (BMt x BNt) in unit of subblock (SBMt x
                // SBNt tiles)
                for (uint32_t h = 0; h < BMt; h += SBMt) {
                    for (uint32_t w = 0; w < BNt; w += SBNt) {
                        tile_regs_acquire();
                        // Reload: Read current accumulated value from Cbuffer
                        if (!is_first_ki) {
                            copy_tile_to_dst_init_short_with_dt(cb_weights, cb_out_buffer);
                            cb_wait_front(cb_out_buffer, subblock_size);
                            copy_block_matmul_partials(cb_out_buffer, 0, 0,
                                                       subblock_size);
                            cb_pop_front(cb_out_buffer, subblock_size);
                        }

                        // Compute one subblock. matmul_block handles (SBMt x
                        // 1) @ (1 x SBNt) matmul with accumulation.
                        // mm_block_init_short(cb_input, cb_weights, false, SBNt, SBMt,
                        //     BKt);
                        mm_block_init_short_with_dt(cb_input, cb_weights, cb_out_buffer, false, SBNt, SBMt,
                                            BKt);
                        for (uint32_t k = 0; k < BKt; k++) {
                            uint32_t input_cb_tile_idx = h * BKt + k;
                            uint32_t weights_cb_tile_idx = k * BNt + w;
                            matmul_block(cb_input, cb_weights, input_cb_tile_idx,
                                         weights_cb_tile_idx, 0, false, SBNt, SBMt,
                                         BKt);
                        }

                        tile_regs_commit();
                        tile_regs_wait();

                        // Spill/Write: Write updated result back to Cbuffer/C
                        uint32_t out_cb = is_last_ki ? cb_output : cb_out_buffer;
                        cb_reserve_back(out_cb, subblock_size);
                        PACK((pack_reconfig_data_format(out_cb)));
                        pack_tile_block(0, out_cb, subblock_size);
                        cb_push_back(out_cb, subblock_size);
                        tile_regs_release();
                    }
                }

                // Release A and B tiles after done
                cb_pop_front(cb_input, BMt * BKt);
                cb_pop_front(cb_weights, BKt * BNt);
            } // ki_iter loop
        } // block_idx_col loop
    } // block_idx_row loop
}

}  // namespace NAMESPACE
