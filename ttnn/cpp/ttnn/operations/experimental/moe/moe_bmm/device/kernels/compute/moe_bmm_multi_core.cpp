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
 * Core coordinates are obtained internally via get_relative_logical_x/y().
 *
 * Compile-time arguments:
 *   - Kt: Number of tiles in K dimension (reduction dimension)
 *   - Nt: Number of tiles in N dimension (output columns)
 *
 * Runtime arguments:
 *   - None (core coordinates obtained internally)
 *
 * Circular buffers:
 *   - cb_in0: Input tiles from input tensor
 *   - cb_in1: Weight tiles from weights tensor
 *   - cb_num_tiles: Total number of output tiles (from reader)
 *   - cb_out: Output tiles
 */
void MAIN {
    const uint32_t Kt = get_compile_time_arg_val(0);
    const uint32_t Nt = get_compile_time_arg_val(1);

    const uint32_t core_x = get_relative_logical_x();
    const uint32_t core_y = get_relative_logical_y();

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_num_tiles = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    volatile uint32_t* num_tiles_addr_ptr;
    cb_wait_front(cb_num_tiles, 1);
    tensix_sync();
    cb_get_tile(cb_num_tiles, 0, &num_tiles_addr_ptr);
    uint32_t total_token_tiles = num_tiles_addr_ptr[4];
    cb_release_tile(cb_num_tiles);
    cb_pop_front(cb_num_tiles, 1);

    const uint32_t num_output_tiles_total = total_token_tiles;
    constexpr uint32_t NUM_CORES = 64;
    const uint32_t core_id = core_y * 8 + core_x;
    uint32_t work_per_core = num_output_tiles_total / NUM_CORES;
    const uint32_t remainder = num_output_tiles_total % NUM_CORES;
    
    if (core_id < remainder) {
        work_per_core += 1;
    }

    mm_init(cb_in0, cb_in1, cb_out);

    // Process work_per_core output tiles
    for (uint32_t tile_idx = 0; tile_idx < work_per_core; tile_idx++) {
        tile_regs_acquire();

        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        PACK((pack_reconfig_data_format(cb_out)));
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}

}
