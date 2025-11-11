#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize SFPU and copy operations
    init_sfpu(cb_in, cb_out);
    copy_tile_init(cb_in);

    // Process assigned tiles
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        cb_wait_front(cb_in, 1);

        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        copy_tile(cb_in, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
} 