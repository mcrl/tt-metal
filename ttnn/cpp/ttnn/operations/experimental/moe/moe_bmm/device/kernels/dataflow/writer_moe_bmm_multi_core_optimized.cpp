#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

#include "moe_bmm_dataflow.hpp"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_experts = get_arg_val<uint32_t>(1);
    uint32_t ph = get_arg_val<uint32_t>(2);
    uint32_t pw = get_arg_val<uint32_t>(3);
    uint32_t Mt_max = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);
    uint32_t Kt = get_arg_val<uint32_t>(6);
    uint32_t BMt = get_arg_val<uint32_t>(7);
    uint32_t BNt = get_arg_val<uint32_t>(8);
    uint32_t BKt = get_arg_val<uint32_t>(9);
    uint32_t SBMt = get_arg_val<uint32_t>(10);
    uint32_t SBNt = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_metadata = tt::CBIndex::c_3;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    cb_wait_front(cb_metadata, 1);
    uint32_t* metadata_ptr = reinterpret_cast<uint32_t*>(get_read_ptr(cb_metadata));
    uint32_t Mt = metadata_ptr[0];
    uint32_t row_bidx0 = metadata_ptr[2];
    uint32_t col_bidx0 = metadata_ptr[4];
    uint32_t row_nblocks_per_core = metadata_ptr[1];
    uint32_t col_nblocks_per_core = metadata_ptr[3];
    uint32_t my_expert = metadata_ptr[5];

    uint32_t output_tile_size = get_tile_size(cb_output);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_tensor = TensorAccessor(output_args, output_addr, output_tile_size);


    for (uint32_t block_idx_row = row_bidx0;
        block_idx_row < row_bidx0 + row_nblocks_per_core; block_idx_row++) {
       for (uint32_t block_idx_col = col_bidx0;
            block_idx_col < col_bidx0 + col_nblocks_per_core; block_idx_col++) {
           // Write BMt × BNt tiles from compute to DRAM
            cb_wait_front(cb_output, BMt * BNt);
           uint32_t l1_read_addr = get_read_ptr(cb_output);

           for (uint32_t bh = 0; bh < BMt; bh += SBMt) {
               for (uint32_t bw = 0; bw < BNt; bw += SBNt) {
                   for (uint32_t h = 0; h < SBMt; h++) {
                       for (uint32_t w = 0; w < SBNt; w++) {
                           uint32_t output_tile_row = block_idx_row * BMt + bh + h;
                           uint32_t output_tile_col = block_idx_col * BNt + bw + w;
                           uint32_t output_tile_index =
                               my_expert * Nt * Mt_max + 
                               output_tile_row * Nt + output_tile_col;
                           noc_async_write_tile(output_tile_index, output_tensor,
                                                l1_read_addr);
                           l1_read_addr += output_tile_size;
                       }
                   }
               }
           }
           noc_async_write_barrier();
           cb_pop_front(cb_output, BMt * BNt);
       }
   }
}
