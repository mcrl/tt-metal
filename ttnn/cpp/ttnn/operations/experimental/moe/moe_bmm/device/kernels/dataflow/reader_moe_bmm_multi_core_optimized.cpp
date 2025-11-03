// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

#include "moe_bmm_dataflow.hpp"

#define PH 8
#define PW 8

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t weights_addr = get_arg_val<uint32_t>(1);
    uint32_t num_routed_addr = get_arg_val<uint32_t>(2);
    uint32_t num_experts = get_arg_val<uint32_t>(3);
    uint32_t ph = get_arg_val<uint32_t>(4);
    uint32_t pw = get_arg_val<uint32_t>(5);
    uint32_t Mt_max = get_arg_val<uint32_t>(6);
    uint32_t Nt = get_arg_val<uint32_t>(7);
    uint32_t Kt = get_arg_val<uint32_t>(8);
    uint32_t BMt = get_arg_val<uint32_t>(9);
    uint32_t BNt = get_arg_val<uint32_t>(10);
    uint32_t BKt = get_arg_val<uint32_t>(11);
    uint32_t sender_sem = get_arg_val<uint32_t>(12);
    uint32_t receiver_sem = get_arg_val<uint32_t>(13);

    // Circular buffer indices
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_num_routed = tt::CBIndex::c_2;
    constexpr uint32_t cb_metadata = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint8_t noc = noc_index;

    // Get core coordinates from NOC
    uint32_t core_x = (uint32_t)my_x[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t core_y = (uint32_t)my_y[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;

    uint32_t my_expert = 0;
    uint32_t cores_per_expert = (PH * PW) / num_experts;
    uint32_t my_rank = core_x * PW + core_y;
    uint32_t my_rank_in_expert = my_rank % cores_per_expert;
    if (num_experts <= PH) {
        my_expert = core_x / (PH / num_experts);
    }
    else {
        uint32_t experts_per_row = num_experts / PH;
        my_expert = core_x * experts_per_row + core_y / (PW / experts_per_row);
    }

    // Create tensor accessors for DRAM buffers
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_tensor = TensorAccessor(input_args, input_addr, get_tile_size(cb_input));
    
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto weights_tensor = TensorAccessor(weights_args, weights_addr, get_tile_size(cb_weights));
    
    // num_routed_tokens is 1D tensor (E/D,) - all elements in one page
    constexpr auto num_routed_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, num_experts * sizeof(uint32_t));

    // Read num_routed_tokens - broadcast to all cores - get Mt value for my expert
    cb_reserve_back(cb_num_routed, 1);
    uint64_t num_routed_l1_addr = get_write_ptr(cb_num_routed);

    if (core_x == 0 && core_y == 0) {
        // Core (0,0): Read from DRAM
        uint64_t num_routed_base_addr = num_routed_accessor.get_noc_addr(0);  // Page 0
        noc_async_read(num_routed_base_addr, num_routed_l1_addr, num_experts * sizeof(uint32_t));
        noc_async_read_barrier();
    }
    broadcast(num_routed_l1_addr, num_experts * sizeof(uint32_t),
              0, 0,           // sender at (0, 0)
              0, PH - 1,           // x range: 0-7
              0, PW - 1,           // y range: 0-7
              sender_sem, receiver_sem, noc);
    cb_push_back(cb_num_routed, 1);
    volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    uint32_t Mt = (num_routed_ptr[my_expert] + 31) / 32;
    uint32_t row_nblocks_per_core = Mt / BMt;
    uint32_t row_bidx0 = 0;
    uint32_t col_nblocks_per_core = Nt / BNt / cores_per_expert;
    uint32_t col_bidx0 = my_rank_in_expert * col_nblocks_per_core;

    uint32_t num_ki_iterations = Kt / BKt;
    uint32_t input_cb_tile_size = get_tile_size(cb_input);
    uint32_t weights_cb_tile_size = get_tile_size(cb_weights);

    cb_reserve_back(cb_metadata, 1);
    uint64_t metadata_l1_addr = get_write_ptr(cb_metadata);
    uint32_t* metadata_ptr = reinterpret_cast<uint32_t*>(metadata_l1_addr);
    metadata_ptr[0] = Mt;
    metadata_ptr[1] = row_nblocks_per_core;
    metadata_ptr[2] = row_bidx0;
    metadata_ptr[3] = col_nblocks_per_core;
    metadata_ptr[4] = col_bidx0;
    metadata_ptr[5] = my_expert;
    cb_push_back(cb_metadata, 1);

    // DPRINT << "row_bidx0: " << row_bidx0 << " col_bidx0: " << col_bidx0 << " row_nblocks_per_core: " << row_nblocks_per_core << " col_nblocks_per_core: " << col_nblocks_per_core << ENDL();

    // Main loop.
    for (uint32_t block_idx_row = row_bidx0; block_idx_row < row_bidx0 + row_nblocks_per_core; block_idx_row++) {
        for (uint32_t block_idx_col = col_bidx0; block_idx_col < col_bidx0 + col_nblocks_per_core; block_idx_col++) {
            for (uint32_t ki_iter = 0; ki_iter < num_ki_iterations; ki_iter++) {
                /************ Read input block ************/
                cb_reserve_back(cb_input, BMt * BKt);
                uint32_t input_l1_ptr = get_write_ptr(cb_input);
                for (uint32_t h = 0; h < BMt; h++) {
                    for (uint32_t ki = 0; ki < BKt; ki++) {
                        uint32_t et_global = my_expert;
                        uint32_t mt_global = block_idx_row * BMt + h;
                        uint32_t kt_global = ki_iter * BKt + ki;
                        uint32_t input_tile_index = et_global * Mt_max * Kt + mt_global * Kt + kt_global;
                        noc_async_read_tile(input_tile_index, input_tensor, input_l1_ptr);
                        input_l1_ptr += input_cb_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_input, BMt * BKt);

                /************ Read weights block ************/
                cb_reserve_back(cb_weights, BKt * BNt);
                uint32_t weights_l1_ptr = get_write_ptr(cb_weights);
                for (uint32_t ki = 0; ki < BKt; ki++) {
                    for (uint32_t w = 0; w < BNt; w++) {
                        uint32_t et_global = my_expert;
                        uint32_t kt_global = ki_iter * BKt + ki;
                        uint32_t nt_global = block_idx_col * BNt + w;
                        uint32_t weights_tile_index = et_global * Kt * Nt + kt_global * Nt + nt_global;
                        noc_async_read_tile(weights_tile_index, weights_tensor, weights_l1_ptr);
                        weights_l1_ptr += weights_cb_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_weights, BKt * BNt);
            } // ki_iter loop
        } // block_idx_col loop
    } // block_idx_row loop

    cb_pop_front(cb_metadata, 1);
    cb_pop_front(cb_num_routed, 1);
}
