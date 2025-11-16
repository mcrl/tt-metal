#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

#include "moe_bmm_dataflow.hpp"

#define PH 8
#define PW 8
constexpr uint32_t A_MASTER_COL = 0;
constexpr uint32_t B_MASTER_ROW = 0;

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t weights_addr = get_arg_val<uint32_t>(1);
    uint32_t num_routed_addr = get_arg_val<uint32_t>(2);
    uint32_t output_addr = get_arg_val<uint32_t>(3);
    uint32_t num_experts = get_arg_val<uint32_t>(4);
    uint32_t ph = get_arg_val<uint32_t>(5);
    uint32_t pw = get_arg_val<uint32_t>(6);
    uint32_t Mt_max = get_arg_val<uint32_t>(7);
    uint32_t Nt = get_arg_val<uint32_t>(8);
    uint32_t Kt = get_arg_val<uint32_t>(9);
    uint32_t BMt = get_arg_val<uint32_t>(10);
    uint32_t BNt = get_arg_val<uint32_t>(11);
    uint32_t BKt = get_arg_val<uint32_t>(12);
    uint32_t SBMt = get_arg_val<uint32_t>(13);
    uint32_t SBNt = get_arg_val<uint32_t>(14);
    uint32_t sender_sem = get_arg_val<uint32_t>(15);
    uint32_t receiver_sem = get_arg_val<uint32_t>(16);
    uint32_t input_master_sem = get_arg_val<uint32_t>(17);
    uint32_t input_slave_sem = get_arg_val<uint32_t>(18);
    uint32_t weight_master_sem = get_arg_val<uint32_t>(19);
    uint32_t weight_slave_sem = get_arg_val<uint32_t>(20);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_num_routed = tt::CBIndex::c_2;
    constexpr uint32_t cb_metadata = tt::CBIndex::c_3;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint8_t noc = noc_index;

    uint32_t core_x = (uint32_t)my_x[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t core_y = (uint32_t)my_y[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;

    uint32_t my_expert;
    uint32_t cores_per_expert = (PH * PW) / num_experts;
    uint32_t my_rank = core_y * PH + core_x;
    uint32_t my_rank_in_expert = my_rank % cores_per_expert;
    uint32_t ph_per_expert;
    uint32_t pw_per_expert;
    uint32_t expert_master_y;
    uint32_t expert_master_x;
    uint32_t experts_per_row = cores_per_expert >= PW ? 1 : num_experts / PH;

    if (cores_per_expert >= PW) { // multiple core rows per expert
        ph_per_expert = cores_per_expert / PW;
        my_expert = core_y / ph_per_expert;
        pw_per_expert = PW;
        expert_master_y = my_expert * ph_per_expert;
        expert_master_x = 0;
    }
    else { // one core row per expert. Multiple experts can share a core row
        ph_per_expert = 1;
        pw_per_expert = cores_per_expert;
        my_expert = core_y * experts_per_row + core_x / pw_per_expert;
        expert_master_y = my_expert / experts_per_row;
        expert_master_x = (my_expert % experts_per_row) * pw_per_expert;
    }

    // Create tensor accessors for DRAM buffers
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_tensor = TensorAccessor(input_args, input_addr, get_tile_size(cb_input));

    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto weights_tensor = TensorAccessor(weights_args, weights_addr, get_tile_size(cb_weights));

    // num_routed_tokens is 1D tensor (E/D,)
    constexpr auto num_routed_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, num_experts * sizeof(uint32_t));

    // Output tensor accessor
    constexpr auto output_args = TensorAccessorArgs<num_routed_args.next_compile_time_args_offset()>();
    const auto output_tensor = TensorAccessor(output_args, output_addr, get_tile_size(cb_output));

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
              0, 0,
              PH - 1, PW - 1,
              sender_sem, receiver_sem, noc);
    cb_push_back(cb_num_routed, 1);
    volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    uint32_t Mt = (num_routed_ptr[my_expert] + 31) / 32;
    uint32_t row_nblocks_per_core = Mt / BMt;
    uint32_t row_bidx0 = 0;
    uint32_t col_nblocks_per_core = Nt / BNt / cores_per_expert;
    uint32_t col_bidx0 = my_rank_in_expert * col_nblocks_per_core;

    // Uneven col block distribution. First few cores get remainder col blocks
    if (Nt / BNt % cores_per_expert != 0) {
        if (my_rank_in_expert < Nt / BNt % cores_per_expert) {
            col_nblocks_per_core += 1;
            col_bidx0 += my_rank_in_expert;
        }
        else {
            col_bidx0 += Nt / BNt % cores_per_expert;
        }
    }

    uint32_t num_ki_iterations = Kt / BKt;
    uint32_t input_cb_tile_size = get_tile_size(cb_input);
    uint32_t output_tile_size = get_tile_size(cb_output);

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

    // Main loop: read input and write output
    uint32_t block_idx_row = row_bidx0;
    uint32_t block_idx_col = col_bidx0;
    
    uint32_t C_noc = core_x >= 3 ? 0 : 1;
    for (; block_idx_row < row_bidx0 + row_nblocks_per_core; block_idx_row++) {
        for (block_idx_col = col_bidx0; block_idx_col < col_bidx0 + col_nblocks_per_core; block_idx_col++) {
            for (uint32_t ki_iter = 0; ki_iter < num_ki_iterations; ki_iter++) {
                // Read input tiles
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
            }

            // Write output tiles (after all K-iterations for this block)
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
                                                 l1_read_addr, C_noc);
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
