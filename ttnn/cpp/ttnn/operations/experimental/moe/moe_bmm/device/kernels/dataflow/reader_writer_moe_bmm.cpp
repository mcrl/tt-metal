// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t weights_addr = get_arg_val<uint32_t>(1);
    uint32_t num_routed_addr = get_arg_val<uint32_t>(2);
    uint32_t output_addr = get_arg_val<uint32_t>(3);
    uint32_t num_experts = get_arg_val<uint32_t>(4);
    uint32_t max_tokens = get_arg_val<uint32_t>(5);
    uint32_t h_in = get_arg_val<uint32_t>(6);
    uint32_t h_out = get_arg_val<uint32_t>(7);
    uint32_t Kt = get_arg_val<uint32_t>(8);
    uint32_t Nt = get_arg_val<uint32_t>(9);
    uint32_t Mt_max = get_arg_val<uint32_t>(10);

    // Circular buffer indices
    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_num_routed = 2;
    constexpr uint32_t cb_num_rows = 3;
    constexpr uint32_t cb_out = 16;

    // Create tensor accessors for DRAM buffers
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_accessor = TensorAccessor(input_args, input_addr, get_tile_size(cb_in0));
    
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, get_tile_size(cb_in1));
    
    constexpr auto num_routed_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, sizeof(uint32_t));
    
    constexpr auto output_args = TensorAccessorArgs<num_routed_args.next_compile_time_args_offset()>();
    const auto output_accessor = TensorAccessor(output_args, output_addr, get_tile_size(cb_out));

    // Calculate strides in tiles
    // Input: (E/D, T, H_in) in tiles -> (num_experts, Mt_max, Kt)
    uint32_t input_expert_stride = Mt_max * Kt;  // Tiles per expert
    uint32_t input_row_stride = Kt;              // Tiles per token row

    // Weights: (E/D, H_in, H_out) in tiles -> (num_experts, Kt, Nt)
    uint32_t weights_expert_stride = Kt * Nt;    // Tiles per expert weight matrix
    uint32_t weights_row_stride = Nt;            // Tiles per weight row

    // Output: (E/D, T, H_out) in tiles -> (num_experts, Mt_max, Nt)
    uint32_t output_expert_stride = Mt_max * Nt; // Tiles per expert output
    uint32_t output_row_stride = Nt;             // Tiles per output row

    // Process each expert sequentially
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        // Read num_routed_tokens[expert_idx, 0]
        // This is a single uint32 value
        cb_reserve_back(cb_num_routed, 1);
        uint32_t l1_write_addr_num_routed = get_write_ptr(cb_num_routed);
        noc_async_read(
            num_routed_accessor.get_noc_addr(expert_idx),
            l1_write_addr_num_routed,
            sizeof(uint32_t));
        noc_async_read_barrier();
        
        // Read the value from L1
        volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr_num_routed);
        uint32_t num_routed = num_routed_ptr[0];

        cb_push_back(cb_num_routed, 1);
        cb_pop_front(cb_num_routed, 1);

        cb_reserve_back(cb_num_rows, 1);
        auto num_rows_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_num_rows));
        num_rows_ptr[0] = num_routed;
        cb_push_back(cb_num_rows, 1);

        // Calculate number of active tile rows for this expert
        // Round up to nearest tile boundary (TILE_HEIGHT = 32)
        constexpr uint32_t TILE_HEIGHT = 32;
        uint32_t Mt = (num_routed + TILE_HEIGHT - 1) / TILE_HEIGHT;

        // Calculate base tile indices for this expert
        uint32_t input_expert_base = expert_idx * input_expert_stride;
        uint32_t weights_expert_base = expert_idx * weights_expert_stride;
        uint32_t output_expert_base = expert_idx * output_expert_stride;

        // Perform batched matmul for this expert
        // Process ALL Mt rows to keep compute kernel synchronized
        for (uint32_t mt = 0; mt < Mt; mt++) {        
            for (uint32_t nt = 0; nt < Nt; nt++) {
                // Inner loop over K dimension
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // Read input tile at (expert, mt, kt)
                    uint32_t input_tile_idx = input_expert_base + mt * input_row_stride + kt;
                    cb_reserve_back(cb_in0, 1);
                    uint32_t l1_write_addr_in0 = get_write_ptr(cb_in0);
                    noc_async_read_tile(input_tile_idx, input_accessor, l1_write_addr_in0);
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);

                    // Read weight tile at (expert, kt, nt) - always needed for compute sync
                    uint32_t weight_tile_idx = weights_expert_base + kt * weights_row_stride + nt;
                    cb_reserve_back(cb_in1, 1);
                    uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
                    noc_async_read_tile(weight_tile_idx, weights_accessor, l1_write_addr_in1);
                    noc_async_read_barrier();
                    cb_push_back(cb_in1, 1);
                }

                // Write output tile at (expert, mt, nt)
                uint32_t output_tile_idx = output_expert_base + mt * output_row_stride + nt;
                cb_wait_front(cb_out, 1);
                uint32_t l1_read_addr_out = get_read_ptr(cb_out);
                noc_async_write_tile(output_tile_idx, output_accessor, l1_read_addr_out);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}