#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

#include "moe_bmm_dataflow.hpp"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t weights_addr = get_arg_val<uint32_t>(1);
    uint32_t num_routed_addr = get_arg_val<uint32_t>(2);
    uint32_t num_experts = get_arg_val<uint32_t>(3);
    uint32_t max_tokens = get_arg_val<uint32_t>(4);
    uint32_t h_in = get_arg_val<uint32_t>(5);
    uint32_t h_out = get_arg_val<uint32_t>(6);
    uint32_t Kt = get_arg_val<uint32_t>(7);
    uint32_t Nt = get_arg_val<uint32_t>(8);
    uint32_t Mt_max = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_num_tiles = 2;
    constexpr uint32_t cb_num_routed = 3;
    constexpr uint8_t noc = noc_index;

    uint32_t core_x = (uint32_t)my_x[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t core_y = (uint32_t)my_y[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;

    // Create tensor accessors for DRAM buffers
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_accessor = TensorAccessor(input_args, input_addr, get_tile_size(cb_in0));
    
    constexpr auto weights_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, get_tile_size(cb_in1));
    
    // num_routed_tokens is 1D tensor (E/D,)
    constexpr auto num_routed_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, num_experts * sizeof(uint32_t));

    // Semaphores for multicast
    constexpr uint32_t sender_sem = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_sem = get_compile_time_arg_val(4);

    // Calculate strides in tiles
    // Input: (E/D, T, H_in) in tiles -> (num_experts, Mt_max, Kt)
    uint32_t input_expert_stride = Mt_max * Kt;  // Tiles per expert
    uint32_t input_row_stride = Kt;              // Tiles per token row

    // Weights: (E/D, H_in, H_out) in tiles -> (num_experts, Kt, Nt)
    uint32_t weights_expert_stride = Kt * Nt;    // Tiles per expert weight matrix
    uint32_t weights_row_stride = Nt;            // Tiles per weight row

    // Cache for Mt (number of token tiles) per expert
    uint32_t num_tiled[num_experts];  // Number of token tiles per expert
    uint32_t total_tiles = 0;

    // Multicast num_routed_tokens: Only core (0,0) reads from DRAM, broadcasts to all cores
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
              0, 7,           // x range: 0-7
              0, 7,           // y range: 0-7
              sender_sem, receiver_sem, noc);
    cb_push_back(cb_num_routed, 1);

    // Access num_routed_tokens from CB
    volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    // Calculate Mt per expert from the loaded array
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        uint32_t num_routed_value = num_routed_ptr[expert_idx];
        num_tiled[expert_idx] = (num_routed_value + 31) / 32;
        total_tiles += num_tiled[expert_idx];
    }

    uint32_t num_output_tiles_total = total_tiles * Nt;

    cb_reserve_back(cb_num_tiles, 1);
    auto num_tiles_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_num_tiles));
    num_tiles_ptr[0] = num_output_tiles_total;
    cb_push_back(cb_num_tiles, 1);

    // Calculate work for this core dynamically
    constexpr uint32_t NUM_CORES = 64;
    uint32_t core_id = core_y * 8 + core_x;
    uint32_t work_per_core = num_output_tiles_total / NUM_CORES;
    uint32_t remainder = num_output_tiles_total % NUM_CORES;
    
    uint32_t work_offset;
    if (core_id < remainder) {
        work_per_core += 1;
        work_offset = core_id * work_per_core;
    } else {
        work_offset = remainder * (work_per_core + 1) + (core_id - remainder) * work_per_core;
    }

    // Calculate the cumulative tile offset for each expert
    uint32_t expert_tile_offsets[num_experts + 1];  // Need num_experts + 1 entries
    expert_tile_offsets[0] = 0;

    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        expert_tile_offsets[expert_idx + 1] = expert_tile_offsets[expert_idx] + num_tiled[expert_idx] * Nt;
    }

    // Process work_per_core output tiles starting from work_offset
    for (uint32_t work_idx = 0; work_idx < work_per_core; work_idx++) {
        uint32_t global_output_tile_id = work_offset + work_idx;
        // Find which expert this tile belongs to
        uint32_t expert_idx = num_experts - 1;
        for (uint32_t e = 0; e < num_experts; e++) {
            if (global_output_tile_id < expert_tile_offsets[e + 1]) {
                expert_idx = e;
                break;
            }
        }
        
        // Calculate local tile position within expert's output
        uint32_t expert_local_tile = global_output_tile_id - expert_tile_offsets[expert_idx];
        uint32_t mt = expert_local_tile / Nt;
        uint32_t nt = expert_local_tile % Nt;
        
        // Calculate base tile indices for this expert
        uint32_t input_expert_base = expert_idx * input_expert_stride;
        uint32_t weights_expert_base = expert_idx * weights_expert_stride;
        
        for (uint32_t kt = 0; kt < Kt; kt++) {
            uint32_t input_tile_idx = input_expert_base + mt * input_row_stride + kt;
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_in0);
            noc_async_read_tile(input_tile_idx, input_accessor, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);

            uint32_t weight_tile_idx = weights_expert_base + kt * weights_row_stride + nt;
            cb_reserve_back(cb_in1, 1);
            uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
            noc_async_read_tile(weight_tile_idx, weights_accessor, l1_write_addr_in1);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}
