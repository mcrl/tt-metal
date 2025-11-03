// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_routed_addr = get_arg_val<uint32_t>(1);
    uint32_t num_experts = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    uint32_t Mt_max = get_arg_val<uint32_t>(4);
    uint32_t core_x = get_arg_val<uint32_t>(5);
    uint32_t core_y = get_arg_val<uint32_t>(6);

    // Circular buffer index for output
    constexpr uint32_t cb_out = 16;

    // Create tensor accessors for DRAM buffers
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_accessor = TensorAccessor(output_args, output_addr, get_tile_size(cb_out));
    
    // num_routed_tokens is 1D tensor (E/D,) - all elements in one page
    constexpr auto num_routed_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, num_experts * sizeof(uint32_t));

    // Output: (E/D, T, H_out) in tiles -> (num_experts, Mt_max, Nt)
    uint32_t output_expert_stride = Mt_max * Nt; // Tiles per expert output
    uint32_t output_row_stride = Nt;             // Tiles per output row

    uint32_t num_tiled[num_experts];  // Number of token tiles per expert
    uint32_t total_tiles = 0;

    // Read entire num_routed_tokens tensor once (1D tensor with all num_experts elements)
    uint64_t temp_l1_addr = get_write_ptr(cb_out);  // Use cb_out space temporarily
    uint64_t num_routed_base_addr = num_routed_accessor.get_noc_addr(0);  // Page 0
    noc_async_read(num_routed_base_addr, temp_l1_addr, num_experts * sizeof(uint32_t));
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(temp_l1_addr);

    // Calculate Mt per expert from the loaded array
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        uint32_t num_routed_value = num_routed_ptr[expert_idx];
        num_tiled[expert_idx] = (num_routed_value + 31) / 32;
        total_tiles += num_tiled[expert_idx];
    }

    uint32_t num_output_tiles_total = total_tiles * Nt;

    // Calculate work for this core dynamically
    constexpr uint32_t NUM_CORES = 64;  // 8x8 grid
    uint32_t core_id = core_y * 8 + core_x;
    uint32_t work_per_core = num_output_tiles_total / NUM_CORES;
    uint32_t remainder = num_output_tiles_total % NUM_CORES;
    
    // Calculate work_offset for this core
    uint32_t work_offset;
    if (core_id < remainder) {
        work_per_core += 1;
        work_offset = core_id * work_per_core;
    } else {
        work_offset = remainder * (work_per_core + 1) + (core_id - remainder) * work_per_core;
    }

    // Calculate the cumulative tile offset for each expert
    // Note: expert_tile_offsets[e] is the starting global tile ID for expert e
    // expert_tile_offsets[e+1] is the ending global tile ID (exclusive) for expert e
    uint32_t expert_tile_offsets[num_experts + 1];  // Need num_experts + 1 entries
    expert_tile_offsets[0] = 0;
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        expert_tile_offsets[expert_idx + 1] = expert_tile_offsets[expert_idx] + num_tiled[expert_idx] * Nt;
    }

    // Process work_per_core output tiles starting from work_offset
    for (uint32_t work_idx = 0; work_idx < work_per_core; work_idx++) {
        uint32_t global_output_tile_id = work_offset + work_idx;
        
        // Find which expert this tile belongs to
        // Handle empty experts (those with 0 tiles) correctly
        uint32_t expert_idx = num_experts - 1;  // Default to last expert
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
        
        // Calculate base tile index for this expert's output
        uint32_t output_expert_base = expert_idx * output_expert_stride;
        
        // Write output tile at (expert, mt, nt)
        uint32_t output_tile_idx = output_expert_base + mt * output_row_stride + nt;
        
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr_out = get_read_ptr(cb_out);
        noc_async_write_tile(output_tile_idx, output_accessor, l1_read_addr_out);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
