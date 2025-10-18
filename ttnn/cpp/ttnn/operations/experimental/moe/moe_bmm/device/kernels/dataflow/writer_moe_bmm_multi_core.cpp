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
    uint32_t work_offset = get_arg_val<uint32_t>(5);  // Starting output tile for this core
    uint32_t work_per_core = get_arg_val<uint32_t>(6);  // Number of output tiles to process

    // Circular buffer index for output
    constexpr uint32_t cb_out = 16;

    // Create tensor accessors for DRAM buffers
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_accessor = TensorAccessor(output_args, output_addr, get_tile_size(cb_out));
    
    constexpr auto num_routed_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto num_routed_accessor = TensorAccessor(num_routed_args, num_routed_addr, sizeof(uint32_t));

    // Output: (E/D, T, H_out) in tiles -> (num_experts, Mt_max, Nt)
    uint32_t output_expert_stride = Mt_max * Nt; // Tiles per expert output
    uint32_t output_row_stride = Nt;             // Tiles per output row

    // Cache for num_routed_tokens per expert
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t num_routed_cache[64];  // Assuming max 64 experts per device
    uint32_t Mt_cache[64];  // Cached Mt values per expert
    
    // Pre-fetch all num_routed_tokens values directly to L1
    // Use a different approach: read them one at a time as needed
    // to avoid potential conflicts with reader kernel
    
    // Actually, we need to read all at once to calculate offsets
    // But let's add a small delay/barrier to avoid simultaneous access with reader
    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        uint64_t temp_noc_addr = num_routed_accessor.get_noc_addr(expert_idx);
        uint64_t temp_l1_addr = get_write_ptr(cb_out);  // Reuse cb_out space temporarily
        
        noc_async_read(temp_noc_addr, temp_l1_addr, sizeof(uint32_t));
        noc_async_read_barrier();
        
        volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(temp_l1_addr);
        num_routed_cache[expert_idx] = num_routed_ptr[0];
        Mt_cache[expert_idx] = (num_routed_cache[expert_idx] + TILE_HEIGHT - 1) / TILE_HEIGHT;
    }

    // Calculate the cumulative tile offset for each expert
    uint32_t expert_tile_offsets[64];
    expert_tile_offsets[0] = 0;
    for (uint32_t expert_idx = 1; expert_idx < num_experts; expert_idx++) {
        expert_tile_offsets[expert_idx] = expert_tile_offsets[expert_idx - 1] + Mt_cache[expert_idx - 1] * Nt;
    }

    // Process work_per_core output tiles starting from work_offset
    for (uint32_t work_idx = 0; work_idx < work_per_core; work_idx++) {
        uint32_t global_output_tile_id = work_offset + work_idx;
        
        // Find which expert this tile belongs to using binary-like search
        uint32_t expert_idx = 0;
        for (uint32_t e = 1; e < num_experts; e++) {
            if (global_output_tile_id < expert_tile_offsets[e]) {
                break;
            }
            expert_idx = e;
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
