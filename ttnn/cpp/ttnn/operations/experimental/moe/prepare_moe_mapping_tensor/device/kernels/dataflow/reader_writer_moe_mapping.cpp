// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// This kernel prepares the MoE mapping tensor by scattering routing weights
// to their corresponding expert positions.
//
// Input:
// - selected_experts: ROW_MAJOR, uint32, shape (num_tokens, top_k)
//   Contains expert indices selected for each token (no duplicates per token)
// - routing_weights: ROW_MAJOR, bfloat16, shape (num_tokens, top_k)
//   Contains weights for each selected expert
//
// Output:
// - output: ROW_MAJOR, bfloat16, shape (num_tokens, padded_num_experts)
//   Sparse tensor with weights at positions corresponding to selected experts
//
// Note: Each token selects top_k unique experts (no duplicates)

void kernel_main() {
    // Get runtime arguments
    const uint32_t selected_experts_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tokens = get_arg_val<uint32_t>(3);
    const uint32_t top_k = get_arg_val<uint32_t>(4);
    const uint32_t num_experts = get_arg_val<uint32_t>(5);

    // Circular buffers
    constexpr uint32_t cb_id_row = tt::CBIndex::c_16;

    // For TILE layout, we need to pad to tile boundaries (32)
    const uint32_t padded_num_experts = (num_experts + 31) & ~31;

    // Create tensor accessors for proper DRAM access
    constexpr auto experts_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weights_accessor_args = TensorAccessorArgs<experts_accessor_args.next_compile_time_args_offset()>();
    constexpr auto output_accessor_args = TensorAccessorArgs<weights_accessor_args.next_compile_time_args_offset()>();

    const auto experts_accessor = TensorAccessor(experts_accessor_args, selected_experts_addr, top_k * sizeof(uint32_t));
    const auto weights_accessor = TensorAccessor(weights_accessor_args, routing_weights_addr, top_k * sizeof(uint16_t));
    const auto output_accessor = TensorAccessor(output_accessor_args, output_addr, padded_num_experts * sizeof(uint16_t));

    const uint32_t row_bytes = padded_num_experts * sizeof(uint16_t);

    // Reserve L1 buffer for row manipulation and intermediate buffers
    cb_reserve_back(cb_id_row, 1);
    uint32_t l1_row_addr = get_write_ptr(cb_id_row);

    //  Use L1 addresses at the end of the circular buffer for intermediate data
    // This avoids potential stack issues
    constexpr uint32_t cb_experts = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    cb_reserve_back(cb_experts, 1);
    cb_reserve_back(cb_weights, 1);
    uint32_t l1_experts_addr = get_write_ptr(cb_experts);
    uint32_t l1_weights_addr = get_write_ptr(cb_weights);

    // Process each token
    for (uint32_t token_idx = 0; token_idx < num_tokens; token_idx++) {
        // Initialize row with zeros in L1
        uint16_t* row_ptr = reinterpret_cast<uint16_t*>(l1_row_addr);
        for (uint32_t i = 0; i < padded_num_experts; i++) {
            row_ptr[i] = 0;
        }

        // Read selected experts for this token into L1
        uint64_t experts_noc_addr = get_noc_addr(token_idx, experts_accessor);
        noc_async_read(experts_noc_addr, l1_experts_addr, top_k * sizeof(uint32_t));
        noc_async_read_barrier();

        // Read routing weights for this token into L1
        uint64_t weights_noc_addr = get_noc_addr(token_idx, weights_accessor);
        noc_async_read(weights_noc_addr, l1_weights_addr, top_k * sizeof(uint16_t));
        noc_async_read_barrier();

        // Scatter weights into the row buffer in L1
        volatile uint32_t* expert_indices = reinterpret_cast<volatile uint32_t*>(l1_experts_addr);
        volatile uint16_t* weights = reinterpret_cast<volatile uint16_t*>(l1_weights_addr);

        for (uint32_t k = 0; k < top_k; k++) {
            uint32_t expert_idx = expert_indices[k];
            if (expert_idx < num_experts) {  // Bounds check
                uint16_t weight = weights[k];
                row_ptr[expert_idx] = weight;
            }
        }

        // Write the complete row to DRAM
        uint64_t output_row_noc_addr = get_noc_addr(token_idx, output_accessor);
        noc_async_write(l1_row_addr, output_row_noc_addr, row_bytes);
        noc_async_write_barrier();
    }

    // Release circular buffers
    cb_push_back(cb_weights, 1);
    cb_push_back(cb_experts, 1);
    cb_push_back(cb_id_row, 1);
}
