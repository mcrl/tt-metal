// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// This kernel prepares MoE routing tensors for efficient expert-parallel computation
//
// Input:
// - selected_experts: ROW_MAJOR, uint32, shape (num_tokens, top_k)
//   Contains expert indices selected for each token (no duplicates per token)
// - routing_weights: ROW_MAJOR, bfloat16, shape (num_tokens, top_k)
//   Contains weights for each selected expert
//
// Output:
// - num_routed_tokens: ROW_MAJOR, uint32, shape (num_experts)
//   Count of tokens routed to each expert
// - routed_tokens: ROW_MAJOR, uint32, shape (num_experts, max_tokens_per_expert)
//   Token indices for each expert (padded with invalid values)
// - routed_token_weights: ROW_MAJOR, bfloat16, shape (num_experts, max_tokens_per_expert)
//   Routing weights for each expert (padded with zeros)
//
// Algorithm:
// 1. First pass: Count tokens routed to each expert
// 2. Second pass: Populate routed_tokens and routed_token_weights

void kernel_main() {
    // Get runtime arguments
    const uint32_t selected_experts_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(2);
    const uint32_t routed_tokens_addr = get_arg_val<uint32_t>(3);
    const uint32_t routed_token_weights_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_tokens = get_arg_val<uint32_t>(5);
    const uint32_t top_k = get_arg_val<uint32_t>(6);
    const uint32_t num_experts = get_arg_val<uint32_t>(7);
    const uint32_t max_tokens_per_expert = get_arg_val<uint32_t>(8);

    // Circular buffer indices
    constexpr uint32_t cb_experts = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_num_routed = tt::CBIndex::c_16;
    constexpr uint32_t cb_routed_tokens = tt::CBIndex::c_17;
    constexpr uint32_t cb_routed_weights = tt::CBIndex::c_18;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;

    // Pad num_experts to 32 for alignment
    const uint32_t padded_num_experts = (num_experts + 31) & ~31;

    // Create tensor accessors for proper DRAM access
    constexpr auto experts_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weights_accessor_args = TensorAccessorArgs<experts_accessor_args.next_compile_time_args_offset()>();
    constexpr auto num_routed_accessor_args = TensorAccessorArgs<weights_accessor_args.next_compile_time_args_offset()>();
    constexpr auto routed_tokens_accessor_args = TensorAccessorArgs<num_routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto routed_weights_accessor_args = TensorAccessorArgs<routed_tokens_accessor_args.next_compile_time_args_offset()>();

    const auto experts_accessor = TensorAccessor(experts_accessor_args, selected_experts_addr, top_k * sizeof(uint32_t));
    const auto weights_accessor = TensorAccessor(weights_accessor_args, routing_weights_addr, top_k * sizeof(uint16_t));
    const auto num_routed_accessor = TensorAccessor(num_routed_accessor_args, num_routed_tokens_addr, padded_num_experts * sizeof(uint32_t));
    const auto routed_tokens_accessor = TensorAccessor(routed_tokens_accessor_args, routed_tokens_addr, max_tokens_per_expert * sizeof(uint32_t));
    const auto routed_weights_accessor = TensorAccessor(routed_weights_accessor_args, routed_token_weights_addr, max_tokens_per_expert * sizeof(uint16_t));

    // Reserve L1 buffers
    cb_reserve_back(cb_experts, 1);
    cb_reserve_back(cb_weights, 1);
    cb_reserve_back(cb_num_routed, 1);
    cb_reserve_back(cb_scratch, 1);

    uint32_t l1_experts_addr = get_write_ptr(cb_experts);
    uint32_t l1_weights_addr = get_write_ptr(cb_weights);
    uint32_t l1_num_routed_addr = get_write_ptr(cb_num_routed);
    uint32_t l1_scratch_addr = get_write_ptr(cb_scratch);

    // Initialize num_routed_tokens to 0
    uint32_t* num_routed_ptr = reinterpret_cast<uint32_t*>(l1_num_routed_addr);
    for (uint32_t e = 0; e < padded_num_experts; e++) {
        num_routed_ptr[e] = 0;
    }

    // Use scratch buffer to store intermediate routing data
    // Layout: [expert_id][token_list][weight_list]
    // Each expert gets max_tokens_per_expert slots for tokens and weights
    uint32_t* scratch_tokens = reinterpret_cast<uint32_t*>(l1_scratch_addr);
    uint16_t* scratch_weights = reinterpret_cast<uint16_t*>(
        l1_scratch_addr + padded_num_experts * max_tokens_per_expert * sizeof(uint32_t));

    // PASS 1: Count tokens per expert and collect routing information
    for (uint32_t token_idx = 0; token_idx < num_tokens; token_idx++) {
        // Read selected experts for this token
        uint64_t experts_noc_addr = get_noc_addr(token_idx, experts_accessor);
        noc_async_read(experts_noc_addr, l1_experts_addr, top_k * sizeof(uint32_t));
        noc_async_read_barrier();

        // Read routing weights for this token
        uint64_t weights_noc_addr = get_noc_addr(token_idx, weights_accessor);
        noc_async_read(weights_noc_addr, l1_weights_addr, top_k * sizeof(uint16_t));
        noc_async_read_barrier();

        volatile uint32_t* expert_indices = reinterpret_cast<volatile uint32_t*>(l1_experts_addr);
        volatile uint16_t* weights = reinterpret_cast<volatile uint16_t*>(l1_weights_addr);

        // Process each expert selection
        for (uint32_t k = 0; k < top_k; k++) {
            uint32_t expert_idx = expert_indices[k];
            if (expert_idx < num_experts) {
                // Get current count for this expert
                uint32_t count = num_routed_ptr[expert_idx];

                // Store token index and weight in scratch buffer
                if (count < max_tokens_per_expert) {
                    uint32_t offset = expert_idx * max_tokens_per_expert + count;
                    scratch_tokens[offset] = token_idx;
                    scratch_weights[offset] = weights[k];

                    // Increment count
                    num_routed_ptr[expert_idx] = count + 1;
                }
            }
        }
    }

    // Write num_routed_tokens to DRAM
    uint64_t num_routed_noc_addr = get_noc_addr(0, num_routed_accessor);
    noc_async_write(l1_num_routed_addr, num_routed_noc_addr, padded_num_experts * sizeof(uint32_t));
    noc_async_write_barrier();

    // PASS 2: Write routed_tokens and routed_token_weights
    cb_reserve_back(cb_routed_tokens, 1);
    cb_reserve_back(cb_routed_weights, 1);
    uint32_t l1_routed_tokens_addr = get_write_ptr(cb_routed_tokens);
    uint32_t l1_routed_weights_addr = get_write_ptr(cb_routed_weights);

    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        uint32_t count = num_routed_ptr[expert_idx];

        // Prepare routed_tokens row
        uint32_t* routed_tokens_row = reinterpret_cast<uint32_t*>(l1_routed_tokens_addr);
        uint16_t* routed_weights_row = reinterpret_cast<uint16_t*>(l1_routed_weights_addr);

        // Copy from scratch buffer and pad with invalid values
        uint32_t base_offset = expert_idx * max_tokens_per_expert;
        for (uint32_t i = 0; i < max_tokens_per_expert; i++) {
            if (i < count) {
                routed_tokens_row[i] = scratch_tokens[base_offset + i];
                routed_weights_row[i] = scratch_weights[base_offset + i];
            } else {
                routed_tokens_row[i] = 0xFFFFFFFF;  // Invalid token index
                routed_weights_row[i] = 0;          // Zero weight for padding
            }
        }

        // Write routed_tokens row to DRAM
        uint64_t routed_tokens_noc_addr = get_noc_addr(expert_idx, routed_tokens_accessor);
        noc_async_write(l1_routed_tokens_addr, routed_tokens_noc_addr, max_tokens_per_expert * sizeof(uint32_t));

        // Write routed_token_weights row to DRAM
        uint64_t routed_weights_noc_addr = get_noc_addr(expert_idx, routed_weights_accessor);
        noc_async_write(l1_routed_weights_addr, routed_weights_noc_addr, max_tokens_per_expert * sizeof(uint16_t));

        noc_async_write_barrier();
    }

    // Pad remaining expert rows with empty data
    for (uint32_t expert_idx = num_experts; expert_idx < padded_num_experts; expert_idx++) {
        // Initialize with invalid/zero values
        uint32_t* routed_tokens_row = reinterpret_cast<uint32_t*>(l1_routed_tokens_addr);
        uint16_t* routed_weights_row = reinterpret_cast<uint16_t*>(l1_routed_weights_addr);

        for (uint32_t i = 0; i < max_tokens_per_expert; i++) {
            routed_tokens_row[i] = 0xFFFFFFFF;  // Invalid token index
            routed_weights_row[i] = 0;          // Zero weight
        }

        // Write padding rows
        uint64_t routed_tokens_noc_addr = get_noc_addr(expert_idx, routed_tokens_accessor);
        noc_async_write(l1_routed_tokens_addr, routed_tokens_noc_addr, max_tokens_per_expert * sizeof(uint32_t));

        uint64_t routed_weights_noc_addr = get_noc_addr(expert_idx, routed_weights_accessor);
        noc_async_write(l1_routed_weights_addr, routed_weights_noc_addr, max_tokens_per_expert * sizeof(uint16_t));

        noc_async_write_barrier();
    }

    // Release circular buffers
    cb_push_back(cb_experts, 1);
    cb_push_back(cb_weights, 1);
    cb_push_back(cb_num_routed, 1);
    cb_push_back(cb_routed_tokens, 1);
    cb_push_back(cb_routed_weights, 1);
    cb_push_back(cb_scratch, 1);
}