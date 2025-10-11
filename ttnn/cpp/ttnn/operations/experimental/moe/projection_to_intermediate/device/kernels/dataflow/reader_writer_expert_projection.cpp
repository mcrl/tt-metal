// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"

// Bfloat16 conversion functions
static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

static inline uint16_t float_to_bfloat16(float val) {
    uint32_t u;
    std::memcpy(&u, &val, sizeof(u));
    return uint16_t(u >> 16);
}

// MoE expert projection kernel - performs batched matrix multiplication for expert processing
// Processes each local expert, gathering tokens and computing: (T_e × H) @ (H × H') = T_e × H'

void kernel_main() {
    // Get runtime arguments
    const uint32_t hidden_states_addr = get_arg_val<uint32_t>(0);
    const uint32_t routed_tokens_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(2);
    const uint32_t expert_weights_addr = get_arg_val<uint32_t>(3);
    const uint32_t device_expert_mapping_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_addr = get_arg_val<uint32_t>(5);
    const uint32_t num_tokens = get_arg_val<uint32_t>(6);
    const uint32_t hidden_dim = get_arg_val<uint32_t>(7);
    const uint32_t expert_dim = get_arg_val<uint32_t>(8);
    const uint32_t experts_per_device = get_arg_val<uint32_t>(9);
    const uint32_t max_tokens_per_expert = get_arg_val<uint32_t>(10);
    const uint32_t output_size = get_arg_val<uint32_t>(11);
    const uint32_t num_experts_padded = get_arg_val<uint32_t>(12);

    // Circular buffers
    constexpr uint32_t cb_hidden_row = tt::CBIndex::c_0;
    constexpr uint32_t cb_routed_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_num_routed_row = tt::CBIndex::c_2;
    constexpr uint32_t cb_mapping = tt::CBIndex::c_3;
    constexpr uint32_t cb_expert_weights = tt::CBIndex::c_4;
    constexpr uint32_t cb_output_row = tt::CBIndex::c_16;

    // Create tensor accessors for proper DRAM access
    constexpr auto hidden_accessor_args = TensorAccessorArgs<0>();
    constexpr auto routed_accessor_args = TensorAccessorArgs<hidden_accessor_args.next_compile_time_args_offset()>();
    constexpr auto num_routed_accessor_args = TensorAccessorArgs<routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto weights_accessor_args = TensorAccessorArgs<num_routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto mapping_accessor_args = TensorAccessorArgs<weights_accessor_args.next_compile_time_args_offset()>();
    constexpr auto output_accessor_args = TensorAccessorArgs<mapping_accessor_args.next_compile_time_args_offset()>();

    const auto hidden_accessor = TensorAccessor(hidden_accessor_args, hidden_states_addr, hidden_dim * sizeof(uint16_t));
    const auto routed_accessor = TensorAccessor(routed_accessor_args, routed_tokens_addr, max_tokens_per_expert * sizeof(uint32_t));
    // num_routed_tokens tensor has shape (1, E) - row size is E elements
    const auto num_routed_accessor = TensorAccessor(num_routed_accessor_args, num_routed_tokens_addr, num_experts_padded * sizeof(uint32_t));
    const auto weights_accessor = TensorAccessor(weights_accessor_args, expert_weights_addr, expert_dim * sizeof(uint16_t));
    const auto mapping_accessor = TensorAccessor(mapping_accessor_args, device_expert_mapping_addr, experts_per_device * sizeof(int32_t));
    const auto output_accessor = TensorAccessor(output_accessor_args, output_addr, expert_dim * sizeof(uint16_t));

    // Reserve L1 buffers for intermediate data
    cb_reserve_back(cb_mapping, 1);
    cb_reserve_back(cb_num_routed_row, 1);
    cb_reserve_back(cb_routed_row, 1);
    cb_reserve_back(cb_hidden_row, 1);
    cb_reserve_back(cb_expert_weights, 1);
    cb_reserve_back(cb_output_row, 1);

    uint32_t l1_mapping_addr = get_write_ptr(cb_mapping);
    uint32_t l1_num_routed_addr = get_write_ptr(cb_num_routed_row);
    uint32_t l1_routed_addr = get_write_ptr(cb_routed_row);
    uint32_t l1_hidden_addr = get_write_ptr(cb_hidden_row);
    uint32_t l1_weights_addr = get_write_ptr(cb_expert_weights);
    uint32_t l1_output_addr = get_write_ptr(cb_output_row);

    // Read device expert mapping and num_routed_tokens
    uint64_t mapping_noc_addr = get_noc_addr(0, mapping_accessor);
    noc_async_read(mapping_noc_addr, l1_mapping_addr, experts_per_device * sizeof(int32_t));
    noc_async_read_barrier();

    // Read the entire num_routed_tokens array (shape: 1 x E)
    uint64_t num_routed_noc_addr = get_noc_addr(0, num_routed_accessor);
    noc_async_read(num_routed_noc_addr, l1_num_routed_addr, num_experts_padded * sizeof(uint32_t));
    noc_async_read_barrier();

    volatile int32_t* expert_mapping = reinterpret_cast<volatile int32_t*>(l1_mapping_addr);
    volatile uint32_t* num_routed_all = reinterpret_cast<volatile uint32_t*>(l1_num_routed_addr);

    // Track write position in output
    uint32_t write_pos = 0;

    // Process each local expert
    for (uint32_t local_expert_idx = 0; local_expert_idx < experts_per_device; local_expert_idx++) {
        // Get global expert index
        int32_t global_expert_idx = expert_mapping[local_expert_idx];

        // Get token count for this expert
        uint32_t token_count = num_routed_all[global_expert_idx];

        if (token_count == 0) {
            continue;
        }

        // Debug: Print what we're processing (will appear in device logs)
        DPRINT << "MOE: local_expert=" << local_expert_idx
               << " global_expert=" << global_expert_idx
               << " token_count=" << token_count << ENDL();

        // Read routed tokens for this expert
        uint64_t routed_noc_addr = get_noc_addr(global_expert_idx, routed_accessor);
        noc_async_read(routed_noc_addr, l1_routed_addr, max_tokens_per_expert * sizeof(uint32_t));
        noc_async_read_barrier();

        volatile uint32_t* token_indices = reinterpret_cast<volatile uint32_t*>(l1_routed_addr);
        volatile uint16_t* hidden_row = reinterpret_cast<volatile uint16_t*>(l1_hidden_addr);
        volatile uint16_t* expert_weights_row = reinterpret_cast<volatile uint16_t*>(l1_weights_addr);
        volatile uint16_t* output_row = reinterpret_cast<volatile uint16_t*>(l1_output_addr);

        // Process each token assigned to this expert
        for (uint32_t t = 0; t < token_count; t++) {
            uint32_t token_idx = token_indices[t];

            // Debug: verify token index is valid
            if (token_idx >= num_tokens) {
                DPRINT << "ERROR: Invalid token_idx=" << token_idx << " >= num_tokens=" << num_tokens << ENDL();
            }
            DPRINT << "  Processing token " << token_idx << ENDL();

            // Read token's hidden state
            uint64_t hidden_noc_addr = get_noc_addr(token_idx, hidden_accessor);
            noc_async_read(hidden_noc_addr, l1_hidden_addr, hidden_dim * sizeof(uint16_t));
            noc_async_read_barrier();

            // Use L1 memory for FP32 accumulators (reuse buffer space after weights)
            float* accumulator = reinterpret_cast<float*>(l1_weights_addr + expert_dim * sizeof(uint16_t));

            // Initialize accumulators
            for (uint32_t j = 0; j < expert_dim; j++) {
                accumulator[j] = 0.0f;
            }

            // Compute matmul: output[j] = sum_k(hidden[k] * weight[k][j])
            for (uint32_t k = 0; k < hidden_dim; k++) {
                // Read weight row k for this expert
                uint32_t weight_row_idx = local_expert_idx * hidden_dim + k;
                uint64_t weights_noc_addr = get_noc_addr(weight_row_idx, weights_accessor);
                noc_async_read(weights_noc_addr, l1_weights_addr, expert_dim * sizeof(uint16_t));
                noc_async_read_barrier();

                float a_float = bfloat16_to_float(hidden_row[k]);

                // Accumulate in FP32
                for (uint32_t j = 0; j < expert_dim; j++) {
                    float b_float = bfloat16_to_float(expert_weights_row[j]);
                    accumulator[j] += a_float * b_float;
                }
            }

            // Convert FP32 to bfloat16
            for (uint32_t j = 0; j < expert_dim; j++) {
                output_row[j] = float_to_bfloat16(accumulator[j]);
            }

            // Write output row
            uint64_t output_noc_addr = get_noc_addr(write_pos, output_accessor);
            noc_async_write(l1_output_addr, output_noc_addr, expert_dim * sizeof(uint16_t));
            noc_async_write_barrier();
            write_pos++;
        }
    }

    // Release circular buffers
    cb_push_back(cb_output_row, 1);
    cb_push_back(cb_expert_weights, 1);
    cb_push_back(cb_hidden_row, 1);
    cb_push_back(cb_routed_row, 1);
    cb_push_back(cb_num_routed_row, 1);
    cb_push_back(cb_mapping, 1);
}
