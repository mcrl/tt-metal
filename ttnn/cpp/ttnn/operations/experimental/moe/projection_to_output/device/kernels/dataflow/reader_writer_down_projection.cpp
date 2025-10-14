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

// MoE down projection kernel - performs batched matrix multiplication with routing weight application
// Processes each local expert, computing: (T_e × H') @ (H' × H) = T_e × H
// Then multiplies by routing weights and accumulates to final output
//
// NOTE: Routing tensors are DEVICE-LOCAL (indexed 0 to E/D-1) from prepare_moe_routing_tensors:
//   - num_routed_tokens: (E/D, 1) 2D tensor - token counts per local expert (read as 1D array in kernel)
//   - routed_tokens: (E/D, max_tokens) 2D tensor - token indices per local expert
//   - routed_token_weights: (E/D, max_tokens) 2D tensor - routing weights per local expert

void kernel_main() {
    // Get runtime arguments
    const uint32_t combined_activations_addr = get_arg_val<uint32_t>(0);
    const uint32_t routed_tokens_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(2);
    const uint32_t routed_token_weights_addr = get_arg_val<uint32_t>(3);
    const uint32_t down_proj_weights_addr = get_arg_val<uint32_t>(4);
    const uint32_t output_addr = get_arg_val<uint32_t>(5);
    const uint32_t num_tokens = get_arg_val<uint32_t>(6);
    const uint32_t hidden_dim = get_arg_val<uint32_t>(7);
    const uint32_t expert_dim = get_arg_val<uint32_t>(8);
    const uint32_t experts_per_device = get_arg_val<uint32_t>(9);
    const uint32_t max_tokens_per_expert = get_arg_val<uint32_t>(10);
    const uint32_t top_k = get_arg_val<uint32_t>(11);
    const uint32_t num_experts_padded = get_arg_val<uint32_t>(12);

    // Circular buffers
    constexpr uint32_t cb_combined_row = tt::CBIndex::c_0;
    constexpr uint32_t cb_routed_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_num_routed_row = tt::CBIndex::c_2;
    constexpr uint32_t cb_routing_weights_row = tt::CBIndex::c_3;
    constexpr uint32_t cb_weights_row = tt::CBIndex::c_4;
    constexpr uint32_t cb_output_row = tt::CBIndex::c_16;
    constexpr uint32_t cb_saved_output = tt::CBIndex::c_24;      // Scratch buffer for saved output
    constexpr uint32_t cb_matmul_result = tt::CBIndex::c_25;     // Scratch buffer for matmul result

    // Create tensor accessors for proper DRAM access
    constexpr auto combined_accessor_args = TensorAccessorArgs<0>();
    constexpr auto routed_accessor_args = TensorAccessorArgs<combined_accessor_args.next_compile_time_args_offset()>();
    constexpr auto num_routed_accessor_args = TensorAccessorArgs<routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto routing_weights_accessor_args = TensorAccessorArgs<num_routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto weights_accessor_args = TensorAccessorArgs<routing_weights_accessor_args.next_compile_time_args_offset()>();
    constexpr auto output_accessor_args = TensorAccessorArgs<weights_accessor_args.next_compile_time_args_offset()>();

    const auto combined_accessor = TensorAccessor(combined_accessor_args, combined_activations_addr, expert_dim * sizeof(uint16_t));
    const auto routed_accessor = TensorAccessor(routed_accessor_args, routed_tokens_addr, max_tokens_per_expert * sizeof(uint32_t));
    // num_routed_tokens tensor shape is (E/D, 1) with per-element pages - page_size is sizeof(uint32_t)
    const auto num_routed_accessor = TensorAccessor(num_routed_accessor_args, num_routed_tokens_addr, sizeof(uint32_t));
    const auto routing_weights_accessor = TensorAccessor(routing_weights_accessor_args, routed_token_weights_addr, max_tokens_per_expert * sizeof(uint16_t));
    const auto weights_accessor = TensorAccessor(weights_accessor_args, down_proj_weights_addr, hidden_dim * sizeof(uint16_t));
    const auto output_accessor = TensorAccessor(output_accessor_args, output_addr, hidden_dim * sizeof(uint16_t));

    // Reserve L1 buffers for intermediate data
    cb_reserve_back(cb_num_routed_row, 1);
    cb_reserve_back(cb_routed_row, 1);
    cb_reserve_back(cb_routing_weights_row, 1);
    cb_reserve_back(cb_combined_row, 1);
    cb_reserve_back(cb_weights_row, 1);
    cb_reserve_back(cb_output_row, 1);
    cb_reserve_back(cb_saved_output, 1);
    cb_reserve_back(cb_matmul_result, 1);

    uint32_t l1_num_routed_addr = get_write_ptr(cb_num_routed_row);
    uint32_t l1_routed_addr = get_write_ptr(cb_routed_row);
    uint32_t l1_routing_weights_addr = get_write_ptr(cb_routing_weights_row);
    uint32_t l1_combined_addr = get_write_ptr(cb_combined_row);
    uint32_t l1_weights_addr = get_write_ptr(cb_weights_row);
    uint32_t l1_output_addr = get_write_ptr(cb_output_row);
    uint32_t l1_saved_output_addr = get_write_ptr(cb_saved_output);
    uint32_t l1_matmul_result_addr = get_write_ptr(cb_matmul_result);
    volatile uint32_t* num_routed_local = reinterpret_cast<volatile uint32_t*>(l1_num_routed_addr);

    // Initialize output buffer to zeros for all tokens
    // This is necessary because we use read-modify-write pattern for accumulation
    volatile uint16_t* temp_output = reinterpret_cast<volatile uint16_t*>(l1_output_addr);
    for (uint32_t tok = 0; tok < num_tokens; tok++) {
        // Zero out the buffer
        for (uint32_t h = 0; h < hidden_dim; h++) {
            temp_output[h] = 0;
        }
        // Write zeros to output
        uint64_t output_noc_addr = get_noc_addr(tok, output_accessor);
        noc_async_write(l1_output_addr, output_noc_addr, hidden_dim * sizeof(uint16_t));
    }
    noc_async_write_barrier();

    // Track read position in combined_activations
    uint32_t read_pos = 0;

    // Process each local expert
    for (uint32_t local_expert_idx = 0; local_expert_idx < experts_per_device; local_expert_idx++) {
        // Get token count for this expert (Per-element page access)
        uint64_t num_routed_noc_addr = get_noc_addr(local_expert_idx, num_routed_accessor);
        noc_async_read(num_routed_noc_addr, l1_num_routed_addr, sizeof(uint32_t));
        noc_async_read_barrier();

        uint32_t token_count = num_routed_local[0];
        if (token_count == 0) continue;

        // Read routed tokens and routing weights for this expert (using LOCAL index)
        uint64_t routed_noc_addr = get_noc_addr(local_expert_idx, routed_accessor);
        noc_async_read(routed_noc_addr, l1_routed_addr, max_tokens_per_expert * sizeof(uint32_t));

        uint64_t routing_weights_noc_addr = get_noc_addr(local_expert_idx, routing_weights_accessor);
        noc_async_read(routing_weights_noc_addr, l1_routing_weights_addr, max_tokens_per_expert * sizeof(uint16_t));
        noc_async_read_barrier();

        volatile uint32_t* token_indices = reinterpret_cast<volatile uint32_t*>(l1_routed_addr);
        volatile uint16_t* routing_weights = reinterpret_cast<volatile uint16_t*>(l1_routing_weights_addr);

        // Pre-allocate space for output accumulation (one row per token for this expert)
        // We'll accumulate locally and write once per token
        volatile uint16_t* output_buffer = reinterpret_cast<volatile uint16_t*>(l1_output_addr);

        // Process each token for this expert
        for (uint32_t token_idx = 0; token_idx < token_count; token_idx++) {
            // Read combined activation row sequentially
            uint64_t combined_noc_addr = get_noc_addr(read_pos + token_idx, combined_accessor);
            noc_async_read(combined_noc_addr, l1_combined_addr, expert_dim * sizeof(uint16_t));
            noc_async_read_barrier();

            volatile uint16_t* activation = reinterpret_cast<volatile uint16_t*>(l1_combined_addr);

            // Get token index for output position
            uint32_t output_token_idx = token_indices[token_idx];

            // Get routing weight for this token-expert pair
            float routing_weight = bfloat16_to_float(routing_weights[token_idx]);

            // Read current output value for accumulation (read-modify-write pattern)
            uint64_t output_noc_addr = get_noc_addr(output_token_idx, output_accessor);
            noc_async_read(output_noc_addr, l1_output_addr, hidden_dim * sizeof(uint16_t));
            noc_async_read_barrier();

            // Use L1 buffers instead of stack arrays to avoid stack overflow with large hidden_dim
            volatile float* saved_output = reinterpret_cast<volatile float*>(l1_saved_output_addr);
            volatile float* matmul_result = reinterpret_cast<volatile float*>(l1_matmul_result_addr);

            // Save the original output values (from previous experts)
            // We need to:  output = output_old + routing_weight * (activation @ weights)
            for (uint32_t h = 0; h < hidden_dim; h++) {
                saved_output[h] = bfloat16_to_float(output_buffer[h]);
            }

            // Initialize matmul_result to zero
            for (uint32_t h = 0; h < hidden_dim; h++) {
                matmul_result[h] = 0.0f;
            }

            // Optimized: iterate over weight rows (expert_dim), accumulate to all output positions
            for (uint32_t h_prime = 0; h_prime < expert_dim; h_prime++) {
                // Read one weight row: weights[h_prime, :] (size: hidden_dim)
                uint32_t weight_row_idx = local_expert_idx * expert_dim + h_prime;
                uint64_t weight_noc_addr = get_noc_addr(weight_row_idx, weights_accessor);
                noc_async_read(weight_noc_addr, l1_weights_addr, hidden_dim * sizeof(uint16_t));
                noc_async_read_barrier();

                volatile uint16_t* weight_row = reinterpret_cast<volatile uint16_t*>(l1_weights_addr);
                float a = bfloat16_to_float(activation[h_prime]);

                // Accumulate contribution of this weight row to matmul result
                for (uint32_t h = 0; h < hidden_dim; h++) {
                    float w = bfloat16_to_float(weight_row[h]);
                    matmul_result[h] += a * w;
                }
            }

            // Apply routing weight and add to saved output
            for (uint32_t h = 0; h < hidden_dim; h++) {
                output_buffer[h] = float_to_bfloat16(saved_output[h] + matmul_result[h] * routing_weight);
            }

            // Write accumulated result back to output
            noc_async_write(l1_output_addr, output_noc_addr, hidden_dim * sizeof(uint16_t));
            noc_async_write_barrier();
        }

        // Update read position for next expert
        read_pos += token_count;
    }

    // Release circular buffers
    cb_push_back(cb_num_routed_row, 1);
    cb_push_back(cb_routed_row, 1);
    cb_push_back(cb_routing_weights_row, 1);
    cb_push_back(cb_combined_row, 1);
    cb_push_back(cb_weights_row, 1);
    cb_push_back(cb_output_row, 1);
    cb_push_back(cb_saved_output, 1);
    cb_push_back(cb_matmul_result, 1);
}