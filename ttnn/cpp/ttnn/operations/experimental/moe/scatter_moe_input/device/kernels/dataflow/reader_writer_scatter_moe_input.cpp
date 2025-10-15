// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * Scatter MoE Input Kernel (Single-Core)
 *
 * This kernel rearranges input tokens based on expert assignments.
 * For each local expert, it:
 * 1. Reads the number of assigned tokens
 * 2. Gathers those tokens from input_hidden_state
 * 3. Writes them to output sequentially
 * 4. Zero-pads remaining positions
 *
 * Compile-time args:
 * - cb_id_input: Circular buffer ID for input rows
 * - cb_id_output: Circular buffer ID for output (zero-padding)
 * - input_is_dram: Whether input buffer is in DRAM
 * - num_routed_is_dram: Whether num_routed_tokens is in DRAM
 * - routed_tokens_is_dram: Whether routed_tokens is in DRAM
 * - output_is_dram: Whether output buffer is in DRAM
 * - hidden_dim: H - hidden dimension
 * - num_tokens: T - total number of tokens
 * - num_local_experts: E/D - number of experts on this device
 * - row_size_bytes: Byte size of one row (H * element_size)
 *
 * Runtime args:
 * - input_buffer_addr: Address of input_hidden_state (T, H)
 * - num_routed_tokens_addr: Address of num_routed_tokens (E/D, 1)
 * - routed_tokens_addr: Address of routed_tokens (E/D, T)
 * - output_buffer_addr: Address of output (E/D, T, H)
 */

void kernel_main() {
    // Runtime arguments
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(1);
    uint32_t routed_tokens_addr = get_arg_val<uint32_t>(2);
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(3);

    // Compile-time arguments
    constexpr uint32_t cb_id_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_output = get_compile_time_arg_val(1);
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool num_routed_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool routed_tokens_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(6);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(7);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(8);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(9);

    // Create address generators
    const InterleavedAddrGen<input_is_dram> input_addrgen = {
        .bank_base_address = input_buffer_addr,
        .page_size = row_size_bytes
    };

    const InterleavedAddrGen<num_routed_is_dram> num_routed_addrgen = {
        .bank_base_address = num_routed_tokens_addr,
        .page_size = sizeof(uint32_t)
    };

    const InterleavedAddrGen<routed_tokens_is_dram> routed_tokens_addrgen = {
        .bank_base_address = routed_tokens_addr,
        .page_size = num_tokens * sizeof(uint32_t)
    };

    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };

    // Process each expert
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        // Step 1: Read num_routed_tokens[expert_idx, 0]
        // Allocate L1 space for reading single uint32
        cb_reserve_back(cb_id_output, 1);
        uint32_t temp_addr = get_write_ptr(cb_id_output);

        uint64_t num_routed_noc_addr = get_noc_addr(expert_idx, num_routed_addrgen);
        noc_async_read(num_routed_noc_addr, temp_addr, sizeof(uint32_t));
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* temp_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(temp_addr);
        uint32_t t_e = temp_ptr[0];  // Number of tokens for this expert

        cb_pop_front(cb_id_output, 1);

        // Step 2: Read routed_tokens row for this expert (E/D rows, each of size T * sizeof(uint32_t))
        cb_reserve_back(cb_id_output, 1);
        uint32_t routed_tokens_l1_addr = get_write_ptr(cb_id_output);

        uint64_t routed_tokens_row_noc_addr = get_noc_addr(expert_idx, routed_tokens_addrgen);
        noc_async_read(routed_tokens_row_noc_addr, routed_tokens_l1_addr, num_tokens * sizeof(uint32_t));
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* routed_tokens_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routed_tokens_l1_addr);

        // Step 3: Gather assigned tokens and write to output
        constexpr uint32_t batch_size = 8;  // Batch writes for efficiency

        for (uint32_t i = 0; i < t_e; i++) {
            // Get token index
            uint32_t token_idx = routed_tokens_ptr[i];

            // Read input row: input_hidden_state[token_idx, :]
            cb_reserve_back(cb_id_input, 1);
            uint32_t input_l1_addr = get_write_ptr(cb_id_input);

            uint64_t input_row_noc_addr = get_noc_addr(token_idx, input_addrgen);
            noc_async_read(input_row_noc_addr, input_l1_addr, row_size_bytes);
            noc_async_read_barrier();

            // Write to output: output[expert_idx, i, :]
            uint32_t output_page_idx = expert_idx * num_tokens + i;
            uint64_t output_row_noc_addr = get_noc_addr(output_page_idx, output_addrgen);

            noc_async_write(input_l1_addr, output_row_noc_addr, row_size_bytes);

            cb_pop_front(cb_id_input, 1);

            // Batch barrier every batch_size writes
            if (((i + 1) % batch_size == 0) || (i == t_e - 1)) {
                noc_async_write_barrier();
            }
        }

        // Ensure all writes complete before zero-padding
        noc_async_write_barrier();

        // Step 4: Zero-pad remaining positions [t_e, num_tokens)
        if (t_e < num_tokens) {
            cb_reserve_back(cb_id_output, 1);
            uint32_t zero_l1_addr = get_write_ptr(cb_id_output);

            // Initialize L1 buffer with zeros
            volatile tt_l1_ptr uint16_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(zero_l1_addr);
            for (uint32_t j = 0; j < hidden_dim; j++) {
                zero_ptr[j] = 0;  // bfloat16 zero
            }

            // Write zeros to padding region
            for (uint32_t i = t_e; i < num_tokens; i++) {
                uint32_t output_page_idx = expert_idx * num_tokens + i;
                uint64_t output_row_noc_addr = get_noc_addr(output_page_idx, output_addrgen);

                noc_async_write(zero_l1_addr, output_row_noc_addr, row_size_bytes);

                // Batch barrier every batch_size writes
                if (((i - t_e + 1) % batch_size == 0) || (i == num_tokens - 1)) {
                    noc_async_write_barrier();
                }
            }

            cb_pop_front(cb_id_output, 1);
        }

        // Free routed_tokens buffer
        cb_pop_front(cb_id_output, 1);

        // Ensure all writes for this expert complete before moving to next
        noc_async_write_barrier();
    }
}
