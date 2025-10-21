// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * Reader Kernel for Local Reduce MoE Output (Multi-Core, Token-Parallel)
 *
 * This kernel reads routing information and hidden states from DRAM and
 * pushes them to separate circular buffers for the compute kernel.
 *
 * For each token assigned to this core:
 * 1. Read routing metadata (token_idx_map, weights, num_routed_tokens) for all experts into separate CBs
 * 2. For each expert, if token matches, read corresponding hidden state row
 * 3. Push data to circular buffers for compute kernel
 *
 * Compile-time args:
 * - input_is_dram: Whether input buffer is in DRAM
 * - token_idx_is_dram: Whether token_idx_map is in DRAM
 * - weights_is_dram: Whether routed_token_weights is in DRAM
 * - num_routed_is_dram: Whether num_routed_tokens is in DRAM
 * - hidden_dim: H - hidden dimension
 * - num_local_experts: E/D - number of experts on this device
 * - max_tokens: T - maximum tokens dimension in input tensor
 * - row_size_bytes: Byte size of one hidden state row (H * element_size)
 *
 * Runtime args:
 * - input_buffer_addr: Address of input_hidden_state (E/D, T, H)
 * - token_idx_map_addr: Address of token_idx_map (E/D, T)
 * - weights_addr: Address of routed_token_weights (E/D, T)
 * - num_routed_tokens_addr: Address of num_routed_tokens (E/D, 1)
 * - start_token_idx: First token index for this core
 * - end_token_idx: Last token index (exclusive) for this core
 */

#define MAX_EXPERTS_PER_DEVICE 16

void kernel_main() {
    // Runtime arguments
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t token_idx_map_addr = get_arg_val<uint32_t>(1);
    uint32_t weights_addr = get_arg_val<uint32_t>(2);
    uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(3);
    uint32_t start_token_idx = get_arg_val<uint32_t>(4);
    uint32_t end_token_idx = get_arg_val<uint32_t>(5);

    // Circular buffer IDs (static)
    constexpr tt::CBIndex cb_id_input = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_id_token_idx = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_id_weights = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_id_num_routed = tt::CBIndex::c_3;
    constexpr tt::CBIndex cb_id_weight_scalar = tt::CBIndex::c_4;
    constexpr tt::CBIndex cb_id_output = tt::CBIndex::c_16;

    // Compile-time arguments
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr bool token_idx_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr bool weights_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool num_routed_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(4);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(5);
    constexpr uint32_t max_tokens = get_compile_time_arg_val(6);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(7);

    uint32_t hidden_dim_tiles = hidden_dim / 1024;
    constexpr uint32_t tile_size_bytes = 32 * 32 * sizeof(uint16_t);

    // Create address generators
    const InterleavedAddrGen<input_is_dram> input_addrgen = {
        .bank_base_address = input_buffer_addr,
        .page_size = row_size_bytes
    };

    const InterleavedAddrGen<token_idx_is_dram> token_idx_addrgen = {
        .bank_base_address = token_idx_map_addr,
        .page_size = max_tokens * sizeof(uint32_t)
    };

    const InterleavedAddrGen<weights_is_dram> weights_addrgen = {
        .bank_base_address = weights_addr,
        .page_size = max_tokens * sizeof(uint16_t)  // bfloat16
    };

    const InterleavedAddrGen<num_routed_is_dram> num_routed_addrgen = {
        .bank_base_address = num_routed_tokens_addr,
        .page_size = sizeof(uint32_t)
    };
    // Helper function to convert bfloat16 to float
    auto bf16_to_float = [](uint16_t bf16) -> float {
        union {
            uint32_t u;
            float f;
        } converter;
        converter.u = ((uint32_t)bf16) << 16;
        return converter.f;
    };

    // Reading num_routed tokens for all experts
    // Workaround for handling (E/D, 1) multi-page tensor in the reader kernel
    uint32_t num_routed_local[MAX_EXPERTS_PER_DEVICE];
    cb_reserve_back(cb_id_num_routed, 1);
    volatile uint32_t* num_routed =
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_id_num_routed));

    // Read routing metadata for all experts
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        // Read num_routed_tokens[expert_idx]
        uint64_t num_routed_noc_addr = get_noc_addr(expert_idx, num_routed_addrgen);
        noc_async_read(num_routed_noc_addr,
                        (uint32_t) num_routed,
                        sizeof(uint32_t));
        noc_async_read_barrier();
        num_routed_local[expert_idx] = num_routed[0];
    }
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        num_routed[expert_idx] = num_routed_local[expert_idx];
    }
    cb_push_back(cb_id_num_routed, 1);

    // Reading token_idx (2D tensor of size (E/D, T)) for all local experts
    volatile uint32_t* token_idx[MAX_EXPERTS_PER_DEVICE];
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        cb_reserve_back(cb_id_token_idx, 1);
        uint64_t token_idx_map_noc_addr = get_noc_addr(expert_idx, token_idx_addrgen);
        token_idx[expert_idx] =
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_id_token_idx));
        noc_async_read(token_idx_map_noc_addr,
                       (uint32_t) token_idx[expert_idx],
                       max_tokens * sizeof(uint32_t));
        noc_async_read_barrier();
        cb_push_back(cb_id_token_idx, 1);
    }

    volatile uint16_t* weights[MAX_EXPERTS_PER_DEVICE]; // handle bf16 with uint16_t
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        cb_reserve_back(cb_id_weights, 1);
        uint64_t weights_noc_addr = get_noc_addr(expert_idx, weights_addrgen);
        weights[expert_idx] =
            reinterpret_cast<volatile uint16_t*>(get_write_ptr(cb_id_weights));
        noc_async_read(weights_noc_addr,
                       (uint32_t) weights[expert_idx],
                       max_tokens * sizeof(uint16_t));
        noc_async_read_barrier();
        cb_push_back(cb_id_weights, 1);
    }

    // Process each token assigned to this core
    for (uint32_t tidx = start_token_idx; tidx < end_token_idx; tidx++) {
        // Now read hidden states for matching tokens
        for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
            uint32_t t_e = num_routed[expert_idx];

            // Search for matching token in this expert
            for (uint32_t i = 0; i < t_e; i++) {
                if (token_idx[expert_idx][i] == tidx) {
                    /* Send weight scalar to compute kernel */
                    cb_reserve_back(cb_id_weight_scalar, 1);
                    uint32_t weight_scalar_addr = get_write_ptr(cb_id_weight_scalar);
                    uint16_t *weight_scalar_addr_bf16 = reinterpret_cast<uint16_t *>(weight_scalar_addr);
                    weight_scalar_addr_bf16[0] = weights[expert_idx][i];
                    cb_push_back(cb_id_weight_scalar, 1);

                    /* Send input hidden state to compute kernel */
                    for (uint32_t h = 0; h < hidden_dim_tiles; h++) {
                        cb_reserve_back(cb_id_input, 1);
                        uint32_t hidden_addr = get_write_ptr(cb_id_input);
                        // Read input_hidden_state[expert_idx, i, h*1024:(h+1)*1024]
                        uint32_t input_page_idx = expert_idx * max_tokens + i;
                        uint64_t input_row_noc_addr = get_noc_addr(input_page_idx, input_addrgen) + h * tile_size_bytes;
                        // Print first 3 bfloat16 values of the input tile for debugging
                        noc_async_read(input_row_noc_addr, hidden_addr, tile_size_bytes);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_input, 1);
                    }
                    break;
                }
            }
        }
    }
}
