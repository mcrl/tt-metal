// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

/**
 * Compute Kernel for Local Reduce MoE Output
 *
 * This kernel performs weighted accumulation of expert outputs for each token.
 * It reads routing metadata and hidden states from circular buffers, performs
 * element-wise multiply-accumulate operations, and writes results to output buffer.
 *
 * For each token:
 * 1. Read routing metadata from CB
 * 2. Initialize accumulator to zero
 * 3. For each matching expert output:
 *    - Read hidden state and weight
 *    - Accumulate: output += hidden * weight
 * 4. Pack accumulated result to output CB
 *
 * Compile-time args:
 * - hidden_dim: Hidden dimension (number of bfloat16 elements per row)
 * - num_local_experts: Number of experts on this device
 * - max_tokens: Maximum tokens per expert
 * - max_tokens_per_core: Maximum tokens per core
 *
 * Note: This kernel operates on row-major data (not tiles), performing
 * element-wise operations using manual bfloat16 arithmetic.
 */

#define MAX_EXPERTS_PER_DEVICE 16

namespace NAMESPACE {
void MAIN {
    // Circular buffer IDs (static)
    constexpr tt::CBIndex cb_id_input = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_id_token_idx = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_id_weights = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_id_num_routed = tt::CBIndex::c_3;
    constexpr tt::CBIndex cb_id_weight_scalar = tt::CBIndex::c_4;
    constexpr tt::CBIndex cb_id_accum = tt::CBIndex::c_5;
    constexpr tt::CBIndex cb_id_output = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_id_input_tile = tt::CBIndex::c_24;  // Tilized input (TILE format)

    // Compile-time arguments
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(0);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(1);
    constexpr uint32_t max_tokens = get_compile_time_arg_val(2);
    constexpr uint32_t max_tokens_per_core = get_compile_time_arg_val(3);

    // Runtime arguments
    uint32_t num_tokens_per_core = get_arg_val<uint32_t>(0);
    uint32_t start_token_idx = get_arg_val<uint32_t>(1);

    const uint32_t hidden_dim_tiles = hidden_dim / 1024;

    // Helper function to convert bfloat16 to float
    auto bf16_to_float = [](uint16_t bf16) -> float {
        union {
            uint32_t u;
            float f;
        } converter;
        converter.u = ((uint32_t)bf16) << 16;
        return converter.f;
    };

    // Helper function to convert float to bfloat16
    auto float_to_bf16 = [](float f) -> uint16_t {
        union {
            uint32_t u;
            float f_val;
        } converter;
        converter.f_val = f;
        return (uint16_t)(converter.u >> 16);
    };

    uint32_t num_routed_local[MAX_EXPERTS_PER_DEVICE];
    cb_wait_front(cb_id_num_routed, 1);
    tensix_sync();
    volatile uint32_t* num_routed;
    cb_get_tile(cb_id_num_routed, 0, &num_routed);
    // The first 4 entries have metadata, so we look at the 5th entry
    // for our value pushed from the reader.
    num_routed += 4;

    volatile uint32_t* token_idx[MAX_EXPERTS_PER_DEVICE];
    for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        cb_wait_front(cb_id_token_idx, 1);
        tensix_sync();
        cb_get_tile(cb_id_token_idx, 0, &token_idx[expert_idx]);
        // The first 4 entries have metadata, so we look at the 5th entry
        // for our value pushed from the reader.
        token_idx[expert_idx] += 4;
        cb_pop_front(cb_id_token_idx, 1);
    }

    // Process each token
    uint32_t end_token_idx = start_token_idx + num_tokens_per_core;
    for (uint32_t tidx = start_token_idx; tidx < end_token_idx; tidx++) {
        cb_reserve_back(cb_id_output, 1);

        init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(
            cb_id_input, cb_id_weight_scalar, cb_id_output
        );
        pack_untilize_dest_init<hidden_dim_tiles>(cb_id_output);
    
        acquire_dst();

        // Accumulate contributions from all experts
        for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
            uint32_t t_e = num_routed[expert_idx];

            for (uint32_t i = 0; i < t_e; i++) {
                if (token_idx[expert_idx][i] == tidx) {
                    // Wait for and read hidden state
                    cb_wait_front(cb_id_weight_scalar, 1);
                    for (uint32_t h = 0; h < hidden_dim_tiles; h++) {
                        cb_wait_front(cb_id_input, 1);
                        mul_tiles_bcast_scalar(
                            cb_id_input,  // Vector tile CB
                            cb_id_weight_scalar,  // Scalar tile CB
                            0,          // Vector tile index in CB (0 if single buffered)
                            0,          // Scalar tile index (always 0)
                            h           // Destination register index
                        );
                        cb_pop_front(cb_id_input, 1);
                    }
                    cb_pop_front(cb_id_weight_scalar, 1);
                    break;
                }
            }
        }
        tile_regs_wait();
        pack_untilize_dest<hidden_dim_tiles>(cb_id_output);
        tile_regs_release();
        release_dst();
        cb_push_back(cb_id_output, 1);
    }
}
}  // namespace NAMESPACE
