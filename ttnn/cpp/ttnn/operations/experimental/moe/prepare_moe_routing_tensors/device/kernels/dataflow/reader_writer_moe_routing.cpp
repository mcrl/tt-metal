#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

// This kernel prepares device-local MoE routing tensors for efficient expert-parallel computation
// Each core processes one local expert independently
//
// Input:
// - selected_experts: ROW_MAJOR, uint32, shape (num_tokens, top_k)
//   Contains GLOBAL expert indices selected for each token (no duplicates per token)
// - routing_weights: ROW_MAJOR, bfloat16, shape (num_tokens, top_k)
//   Contains weights for each selected expert
// - device_expert_mapping: ROW_MAJOR, int32, shape (num_local_experts)
//   Contains GLOBAL expert indices assigned to this device
//
// Output (device-local):
// - num_routed_tokens: ROW_MAJOR, uint32, shape (num_local_experts)
//   Count of tokens routed to each LOCAL expert
// - routed_tokens: ROW_MAJOR, uint32, shape (num_local_experts, max_tokens_per_expert)
//   Token indices for each LOCAL expert (padded with invalid values)
// - routed_token_weights: ROW_MAJOR, bfloat16, shape (num_local_experts, max_tokens_per_expert)
//   Routing weights for each LOCAL expert (padded with zeros)
// - token_idx_map: ROW_MAJOR, uint32, shape (num_local_experts, max_tokens_per_expert)
//   Mapping from expert-local token index to global token index
//   For expert e, token_idx_map[e][t_e] = t_g
//   where t_e is the local index (0 to num_routed_tokens[e]-1) and t_g is the global token index
//
// Algorithm (per core):
// 1. Load global expert ID for this core's assigned local expert
// 2. Scan all tokens, collect matches for this expert only
// 3. Write outputs for this expert only (no loops over experts)

void kernel_main() {
    const uint32_t selected_experts_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t device_expert_mapping_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(3);
    const uint32_t routed_tokens_addr = get_arg_val<uint32_t>(4);
    const uint32_t routed_token_weights_addr = get_arg_val<uint32_t>(5);
    const uint32_t token_idx_map_addr = get_arg_val<uint32_t>(6);
    const uint32_t num_tokens = get_arg_val<uint32_t>(7);
    const uint32_t top_k = get_arg_val<uint32_t>(8);
    const uint32_t num_experts = get_arg_val<uint32_t>(9);
    const uint32_t num_local_experts = get_arg_val<uint32_t>(10);
    const uint32_t max_tokens_per_expert = get_arg_val<uint32_t>(11);
    const uint32_t local_expert_id = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_experts = tt::CBIndex::c_0;
    constexpr uint32_t cb_weights = tt::CBIndex::c_1;
    constexpr uint32_t cb_device_mapping = tt::CBIndex::c_2;
    constexpr uint32_t cb_num_routed = tt::CBIndex::c_16;
    constexpr uint32_t cb_routed_tokens = tt::CBIndex::c_17;
    constexpr uint32_t cb_routed_weights = tt::CBIndex::c_18;
    constexpr uint32_t cb_tokenidx_map = tt::CBIndex::c_19;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;

    constexpr auto experts_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weights_accessor_args = TensorAccessorArgs<experts_accessor_args.next_compile_time_args_offset()>();
    constexpr auto mapping_accessor_args = TensorAccessorArgs<weights_accessor_args.next_compile_time_args_offset()>();
    constexpr auto num_routed_accessor_args = TensorAccessorArgs<mapping_accessor_args.next_compile_time_args_offset()>();
    constexpr auto routed_tokens_accessor_args = TensorAccessorArgs<num_routed_accessor_args.next_compile_time_args_offset()>();
    constexpr auto routed_weights_accessor_args = TensorAccessorArgs<routed_tokens_accessor_args.next_compile_time_args_offset()>();
    constexpr auto tokenidx_map_accessor_args = TensorAccessorArgs<routed_weights_accessor_args.next_compile_time_args_offset()>();

    const auto experts_accessor = TensorAccessor(experts_accessor_args, selected_experts_addr, top_k * sizeof(uint32_t));
    const auto weights_accessor = TensorAccessor(weights_accessor_args, routing_weights_addr, top_k * sizeof(uint16_t));
    const auto mapping_accessor = TensorAccessor(mapping_accessor_args, device_expert_mapping_addr, num_local_experts * sizeof(int32_t));
    const auto num_routed_accessor = TensorAccessor(num_routed_accessor_args, num_routed_tokens_addr, num_local_experts * sizeof(uint32_t));
    const auto routed_tokens_accessor = TensorAccessor(routed_tokens_accessor_args, routed_tokens_addr, max_tokens_per_expert * sizeof(uint32_t));
    const auto routed_weights_accessor = TensorAccessor(routed_weights_accessor_args, routed_token_weights_addr, max_tokens_per_expert * sizeof(uint16_t));
    const auto tokenidx_map_accessor = TensorAccessor(tokenidx_map_accessor_args, token_idx_map_addr, max_tokens_per_expert * sizeof(uint32_t));

    cb_reserve_back(cb_experts, 1);
    cb_reserve_back(cb_weights, 1);
    cb_reserve_back(cb_device_mapping, 1);
    cb_reserve_back(cb_scratch, 1);

    uint32_t l1_experts_addr = get_write_ptr(cb_experts);
    uint32_t l1_weights_addr = get_write_ptr(cb_weights);
    uint32_t l1_device_mapping_addr = get_write_ptr(cb_device_mapping);
    uint32_t l1_scratch_addr = get_write_ptr(cb_scratch);

    uint64_t mapping_noc_addr = get_noc_addr(0, mapping_accessor);
    noc_async_read(mapping_noc_addr, l1_device_mapping_addr, num_local_experts * sizeof(int32_t));
    noc_async_read_barrier();

    int32_t* device_expert_mapping = reinterpret_cast<int32_t*>(l1_device_mapping_addr);
    uint32_t global_expert_id = static_cast<uint32_t>(device_expert_mapping[local_expert_id]);

    // Use scratch buffer to store intermediate routing data for this expert only
    uint32_t* scratch_tokens = reinterpret_cast<uint32_t*>(l1_scratch_addr);
    uint16_t* scratch_weights = reinterpret_cast<uint16_t*>(l1_scratch_addr + max_tokens_per_expert * sizeof(uint32_t));

    uint32_t token_count = 0;

    // Scan all tokens and collect matches for this expert
    for (uint32_t token_idx = 0; token_idx < num_tokens; token_idx++) {
        // Read selected experts for this token
        uint64_t experts_noc_addr = get_noc_addr(token_idx, experts_accessor);
        noc_async_read(experts_noc_addr, l1_experts_addr, top_k * sizeof(uint32_t));

        // Read routing weights for this token
        uint64_t weights_noc_addr = get_noc_addr(token_idx, weights_accessor);
        noc_async_read(weights_noc_addr, l1_weights_addr, top_k * sizeof(uint16_t));
        noc_async_read_barrier();

        volatile uint32_t* expert_indices = reinterpret_cast<volatile uint32_t*>(l1_experts_addr);
        volatile uint16_t* weights = reinterpret_cast<volatile uint16_t*>(l1_weights_addr);

        // Check if this token selects this expert
        for (uint32_t k = 0; k < top_k; k++) {
            uint32_t selected_expert_id = expert_indices[k];
            if (selected_expert_id == global_expert_id) {
                if (token_count < max_tokens_per_expert) {
                    scratch_tokens[token_count] = token_idx;
                    scratch_weights[token_count] = weights[k];
                    token_count++;
                }
                break;
            }
        }
    }

    // Write num_routed_tokens for this expert
    cb_reserve_back(cb_num_routed, 1);
    uint32_t l1_num_routed_addr = get_write_ptr(cb_num_routed);
    uint32_t* num_routed_ptr = reinterpret_cast<uint32_t*>(l1_num_routed_addr);
    num_routed_ptr[local_expert_id % 4] = token_count;

    uint64_t num_routed_base_addr = get_noc_addr(0, num_routed_accessor);
    uint64_t num_routed_noc_addr = num_routed_base_addr + local_expert_id * sizeof(uint32_t);
    noc_async_write(l1_num_routed_addr + (local_expert_id % 4) * sizeof(uint32_t),
                    num_routed_noc_addr,
                    sizeof(uint32_t));
    noc_async_write_barrier();

    // Write routed_tokens and routed_token_weights for this expert
    cb_reserve_back(cb_routed_tokens, 1);
    cb_reserve_back(cb_routed_weights, 1);
    cb_reserve_back(cb_tokenidx_map, 1);
    uint32_t l1_routed_tokens_addr = get_write_ptr(cb_routed_tokens);
    uint32_t l1_routed_weights_addr = get_write_ptr(cb_routed_weights);
    uint32_t l1_tokenidx_map_addr = get_write_ptr(cb_tokenidx_map);

    // Prepare routed_tokens row with padding
    uint32_t* routed_tokens_row = reinterpret_cast<uint32_t*>(l1_routed_tokens_addr);
    uint16_t* routed_weights_row = reinterpret_cast<uint16_t*>(l1_routed_weights_addr);
    uint32_t* tokenidx_map_row = reinterpret_cast<uint32_t*>(l1_tokenidx_map_addr);

    for (uint32_t i = 0; i < max_tokens_per_expert; i++) {
        if (i < token_count) {
            routed_tokens_row[i] = scratch_tokens[i];
            routed_weights_row[i] = scratch_weights[i];
            tokenidx_map_row[i] = scratch_tokens[i];  // The global token index is the same as what we stored in scratch_tokens
        }
        // else {
        //     routed_tokens_row[i] = 0xFFFFFFFF;  // Invalid token index
        //     routed_weights_row[i] = 0;          // Zero weight for padding
        //     tokenidx_map_row[i] = 0xFFFFFFFF;   // Invalid mapping for padding
        // }
    }

    uint64_t routed_tokens_noc_addr = get_noc_addr(local_expert_id, routed_tokens_accessor);
    noc_async_write(l1_routed_tokens_addr, routed_tokens_noc_addr, max_tokens_per_expert * sizeof(uint32_t));

    uint64_t routed_weights_noc_addr = get_noc_addr(local_expert_id, routed_weights_accessor);
    noc_async_write(l1_routed_weights_addr, routed_weights_noc_addr, max_tokens_per_expert * sizeof(uint16_t));

    uint64_t tokenidx_map_noc_addr = get_noc_addr(local_expert_id, tokenidx_map_accessor);
    noc_async_write(l1_tokenidx_map_addr, tokenidx_map_noc_addr, max_tokens_per_expert * sizeof(uint32_t));

    noc_async_write_barrier();
}