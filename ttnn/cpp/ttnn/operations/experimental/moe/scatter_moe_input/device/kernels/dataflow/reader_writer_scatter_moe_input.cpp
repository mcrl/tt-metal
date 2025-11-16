#include <stdint.h>
#include "dataflow_api.h"

/**
 * Scatter MoE Input Kernel
 *
 * This kernel rearranges input tokens based on expert assignments.
 * Each core processes a subset of local experts assigned to it.
 * Optimization: Reads entire num_routed_tokens tensor once at the beginning.
 * For each assigned expert, it:
 * 1. Gets the number of assigned tokens from pre-loaded L1 buffer
 * 2. Gathers those tokens from input_hidden_state
 * 3. Writes them to output sequentially
 * 4. Zero-pads remaining positions
 *
 * Compile-time args:
 * - cb_id_input: Circular buffer ID for input rows
 * - cb_id_output: Circular buffer ID for output (zero-padding)
 * - cb_id_num_routed: Circular buffer ID for num_routed_tokens (entire 1D tensor)
 * - input_is_dram: Whether input buffer is in DRAM
 * - num_routed_is_dram: Whether num_routed_tokens is in DRAM
 * - routed_tokens_is_dram: Whether routed_tokens is in DRAM
 * - output_is_dram: Whether output buffer is in DRAM
 * - hidden_dim: H - hidden dimension
 * - num_tokens: T - total number of tokens
 * - num_local_experts: E/D - number of experts on this device (total, not per-core)
 * - row_size_bytes: Byte size of one row (H * element_size)
 *
 * Runtime args:
 * - input_buffer_addr: Address of input_hidden_state (T, H)
 * - num_routed_tokens_addr: Address of num_routed_tokens (E/D,) - 1D tensor
 * - routed_tokens_addr: Address of routed_tokens (E/D, T)
 * - output_buffer_addr: Address of output (E/D, T, H)
 * - start_expert_idx: First expert index this core processes
 * - num_experts_per_core: Number of experts assigned to this core
 */

void kernel_main() {
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(1);
    uint32_t routed_tokens_addr = get_arg_val<uint32_t>(2);
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(3);
    uint32_t start_expert_idx = get_arg_val<uint32_t>(4);
    uint32_t num_experts_per_core = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_num_routed = get_compile_time_arg_val(2);
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool num_routed_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool routed_tokens_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(6);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(8);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(9);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(10);

    const InterleavedAddrGen<input_is_dram> input_addrgen = {
        .bank_base_address = input_buffer_addr,
        .page_size = row_size_bytes
    };

    const InterleavedAddrGen<num_routed_is_dram> num_routed_addrgen = {
        .bank_base_address = num_routed_tokens_addr,
        .page_size = num_local_experts * sizeof(uint32_t)
    };

    const InterleavedAddrGen<routed_tokens_is_dram> routed_tokens_addrgen = {
        .bank_base_address = routed_tokens_addr,
        .page_size = num_tokens * sizeof(uint32_t)
    };

    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };

    cb_reserve_back(cb_id_num_routed, 1);
    uint32_t num_routed_l1_addr = get_write_ptr(cb_id_num_routed);

    uint64_t num_routed_base_addr = get_noc_addr(0, num_routed_addrgen);  // Page 0
    noc_async_read(num_routed_base_addr, num_routed_l1_addr, num_local_experts * sizeof(uint32_t));
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* num_routed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    // Process each expert assigned to this core
    uint32_t end_expert_idx = start_expert_idx + num_experts_per_core;
    for (uint32_t expert_idx = start_expert_idx; expert_idx < end_expert_idx; expert_idx++) {
        // Step 1: Get num_routed_tokens[expert_idx] from L1 buffer
        uint32_t t_e = num_routed_ptr[expert_idx];  // Number of tokens for this expert

        // Step 2: Read routed_tokens row for this expert (E/D rows, each of size T * sizeof(uint32_t))
        cb_reserve_back(cb_id_output, 1);
        uint32_t routed_tokens_l1_addr = get_write_ptr(cb_id_output);

        uint64_t routed_tokens_row_noc_addr = get_noc_addr(expert_idx, routed_tokens_addrgen);
        noc_async_read(routed_tokens_row_noc_addr, routed_tokens_l1_addr, num_tokens * sizeof(uint32_t));
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* routed_tokens_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routed_tokens_l1_addr);

        // Step 3: Gather assigned tokens and write to output
        for (uint32_t i = 0; i < t_e; i++) {
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
            noc_async_write_barrier();

            cb_pop_front(cb_id_input, 1);
        }
        cb_pop_front(cb_id_output, 1);
    }
    cb_pop_front(cb_id_num_routed, 1);
}
