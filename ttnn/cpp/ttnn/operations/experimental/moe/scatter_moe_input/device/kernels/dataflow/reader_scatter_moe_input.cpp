#include <stdint.h>
#include "dataflow_api.h"

/**
 * Scatter MoE Input - Reader Kernel
 *
 * Gathers input tokens based on expert assignments and writes to intermediate buffer.
 * Runs on RISCV_0, works in parallel with writer on RISCV_1.
 *
 * For each assigned expert:
 * 1. Read routed_tokens row (T uint32 values)
 * 2. Process tokens in batches of BMt
 * 3. For each batch:
 *    - Gather BMt input rows directly to CB_GATHERED
 *    - Send metadata to writer (expert_idx, batch_size, batch_offset)
 *
 * Compile-time args:
 * - cb_id_routed: CB for routed_tokens row (c_1)
 * - cb_id_num_routed: CB for num_routed_tokens array (c_2)
 * - cb_id_metadata: CB for metadata to writer (c_3)
 * - cb_id_gathered: CB for gathered input rows (c_4)
 * - input_is_dram: Whether input buffer is in DRAM
 * - num_routed_is_dram: Whether num_routed_tokens is in DRAM
 * - routed_tokens_is_dram: Whether routed_tokens is in DRAM
 * - hidden_dim: H - hidden dimension
 * - num_tokens: T - total number of tokens
 * - num_local_experts: E/D - number of experts on this device
 * - row_size_bytes: Byte size of one row (H * element_size)
 * - BMt: Batch size (number of tokens/rows per batch)
 *
 * Runtime args:
 * - input_buffer_addr: Address of input_hidden_state (T, H)
 * - num_routed_tokens_addr: Address of num_routed_tokens (E/D,)
 * - routed_tokens_addr: Address of routed_tokens (E/D, T)
 * - start_expert_idx: First expert index this core processes
 * - num_experts_per_core: Number of experts assigned to this core
 */

void kernel_main() {
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(1);
    uint32_t routed_tokens_addr = get_arg_val<uint32_t>(2);
    uint32_t start_expert_idx = get_arg_val<uint32_t>(3);
    uint32_t num_experts_per_core = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_routed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_num_routed = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_metadata = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_gathered = get_compile_time_arg_val(3);
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool num_routed_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr bool routed_tokens_is_dram = (bool)get_compile_time_arg_val(6);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(8);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(9);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t BMt = get_compile_time_arg_val(11);

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

    // Load num_routed_tokens array (shared with writer via CB2)
    cb_reserve_back(cb_id_num_routed, 1);
    uint32_t num_routed_l1_addr = get_write_ptr(cb_id_num_routed);
    uint64_t num_routed_base_addr = get_noc_addr(0, num_routed_addrgen);
    noc_async_read(num_routed_base_addr, num_routed_l1_addr, num_local_experts * sizeof(uint32_t));
    noc_async_read_barrier();
    cb_push_back(cb_id_num_routed, 1);

    volatile tt_l1_ptr uint32_t* num_routed_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    // Process each assigned expert
    uint32_t end_expert_idx = start_expert_idx + num_experts_per_core;
    for (uint32_t expert_idx = start_expert_idx; expert_idx < end_expert_idx; expert_idx++) {
        uint32_t t_e = num_routed_ptr[expert_idx];

        if (t_e == 0) continue;  // Skip experts with no tokens

        // Read routed_tokens row for this expert
        cb_reserve_back(cb_id_routed, 1);
        uint32_t routed_tokens_l1_addr = get_write_ptr(cb_id_routed);
        uint64_t routed_tokens_row_noc_addr = get_noc_addr(expert_idx, routed_tokens_addrgen);
        noc_async_read(routed_tokens_row_noc_addr, routed_tokens_l1_addr, num_tokens * sizeof(uint32_t));
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* routed_tokens_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(routed_tokens_l1_addr);

        // Process in batches of BMt tokens
        uint32_t num_batches = (t_e + BMt - 1) / BMt;
        for (uint32_t batch = 0; batch < num_batches; batch++) {
            uint32_t batch_start = batch * BMt;
            uint32_t batch_size = (batch == num_batches - 1) ?
                                  (t_e - batch_start) : BMt;

            // Reserve CB_GATHERED for this batch
            cb_reserve_back(cb_id_gathered, batch_size);
            uint32_t gathered_l1_base = get_write_ptr(cb_id_gathered);

            // Gather tokens for this batch - issue all reads before barrier (optimized)
            for (uint32_t i = 0; i < batch_size; i++) {
                uint32_t token_idx = routed_tokens_ptr[batch_start + i];
                uint32_t gathered_l1_addr = gathered_l1_base + i * row_size_bytes;
                uint64_t input_row_noc_addr = get_noc_addr(token_idx, input_addrgen);
                noc_async_read(input_row_noc_addr, gathered_l1_addr, row_size_bytes);
            }

            // Single barrier for entire batch
            noc_async_read_barrier();

            // Push gathered batch
            cb_push_back(cb_id_gathered, batch_size);

            // Send metadata to writer
            cb_reserve_back(cb_id_metadata, 1);
            uint32_t metadata_l1_addr = get_write_ptr(cb_id_metadata);
            uint32_t* metadata_ptr = reinterpret_cast<uint32_t*>(metadata_l1_addr);
            metadata_ptr[0] = expert_idx;
            metadata_ptr[1] = batch_size;    // Actual tokens in this batch
            metadata_ptr[2] = batch_start;   // Starting token index in expert
            cb_push_back(cb_id_metadata, 1);
        }

        cb_pop_front(cb_id_routed, 1);
    }

    cb_pop_front(cb_id_num_routed, 1);
}
