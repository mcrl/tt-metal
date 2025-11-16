#include <stdint.h>
#include "dataflow_api.h"

/**
 * Scatter MoE Input - Writer Kernel
 *
 * Writes gathered tokens to output tensor based on expert assignments.
 * Runs on RISCV_1, works in parallel with reader on RISCV_0.
 *
 * For each assigned expert:
 * 1. For each batch:
 *    - Wait for metadata from reader (expert_idx, batch_size, batch_offset)
 *    - Wait for gathered tokens in CB_GATHERED
 *    - Write batch to output[expert_idx, batch_offset:batch_offset+batch_size, :]
 *
 * Compile-time args:
 * - cb_id_num_routed: CB for num_routed_tokens array (c_2, shared with reader)
 * - cb_id_metadata: CB for metadata from reader (c_3)
 * - cb_id_gathered: CB for gathered input rows (c_4)
 * - output_is_dram: Whether output buffer is in DRAM
 * - num_tokens: T - total number of tokens
 * - num_local_experts: E/D - number of experts on this device
 * - row_size_bytes: Byte size of one row (H * element_size)
 * - BMt: Batch size (number of tokens/rows per batch)
 *
 * Runtime args:
 * - output_buffer_addr: Address of output (E/D, T, H)
 * - start_expert_idx: First expert index this core processes
 * - num_experts_per_core: Number of experts assigned to this core
 */

void kernel_main() {
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t start_expert_idx = get_arg_val<uint32_t>(1);
    uint32_t num_experts_per_core = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_num_routed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_metadata = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_gathered = get_compile_time_arg_val(2);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(4);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(5);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t BMt = get_compile_time_arg_val(7);

    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };

    // Wait for num_routed_tokens from reader
    cb_wait_front(cb_id_num_routed, 1);
    uint32_t num_routed_l1_addr = get_read_ptr(cb_id_num_routed);
    volatile tt_l1_ptr uint32_t* num_routed_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_routed_l1_addr);

    // Process each assigned expert
    uint32_t end_expert_idx = start_expert_idx + num_experts_per_core;
    for (uint32_t expert_idx = start_expert_idx; expert_idx < end_expert_idx; expert_idx++) {
        uint32_t t_e = num_routed_ptr[expert_idx];

        if (t_e == 0) continue;  // Skip experts with no tokens

        // Process batches for this expert
        uint32_t num_batches = (t_e + BMt - 1) / BMt;
        for (uint32_t batch = 0; batch < num_batches; batch++) {
            // Wait for metadata from reader
            cb_wait_front(cb_id_metadata, 1);
            uint32_t metadata_l1_addr = get_read_ptr(cb_id_metadata);
            volatile tt_l1_ptr uint32_t* metadata_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_l1_addr);

            uint32_t recv_expert_idx = metadata_ptr[0];
            uint32_t batch_size = metadata_ptr[1];
            uint32_t batch_start = metadata_ptr[2];

            cb_pop_front(cb_id_metadata, 1);

            // Wait for gathered tokens from reader
            cb_wait_front(cb_id_gathered, batch_size);
            uint32_t gathered_l1_base = get_read_ptr(cb_id_gathered);

            // Write gathered tokens to output - issue all writes before barrier
            for (uint32_t i = 0; i < batch_size; i++) {
                uint32_t gathered_l1_addr = gathered_l1_base + i * row_size_bytes;

                // Output page: output[expert_idx, batch_start + i, :]
                uint32_t output_page_idx = expert_idx * num_tokens + batch_start + i;
                uint64_t output_row_noc_addr = get_noc_addr(output_page_idx, output_addrgen);

                noc_async_write(gathered_l1_addr, output_row_noc_addr, row_size_bytes);
            }

            // Single barrier for entire batch
            noc_async_write_barrier();

            cb_pop_front(cb_id_gathered, batch_size);
        }
    }

    cb_pop_front(cb_id_num_routed, 1);
}
