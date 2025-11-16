#include <stdint.h>
#include "dataflow_api.h"

/**
 * Local Reduce MoE Output Kernel (Multi-Core, Token-Parallel)
 *
 * This kernel performs intra-device reduction by gathering expert outputs
 * back to token order and applying routing weights.
 *
 * For each global token index t:
 * 1. Initialize output[t, :] = 0
 * 2. For each local expert e:
 *    - For each expert-local position i with valid data:
 *      - If token_idx_map[e, i] == t:
 *        - Read hidden = input_hidden_state[e, i, :]
 *        - Read weight = routed_token_weights[e, i]
 *        - Accumulate: output[t, :] += hidden * weight
 *
 * Compile-time args:
 * - cb_id_input: Circular buffer ID for input rows
 * - cb_id_output: Circular buffer ID for output accumulator
 * - cb_id_weight: Circular buffer ID for weight scalar
 * - input_is_dram: Whether input buffer is in DRAM
 * - token_idx_is_dram: Whether token_idx_map is in DRAM
 * - weights_is_dram: Whether routed_token_weights is in DRAM
 * - num_routed_is_dram: Whether num_routed_tokens is in DRAM
 * - output_is_dram: Whether output buffer is in DRAM
 * - hidden_dim: H - hidden dimension
 * - num_tokens: T - total number of tokens
 * - num_local_experts: E/D - number of experts on this device
 * - max_tokens: T - maximum tokens dimension in input tensor
 * - row_size_bytes: Byte size of one row (H * element_size)
 *
 * Runtime args:
 * - input_buffer_addr: Address of input_hidden_state (E/D, T, H)
 * - token_idx_map_addr: Address of token_idx_map (E/D, T)
 * - weights_addr: Address of routed_token_weights (E/D, T)
 * - num_routed_tokens_addr: Address of num_routed_tokens (E/D, 1)
 * - output_buffer_addr: Address of output (T, H)
 */

void kernel_main() {
    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t token_idx_map_addr = get_arg_val<uint32_t>(1);
    uint32_t weights_addr = get_arg_val<uint32_t>(2);
    uint32_t num_routed_tokens_addr = get_arg_val<uint32_t>(3);
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(4);

    uint32_t start_token_idx = get_arg_val<uint32_t>(5);
    uint32_t end_token_idx = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(2);
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool token_idx_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool weights_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr bool num_routed_is_dram = (bool)get_compile_time_arg_val(6);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(7);
    constexpr uint32_t hidden_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(9);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(10);
    constexpr uint32_t max_tokens = get_compile_time_arg_val(11);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(12);

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
        .page_size = max_tokens * sizeof(uint16_t)
    };

    const InterleavedAddrGen<num_routed_is_dram> num_routed_addrgen = {
        .bank_base_address = num_routed_tokens_addr,
        .page_size = sizeof(uint32_t)
    };

    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };

    // Token-stationary
    for (uint32_t token_idx = start_token_idx; token_idx < end_token_idx; token_idx++) {
        cb_reserve_back(cb_id_output, 1);
        uint32_t accumulator_l1_addr = get_write_ptr(cb_id_output);

        volatile tt_l1_ptr uint16_t* accumulator_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(accumulator_l1_addr);

        for (uint32_t h = 0; h < hidden_dim; h++) {
            accumulator_ptr[h] = 0;
        }

        for (uint32_t expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
            // Read num_routed_tokens[expert_idx]
            cb_reserve_back(cb_id_weight, 1);
            uint32_t temp_addr = get_write_ptr(cb_id_weight);

            uint64_t num_routed_noc_addr = get_noc_addr(expert_idx, num_routed_addrgen);
            noc_async_read(num_routed_noc_addr, temp_addr, sizeof(uint32_t));
            noc_async_read_barrier();

            volatile tt_l1_ptr uint32_t* temp_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(temp_addr);
            uint32_t t_e = temp_ptr[0];

            cb_pop_front(cb_id_weight, 1);

            // Read token_idx_map row for this expert
            cb_reserve_back(cb_id_weight, 1);
            uint32_t token_idx_map_l1_addr = get_write_ptr(cb_id_weight);

            uint64_t token_idx_map_noc_addr = get_noc_addr(expert_idx, token_idx_addrgen);
            noc_async_read(token_idx_map_noc_addr, token_idx_map_l1_addr, max_tokens * sizeof(uint32_t));
            noc_async_read_barrier();

            volatile tt_l1_ptr uint32_t* token_idx_map_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_idx_map_l1_addr);

            // Read weights row for this expert
            uint64_t weights_row_noc_addr = get_noc_addr(expert_idx, weights_addrgen);
            noc_async_read(weights_row_noc_addr, token_idx_map_l1_addr + max_tokens * sizeof(uint32_t), max_tokens * sizeof(uint16_t));
            noc_async_read_barrier();

            volatile tt_l1_ptr uint16_t* weights_row_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(token_idx_map_l1_addr + max_tokens * sizeof(uint32_t));

            // Search for matching token
            for (uint32_t i = 0; i < t_e; i++) {
                if (token_idx_map_ptr[i] == token_idx) {
                    cb_reserve_back(cb_id_input, 1);
                    uint32_t hidden_l1_addr = get_write_ptr(cb_id_input);

                    uint32_t input_page_idx = expert_idx * max_tokens + i;
                    uint64_t input_row_noc_addr = get_noc_addr(input_page_idx, input_addrgen);
                    noc_async_read(input_row_noc_addr, hidden_l1_addr, row_size_bytes);
                    noc_async_read_barrier();

                    volatile tt_l1_ptr uint16_t* hidden_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(hidden_l1_addr);

                    uint16_t weight_bf16 = weights_row_ptr[i];

                    union {
                        uint32_t u;
                        float f;
                    } weight_converter;
                    weight_converter.u = ((uint32_t)weight_bf16) << 16;
                    float weight = weight_converter.f;

                    for (uint32_t h = 0; h < hidden_dim; h++) {
                        union {
                            uint32_t u;
                            float f;
                        } hidden_converter, accum_converter;

                        hidden_converter.u = ((uint32_t)hidden_ptr[h]) << 16;
                        accum_converter.u = ((uint32_t)accumulator_ptr[h]) << 16;

                        float result = accum_converter.f + hidden_converter.f * weight;

                        union {
                            uint32_t u;
                            float f;
                        } result_converter;
                        result_converter.f = result;
                        accumulator_ptr[h] = (uint16_t)(result_converter.u >> 16);
                    }

                    cb_pop_front(cb_id_input, 1);

                    break;
                }
            }
            cb_pop_front(cb_id_weight, 1);
        }

        // Write accumulated result to output[token_idx, :]
        uint64_t output_row_noc_addr = get_noc_addr(token_idx, output_addrgen);
        noc_async_write(accumulator_l1_addr, output_row_noc_addr, row_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_output, 1);
    }
}
