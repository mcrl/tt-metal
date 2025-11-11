#include <stdint.h>
#include "dataflow_api.h"

/**
 * Writer Kernel for Local Reduce MoE Output (Multi-Core, Token-Parallel)
 *
 * This kernel reads accumulated results from the compute kernel and writes
 * them to the output tensor in DRAM.
 *
 * For each token assigned to this core:
 * 1. Wait for accumulated result from compute kernel
 * 2. Convert tile layout to row-major format
 * 3. Write result to output[token_idx, :] in DRAM
 *
 * Compile-time args:
 * - output_is_dram: Whether output buffer is in DRAM
 * - row_size_bytes: Byte size of one output row (H * element_size)
 *
 * Runtime args:
 * - output_buffer_addr: Address of output tensor (T, H)
 * - start_token_idx: First token index for this core
 * - end_token_idx: Last token index (exclusive) for this core
 */

/**
 * Convert tile layout to row-major format
 * Tile layout: 4 faces (16x16 each) arranged as [Face0][Face1][Face2][Face3]
 * Face 0: rows 0-15,  cols 0-15  (top-left)
 * Face 1: rows 0-15,  cols 16-31 (top-right)
 * Face 2: rows 16-31, cols 0-15  (bottom-left)
 * Face 3: rows 16-31, cols 16-31 (bottom-right)
 */
void tile_to_row_major(
    const uint16_t* src_ptr,  // Tile layout input
    uint16_t* dst_ptr         // Row-major output
) {
    constexpr uint32_t FACE_SIZE = 16;
    constexpr uint32_t TILE_SIZE = 32;
    constexpr uint32_t FACE_ELEMENTS = FACE_SIZE * FACE_SIZE;

    // Pointers to each face
    const uint16_t* face0 = src_ptr;
    const uint16_t* face1 = src_ptr + FACE_SIZE;
    const uint16_t* face2 = src_ptr + FACE_SIZE * TILE_SIZE;
    const uint16_t* face3 = src_ptr + FACE_SIZE * TILE_SIZE + FACE_SIZE;

    const uint16_t* faces[4] = { face0, face1, face2, face3 };

    uint32_t dst_offset = 0;
    for (uint32_t face_idx = 0; face_idx < 4; face_idx++) {
        const uint16_t* face = faces[face_idx];
        for (uint32_t row = 0; row < FACE_SIZE; row++) {
            for (uint32_t col = 0; col < FACE_SIZE; col++) {
                dst_ptr[dst_offset++] = face[row * TILE_SIZE + col];
            }
        }
    }
}

void kernel_main() {
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t start_token_idx = get_arg_val<uint32_t>(1);
    uint32_t end_token_idx = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_id_output = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_id_temp = tt::CBIndex::c_7;

    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t row_size_bytes = get_compile_time_arg_val(1);

    constexpr uint32_t tile_size = 32 * 32;
    constexpr uint32_t hidden_dim_tiles = row_size_bytes / (tile_size * sizeof(uint16_t));
    constexpr uint32_t tile_size_bytes = tile_size * sizeof(uint16_t);

    const InterleavedAddrGen<output_is_dram> output_addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = row_size_bytes
    };

    cb_reserve_back(cb_id_temp, 1);
    uint32_t temp_addr = get_write_ptr(cb_id_temp);
    uint16_t* row_major_buffer = reinterpret_cast<uint16_t*>(temp_addr);

    for (uint32_t token_idx = start_token_idx; token_idx < end_token_idx; token_idx++) {
        uint64_t output_row_noc_addr = get_noc_addr(token_idx, output_addrgen);

        cb_wait_front(cb_id_output, hidden_dim_tiles);
        for (uint32_t h = 0; h < hidden_dim_tiles; h++) {
            uint32_t output_l1_addr = get_read_ptr(cb_id_output);
            uint64_t output_tile_noc_addr = get_noc_addr(token_idx, output_addrgen) + h * tile_size_bytes;
            noc_async_write(output_l1_addr, output_tile_noc_addr, tile_size_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_output, 1);
        }
    }
}
