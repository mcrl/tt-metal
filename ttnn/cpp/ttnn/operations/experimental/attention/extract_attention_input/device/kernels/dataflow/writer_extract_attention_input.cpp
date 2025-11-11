#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t output_data_format = get_compile_time_arg_val(2);

    const uint32_t tile_bytes = get_tile_size(cb_id);

    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Write tiles from [start_tile_id, start_tile_id + num_tiles)
    uint32_t end_tile_id = start_tile_id + num_tiles;
    for (uint32_t tile_idx = start_tile_id; tile_idx < end_tile_id; ++tile_idx) {
        cb_wait_front(cb_id, 1);

        uint32_t l1_read_addr = get_read_ptr(cb_id);

        noc_async_write_tile(tile_idx, accessor, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_id, 1);
    }
}
