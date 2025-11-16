#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t dp_degree_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t local_tile_offset = get_arg_val<uint32_t>(3);
    uint32_t tiles_per_device = get_arg_val<uint32_t>(4);

    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr bool dp_degree_is_dram = (bool)get_compile_time_arg_val(1);

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t cb_dp_degree = 1;

    const uint32_t tile_bytes = get_tile_size(cb_id);

    // Read dp_degree value (4 bytes) to calculate global start offset
    cb_reserve_back(cb_dp_degree, 1);
    uint32_t dp_degree_l1_addr = get_write_ptr(cb_dp_degree);
    const InterleavedAddrGenFast<dp_degree_is_dram> dp_degree_addrgen = {
        .bank_base_address = dp_degree_addr,
        .page_size = sizeof(uint32_t),
        .data_format = DataFormat::UInt32
    };

    uint64_t dp_degree_noc_addr = get_noc_addr(0, dp_degree_addrgen);
    noc_async_read(dp_degree_noc_addr, dp_degree_l1_addr, sizeof(uint32_t));
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* dp_degree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dp_degree_l1_addr);
    uint32_t dp_degree_value = dp_degree_ptr[0];

    cb_push_back(cb_dp_degree, 1);
    cb_pop_front(cb_dp_degree, 1);

    // Calculate global start tile for this device
    // global_start_tile = dp_degree_value * tiles_per_device + local_tile_offset
    uint32_t global_start_tile = dp_degree_value * tiles_per_device + local_tile_offset;

    const auto accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    // Read this core's assigned tiles from [global_start_tile, global_start_tile + num_tiles)
    uint32_t end_tile_id = global_start_tile + num_tiles;
    for (uint32_t tile_idx = global_start_tile; tile_idx < end_tile_id; ++tile_idx) {
        cb_reserve_back(cb_id, 1);

        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read_tile(tile_idx, accessor, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_id, 1);
    }
}
