# Tenstorrent Hardware: Tensix Core Architecture

## Overview

The Tensix core is the fundamental compute unit in Tenstorrent AI accelerators. Each Tensix core contains 5 RISC-V processors, 1.5 MB of shared L1 SRAM, dedicated compute engines, and dual Network-on-Chip (NoC) interfaces for high-bandwidth data movement.

## 1. Five RISC-V Cores and Their Roles

Each Tensix core contains **5 specialized RISC-V processors** with distinct responsibilities:

### BRISC (Binary RISC) - Control Processor

From [`tt_metal/hw/inc/wormhole/dev_mem_map.h`](../../../../../../tt_metal/hw/inc/wormhole/dev_mem_map.h):

```cpp
// BRISC Memory Layout
#define BRISC_MEM_MAP__LOCAL_BASE          0xFFB00000  // Local memory base
#define BRISC_MEM_MAP__LOCAL_SIZE          4096        // 4 KB
#define BRISC_MEM_MAP__FW_SIZE              5632       // 5.5 KB firmware
#define BRISC_MEM_MAP__KERNEL_SIZE         49152       // 48 KB kernel space
```

**Role:**
- Main control processor and kernel coordinator
- Manages overall program execution
- Coordinates Trisc cores for compute pipeline
- Handles program dispatch and synchronization

### NCRISC (Network Controller RISC) - Data Movement

```cpp
// NCRISC Memory Layout
#define NCRISC_MEM_MAP__LOCAL_BASE         0xFFB01000  // Local memory base
#define NCRISC_MEM_MAP__LOCAL_SIZE         4096        // 4 KB (Wormhole)
#define NCRISC_MEM_MAP__IRAM_BASE          0xFFC00000  // Instruction RAM
#define NCRISC_MEM_MAP__IRAM_SIZE          16384       // 16 KB
```

**Role:**
- Dedicated data movement processor
- Manages NOC (Network-on-Chip) transfers
- Coordinates DRAM/L1 data movement
- Has dedicated IRAM for performance

### TRISC (Tensix RISC) × 3 - Compute Pipeline

```cpp
// TRISC Memory Layout (3 cores: TRISC0, TRISC1, TRISC2)
#define TRISC0_MEM_MAP__LOCAL_BASE         0xFFB02000  // TRISC0 local
#define TRISC1_MEM_MAP__LOCAL_BASE         0xFFB02800  // TRISC1 local
#define TRISC2_MEM_MAP__LOCAL_BASE         0xFFB03000  // TRISC2 local
#define TRISC_MEM_MAP__LOCAL_SIZE          2048        // 2 KB each
#define TRISC_MEM_MAP__KERNEL_SIZE         24576       // 24 KB kernel each
```

**Roles:**
- **TRISC0**: Typically handles unpacking (reading from CBs, format conversion)
- **TRISC1**: Controls math/compute operations
- **TRISC2**: Manages packing (writing results to CBs)

### Architecture Comparison

| Architecture | L1 Size | BRISC Local | NCRISC Local | TRISC Local | NCRISC IRAM |
|-------------|---------|-------------|--------------|-------------|-------------|
| Wormhole B0 | 1464 KB | 4 KB | 4 KB | 2 KB each | 16 KB |
| Blackhole | 1536 KB | 8 KB | 8 KB | 4 KB each | 16 KB |

## 2. 1.5 MB Local SRAM Organization

### Memory Map Structure

From [`tt_metal/hw/inc/wormhole/dev_mem_map.h`](../../../../../../tt_metal/hw/inc/wormhole/dev_mem_map.h):

```cpp
// L1 Memory Organization (Wormhole B0: 1464 KB total)
0x00000000 - 0x00000010: Boot code, NOC atomic return, L1 barrier
0x00000010 - 0x00003180: Mailbox (12,656 bytes) - host/FW/kernel communication
0x00003180 - 0x00005xxx: Firmware sections
    ├─ BRISC firmware (5.5 KB)
    ├─ NCRISC firmware (2 KB)
    └─ TRISC0/1/2 firmware (1.5 KB each)
0x00005xxx - 0x00006xxx: NOC counters, routing tables, fabric connections
0x00006xxx - 0x00007xxx: Packet header pool for networking
0x00007xxx - 0x16FFFF:   User space
    ├─ Circular buffers (32 available)
    ├─ Kernel code space
    └─ Kernel data buffers
```

### Circular Buffers - The Core L1 Organization

From [`tt_metal/api/tt-metalium/circular_buffer_config.hpp`](../../../../../../tt_metal/api/tt-metalium/circular_buffer_config.hpp):

```cpp
#define NUM_CIRCULAR_BUFFERS 32

// Standard CB allocation pattern:
// CB 0-7:   Input buffers
// CB 16-23: Output buffers
// CB 24-31: Intermediate/scratch buffers

class CircularBufferConfig {
    uint32_t total_size_;                              // Total size in bytes
    std::optional<uint32_t> globally_allocated_address_; // Fixed L1 address
    std::map<uint8_t, DataFormat> data_format_map_;   // Format per CB index

    CircularBufferConfig& set_page_size(uint8_t cb_id, uint32_t page_size) {
        page_sizes_[cb_id] = page_size;
        return *this;
    }
};
```

### Producer-Consumer Synchronization

Circular buffers implement a lock-free producer-consumer pattern:

```cpp
// Producer (e.g., reader kernel on NCRISC)
void producer_kernel() {
    cb_reserve_back(cb_id, n_tiles);      // Reserve space (blocks if full)
    uint32_t l1_addr = get_write_ptr(cb_id);
    // Write data to l1_addr...
    noc_async_read(dram_addr, l1_addr, size);
    noc_async_read_barrier();
    cb_push_back(cb_id, n_tiles);         // Mark data available
}

// Consumer (e.g., compute kernel on TRISC)
void consumer_kernel() {
    cb_wait_front(cb_id, n_tiles);        // Wait for data (blocks if empty)
    uint32_t l1_addr = get_read_ptr(cb_id);
    // Process data from l1_addr...
    matmul_tiles(cb_id, ...);
    cb_pop_front(cb_id, n_tiles);         // Free space
}
```

### SRAM Sharing Between RISC-V Cores

All 5 RISC-V cores within a Tensix core share the same L1 SRAM:

1. **Private local memory**: Each core has its own local scratch space
2. **Shared circular buffers**: Synchronized access via CB APIs
3. **Automatic synchronization**: CB operations handle inter-core sync
4. **Backpressure**: Reserve/wait operations block automatically

## 3. Matrix and Vector Engines

### Compute Pipeline Architecture

The Tensix compute pipeline consists of three main components:

#### Unpacker Engine

Controlled by TRISC, reads tiles from circular buffers:

```cpp
// Unpacker operations
unpack_reconfig_data_format(cb_in0, cb_in1);  // Configure formats
unpack_tiles(cb_id, tile_idx, dst_idx);       // Unpack to DST registers
tilize_block(cb_in, num_tiles, cb_out);       // RM → Tiled conversion
untilize_block(cb_in, num_tiles, cb_out);     // Tiled → RM conversion
```

#### Math/Compute Engine

Performs matrix and vector operations on 32×32 tiles:

```cpp
// Matrix multiplication
mm_init(cb_in0, cb_in1, cb_out);
matmul_tiles(cb_in0, cb_in1, in0_idx, in1_idx, dst_idx, transpose);

// Element-wise operations (vector-like)
binary_op_init_common(cb_in0, cb_in1, cb_out);
ADD_TILES(cb_in0, cb_in1, dst_tile);
MUL_TILES(cb_in0, cb_in1, dst_tile);
SUB_TILES(cb_in0, cb_in1, dst_tile);

// Special functions (SFPU - Special Function Processing Unit)
sfpu_exp_init();
sfpu_exp(dst_tile);
sfpu_log_init();
sfpu_log(dst_tile);
sfpu_sqrt_init();
sfpu_sqrt(dst_tile);
```

#### Packer Engine

Writes results from destination registers to output CBs:

```cpp
// Packer operations
pack_reconfig_data_format(old_cb, new_cb);    // Change output format
pack_tile(dst_idx, cb_out);                   // Pack single tile
pack_block(num_tiles, cb_out);                // Pack multiple tiles
```

### Destination Registers

The compute engine uses destination registers for accumulation:

```cpp
void compute_kernel() {
    // Acquire destination registers for computation
    acquire_dst();

    // Perform computations accumulating in DST
    for (uint32_t i = 0; i < num_iterations; i++) {
        matmul_tiles(cb_in0, cb_in1, i, i, 0, false);
    }

    // Pack results to output CB
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    // Release destination registers
    release_dst();
}
```

## 4. Page-Based Memory Access

### What is Page-Based Access?

From [`tt_metal/api/tt-metalium/buffer_types.hpp`](../../../../../../tt_metal/api/tt-metalium/buffer_types.hpp):

"Pages" are the fundamental unit of data transfer between DRAM and L1:

```cpp
// Page size depends on data format and tile dimensions
// For tiled layout: page_size = tile_size (32×32 elements)
// For row-major: page_size = row_size

// Example: BF16 tiled tensor
uint32_t element_size = 2;  // BF16 = 2 bytes
uint32_t tile_size = 32 * 32 * element_size;  // 2048 bytes per tile
uint32_t page_size = tile_size;  // Page = Tile for tiled layout
```

### Interleaved vs Consecutive Allocation

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED = 0,        // Pages distributed across DRAM banks
    HEIGHT_SHARDED = 2,     // Rows sharded across cores
    WIDTH_SHARDED = 3,      // Columns sharded across cores
    BLOCK_SHARDED = 4,      // 2D blocks sharded across cores
};
```

#### Interleaved Layout (Default)

Pages are distributed round-robin across DRAM banks:

```cpp
// Page distribution for interleaved layout
// Page 0 → Bank 0
// Page 1 → Bank 1
// Page 2 → Bank 2
// ...
// Page N → Bank (N % num_banks)

// Benefits:
// - Parallel access across banks
// - Better bandwidth utilization
// - Reduced bank conflicts

InterleavedBufferConfig config{
    .device = device,
    .size = tensor_size,
    .page_size = tile_size,
    .buffer_type = BufferType::DRAM,
    .buffer_layout = TensorMemoryLayout::INTERLEAVED
};
auto buffer = CreateBuffer(config);
```

#### Consecutive Layout

Pages stored sequentially in one location:

```cpp
// All pages in single bank/core
// Page 0, 1, 2, ... N stored consecutively

// Use cases:
// - Small buffers
// - Device-local scratch memory
// - Sequential access patterns
```

## 5. Network-on-Chip (NoC) Interfaces

### Dual NoC Architecture

From [`tt_metal/hw/inc/wormhole/noc/noc_parameters.h`](../../../../../../tt_metal/hw/inc/wormhole/noc/noc_parameters.h):

```cpp
#define NUM_NOCS 2                    // Two independent NoCs per core

// NoC bandwidth varies by architecture:
#define NOC_PAYLOAD_WIDTH_WH 256      // Wormhole: 256 bits = 32 bytes/cycle
#define NOC_PAYLOAD_WIDTH_BH 512      // Blackhole: 512 bits = 64 bytes/cycle

#define NOC_MAX_BURST_SIZE 8192        // Maximum transaction size: 8 KB
#define NOC_NUM_VIRTUAL_CHANNELS 16   // 16 VCs for QoS and deadlock avoidance
```

### NoC Addressing

```cpp
// Calculate NOC address for core (x,y) at L1 offset
uint64_t noc_addr = NOC_XY_ADDR(x, y, l1_offset);

// Multi-cast address for region
uint64_t mcast_addr = NOC_MULTICAST_ADDR(
    start_x, start_y,    // Top-left corner
    end_x, end_y,        // Bottom-right corner
    l1_offset
);
```

### NoC Data Transfer APIs

Used in kernel code for data movement:

```cpp
// Asynchronous read from remote core/DRAM
void reader_kernel() {
    uint64_t src_noc_addr = get_noc_addr(src_x, src_y, src_offset);
    uint32_t dst_l1_addr = get_write_ptr(cb_id);

    // Issue async read
    noc_async_read(src_noc_addr, dst_l1_addr, size);

    // Can issue multiple reads before barrier
    noc_async_read(src_noc_addr2, dst_l1_addr2, size2);

    // Wait for all reads to complete
    noc_async_read_barrier();
}

// Asynchronous write to remote core/DRAM
void writer_kernel() {
    uint32_t src_l1_addr = get_read_ptr(cb_id);
    uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, dst_offset);

    // Issue async write
    noc_async_write(src_l1_addr, dst_noc_addr, size);

    // Wait for write completion
    noc_async_write_barrier();
}

// Multicast write (broadcast to multiple cores)
void multicast_kernel() {
    uint64_t mcast_addr = NOC_MULTICAST_ADDR(
        0, 0, 3, 3,  // Broadcast to 4×4 grid
        l1_offset
    );
    noc_async_write(src_l1_addr, mcast_addr, size);
    noc_async_write_barrier();
}
```

### NoC Alignment Requirements

```cpp
// From noc_parameters.h
#define NOC_L1_READ_ALIGNMENT_BYTES    16   // L1 reads must be 16-byte aligned
#define NOC_L1_WRITE_ALIGNMENT_BYTES   16   // L1 writes must be 16-byte aligned
#define NOC_DRAM_READ_ALIGNMENT_BYTES  32   // DRAM reads must be 32-byte aligned
#define NOC_DRAM_WRITE_ALIGNMENT_BYTES 16   // DRAM writes must be 16-byte aligned
```

### NoC Usage Patterns

#### Pattern 1: Double Buffering

```cpp
void double_buffer_reader() {
    uint32_t buffer_index = 0;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve next buffer while compute processes current
        cb_reserve_back(cb_id, block_tiles);
        uint32_t l1_addr = get_write_ptr(cb_id);

        // Fetch next block
        noc_async_read(get_block_addr(block), l1_addr, block_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, block_tiles);
        buffer_index ^= 1;  // Toggle between buffers
    }
}
```

#### Pattern 2: Batched NOC Operations

```cpp
void batched_noc_reader() {
    // Issue multiple NOC operations before barrier
    for (uint32_t i = 0; i < batch_size; i++) {
        noc_async_read(src_addrs[i], dst_addrs[i], sizes[i]);
    }
    // Single barrier for all operations (more efficient)
    noc_async_read_barrier();
}
```

## 6. Practical Examples

### Example 1: Complete Data Flow Through Tensix Core

```cpp
// Reader kernel (NCRISC) - Fetch data from DRAM
void reader_kernel() {
    constexpr uint32_t cb_in = 0;
    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        uint64_t dram_noc_addr = get_noc_addr(dram_addr + i * tile_size);
        noc_async_read(dram_noc_addr, l1_write_addr, tile_size);
        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
    }
}

// Compute kernel (TRISC) - Process tiles
void compute_kernel() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Initialize compute operation
    unary_op_init_common(cb_in, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);

        acquire_dst();

        // Process tile (e.g., apply exponential)
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, 0);
        sfpu_exp_init();
        sfpu_exp(0);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();

        cb_pop_front(cb_in, 1);
    }
}

// Writer kernel (NCRISC) - Write results to DRAM
void writer_kernel() {
    constexpr uint32_t cb_out = 16;
    uint32_t dram_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        uint64_t dram_noc_addr = get_noc_addr(dram_addr + i * tile_size);
        noc_async_write(l1_read_addr, dram_noc_addr, tile_size);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
```

### Example 2: Matrix Multiplication on Tensix

```cpp
void matmul_compute_kernel() {
    constexpr uint32_t cb_in0 = 0;   // A matrix tiles
    constexpr uint32_t cb_in1 = 1;   // B matrix tiles
    constexpr uint32_t cb_out = 16;  // C matrix tiles

    uint32_t M = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t K = get_arg_val<uint32_t>(2);

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            acquire_dst();

            // Accumulate K dimension
            for (uint32_t k = 0; k < K; k++) {
                cb_wait_front(cb_in0, 1);  // A[m,k]
                cb_wait_front(cb_in1, 1);  // B[k,n]

                matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // Write output tile C[m,n]
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            release_dst();
        }
    }
}
```

### Example 3: Using Both NoCs for Parallel Transfers

```cpp
void dual_noc_reader() {
    // Configure NOC indices
    constexpr uint32_t noc0_index = 0;
    constexpr uint32_t noc1_index = 1;

    // Can use both NoCs simultaneously for different transfers
    uint64_t src0_addr = get_noc_addr_on_noc(src0_x, src0_y, offset0, noc0_index);
    uint64_t src1_addr = get_noc_addr_on_noc(src1_x, src1_y, offset1, noc1_index);

    // Issue reads on different NoCs
    noc_async_read_on_noc(src0_addr, dst0_l1, size0, noc0_index);
    noc_async_read_on_noc(src1_addr, dst1_l1, size1, noc1_index);

    // Wait for both
    noc_async_read_barrier_on_noc(noc0_index);
    noc_async_read_barrier_on_noc(noc1_index);
}
```

## Architecture Summary

The Tensix core architecture enables high-performance AI computation through:

1. **Specialized RISC-V cores**: Each handling specific aspects of the compute pipeline
2. **1.5 MB shared L1 SRAM**: Fast local storage with circular buffer organization
3. **Producer-consumer synchronization**: Lock-free data flow between pipeline stages
4. **Dual NoC interfaces**: 2× 64 B/cycle bandwidth for parallel data movement
5. **Flexible compute engines**: Matrix operations, vector operations, and special functions
6. **Page-based memory access**: Efficient DRAM bandwidth utilization through interleaving

This design allows overlapping of data movement and computation, achieving high utilization of both memory bandwidth and compute resources.

## Key References

- Memory maps: [`dev_mem_map.h`](../../../../../../tt_metal/hw/inc/wormhole/dev_mem_map.h)
- Circular buffers: [`circular_buffer_config.hpp`](../../../../../../tt_metal/api/tt-metalium/circular_buffer_config.hpp)
- Buffer types: [`buffer_types.hpp`](../../../../../../tt_metal/api/tt-metalium/buffer_types.hpp)
- NoC parameters: [`noc_parameters.h`](../../../../../../tt_metal/hw/inc/wormhole/noc/noc_parameters.h)
- Dataflow APIs: [`dataflow_api.h`](../../../../../../tt_metal/hw/inc/dataflow_api.h)
- Compute APIs: [`compute_kernel_api.h`](../../../../../../tt_metal/hw/inc/compute_kernel_api.h)