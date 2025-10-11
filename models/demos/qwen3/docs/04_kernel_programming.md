# TT-Metal Kernel Programming

## Overview

TT-Metal kernels are low-level programs that run directly on Tenstorrent AI accelerator hardware. The architecture uses three types of specialized kernels running on different processors within each Tensix core, coordinated through circular buffers for high-performance pipelined execution.

## 1. Kernel Types and Structure

### Three Main Kernel Types

#### Data Movement Kernels (BRISC/NCRISC)
- **Processors**: Run on RISC-V cores (BRISC or NCRISC)
- **Purpose**: Handle data transfers between DRAM and L1 memory
- **Entry Point**: `void kernel_main()`
- **Subtypes**:
  - **Reader kernels**: Transfer data from DRAM → L1 circular buffers
  - **Writer kernels**: Transfer data from L1 circular buffers → DRAM

#### Compute Kernels (TRISC)
- **Processors**: Run on Tensix compute cores (3 TRISC threads)
- **Purpose**: Perform mathematical operations
- **Entry Point**: `void MAIN` (expands to thread-specific main)
- **Sub-threads**:
  - **TRISC0 (Unpack)**: Unpacks tiles from CBs to source registers
  - **TRISC1 (Math)**: Performs computation (matmul, SFPU operations)
  - **TRISC2 (Pack)**: Packs results from destination registers to output CBs

#### Ethernet Kernels (ERISC)
- **Processors**: Run on Ethernet cores
- **Purpose**: Handle multi-device communication over fabric
- **Entry Point**: `void kernel_main()`

### Basic Kernel Structure

#### Data Movement Kernel Example

```cpp
#include "dataflow_api.h"

void kernel_main() {
    // Get runtime arguments
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // Get compile-time arguments
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    // Main processing loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Reserve space in circular buffer
        cb_reserve_back(cb_id_out, 1);

        // Get L1 write pointer
        uint32_t l1_write_addr = get_write_ptr(cb_id_out);

        // Read tile from DRAM to L1
        uint64_t src_noc_addr = get_noc_addr(src_addr + i * tile_size);
        noc_async_read(src_noc_addr, l1_write_addr, tile_size);
        noc_async_read_barrier();

        // Push tile to circular buffer
        cb_push_back(cb_id_out, 1);
    }
}
```

#### Compute Kernel Example

```cpp
#include <cstdint>
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    // Get compile-time arguments
    uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize operation
    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for input tile
        cb_wait_front(cb_in0, 1);

        // Acquire destination registers
        acquire_dst();

        // Copy tile to DST and apply operation
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);

        // Apply exponential function
        exp_tile_init();
        exp_tile(0);

        // Pack result to output CB
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        // Release destination registers
        release_dst();

        // Pop input tile
        cb_pop_front(cb_in0, 1);
    }
}
}
```

## 2. Accessing Arguments

### Compile-Time Arguments

Constants embedded in the kernel binary at compilation:

```cpp
// Access compile-time argument by index
uint32_t block_size = get_compile_time_arg_val(0);
uint32_t num_blocks = get_compile_time_arg_val(1);
constexpr bool is_dram = get_compile_time_arg_val(2) == 1;

// Use for:
// - Buffer types (DRAM/L1)
// - Data formats
// - Block sizes and dimensions
// - Loop bounds that don't change between invocations
```

### Runtime Arguments

Dynamic values set per invocation:

```cpp
// Unique (per-core) runtime arguments
uint32_t buffer_addr = get_arg_val<uint32_t>(0);
uint32_t start_tile = get_arg_val<uint32_t>(1);
uint32_t stride_h = get_arg_val<uint32_t>(2);
uint32_t stride_w = get_arg_val<uint32_t>(3);

// Common (shared across all cores) runtime arguments
uint32_t batch_size = get_common_arg_val<uint32_t>(0);
uint32_t global_offset = get_common_arg_val<uint32_t>(1);

// Maximum: 341 args per kernel (4 bytes each)
```

### Core Coordinates

Get the current core's position:

```cpp
// Absolute logical coordinates
uint8_t my_x = get_absolute_logical_x();
uint8_t my_y = get_absolute_logical_y();

// Relative to sub-device origin
uint8_t rel_x = get_relative_logical_x();
uint8_t rel_y = get_relative_logical_y();
```

## 3. Circular Buffer APIs

### Overview

- **32 CBs per core** (indices 0-31)
- **Standard allocation**:
  - CB 0-7: Input buffers
  - CB 16-23: Output buffers
  - CB 24-31: Intermediate/scratch buffers

### Producer Operations (Writing to CB)

From [`tt_metal/hw/inc/dataflow_api.h`](../../../../../../tt_metal/hw/inc/dataflow_api.h):

```cpp
// Reserve space in CB (blocks if full)
void cb_reserve_back(uint32_t cb_id, uint32_t num_pages);

// Check if space available (non-blocking)
bool cb_pages_reservable_at_back(uint32_t cb_id, uint32_t num_pages);

// Get write pointer to reserved space
uint32_t get_write_ptr(uint32_t cb_id);

// Mark pages as ready for consumer
void cb_push_back(uint32_t cb_id, uint32_t num_pages);
```

**Producer Pattern:**
```cpp
// Reserve space
cb_reserve_back(cb_id, num_tiles);

// Get L1 address for writing
uint32_t l1_write_addr = get_write_ptr(cb_id);

// Write data to L1 address
noc_async_read(src_addr, l1_write_addr, size);
noc_async_read_barrier();

// Mark data as ready
cb_push_back(cb_id, num_tiles);
```

### Consumer Operations (Reading from CB)

```cpp
// Wait for pages (blocks if empty)
void cb_wait_front(uint32_t cb_id, uint32_t num_pages);

// Check if pages available (non-blocking)
bool cb_pages_available_at_front(uint32_t cb_id, uint32_t num_pages);

// Get read pointer to available pages
uint32_t get_read_ptr(uint32_t cb_id);

// Free pages after consumption
void cb_pop_front(uint32_t cb_id, uint32_t num_pages);
```

**Consumer Pattern:**
```cpp
// Wait for data
cb_wait_front(cb_id, num_tiles);

// Get L1 address for reading
uint32_t l1_read_addr = get_read_ptr(cb_id);

// Read/process data from L1 address
noc_async_write(l1_read_addr, dst_addr, size);
noc_async_write_barrier();

// Free pages
cb_pop_front(cb_id, num_tiles);
```

## 4. NOC (Network-on-Chip) Programming

### Asynchronous Read Operations

```cpp
// Generic read (handles any size)
void noc_async_read(
    uint64_t src_noc_addr,      // Source NOC address
    uint32_t dst_local_l1_addr, // Destination L1 address
    uint32_t size,               // Size in bytes
    uint8_t noc = noc_index      // NOC index (0 or 1)
);

// Read barrier - MUST call after async reads
void noc_async_read_barrier(uint8_t noc = noc_index);
```

**Example:**
```cpp
// Calculate NOC address
uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(
    dram_bank_id, dram_buffer_addr);

// Issue read
noc_async_read(dram_noc_addr, l1_buffer_addr, buffer_size);

// Wait for completion
noc_async_read_barrier();
```

### Asynchronous Write Operations

```cpp
// Generic write
void noc_async_write(
    uint32_t src_local_l1_addr,  // Source L1 address
    uint64_t dst_noc_addr,        // Destination NOC address
    uint32_t size,                // Size in bytes
    uint8_t noc = noc_index       // NOC index
);

// Write barrier - MUST call after async writes
void noc_async_write_barrier(uint8_t noc = noc_index);
```

### Multicast Operations

Broadcast data to multiple cores:

```cpp
// Multicast write
void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,           // Number of destination cores
    bool linked = false,           // Link multicast packets
    uint8_t noc = noc_index
);
```

**Example:**
```cpp
// Create multicast address for 4×4 grid
uint64_t mcast_addr = NOC_MULTICAST_ADDR(
    0, 0,    // Top-left corner (start_x, start_y)
    3, 3,    // Bottom-right corner (end_x, end_y)
    l1_offset
);

// Broadcast data
noc_async_write_multicast(src_l1_addr, mcast_addr, size, 16);
noc_async_write_barrier();
```

### NOC Address Calculation

```cpp
// Get NOC address from DRAM bank
template <bool DRAM>
uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t addr);

// Get NOC address from core coordinates
#define NOC_XY_ADDR(noc_x, noc_y, addr) \
    (((uint64_t)(noc_x) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(noc_y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | \
     (addr))
```

### NOC Best Practices

1. **Always use barriers**: Call `noc_async_read_barrier()` or `noc_async_write_barrier()` after async operations
2. **Batch operations**: Issue multiple reads/writes before a single barrier
3. **Alignment requirements**:
   - L1 reads/writes: 16-byte aligned
   - DRAM reads: 32-byte aligned
   - DRAM writes: 16-byte aligned

**Batched Operations Example:**
```cpp
// Issue multiple reads
for (uint32_t i = 0; i < num_tiles; i++) {
    noc_async_read(src_addrs[i], dst_addrs[i], tile_size);
}
// Single barrier for all reads
noc_async_read_barrier();
```

## 5. Compute Operations

### Destination Register Management

The compute engine uses destination (DST) registers for accumulation:

```cpp
// Legacy API (still widely used)
acquire_dst();     // Acquire exclusive lock on DST registers
release_dst();     // Release DST registers

// New API (recommended for complex pipelines)
tile_regs_acquire();  // MATH thread: acquire DST
tile_regs_commit();   // MATH thread: signal completion
tile_regs_wait();     // PACK thread: wait for MATH
tile_regs_release();  // PACK thread: release DST
```

### Matrix Multiplication

From [`tt_metal/include/compute_kernel_api/matmul.h`](../../../../../../tt_metal/include/compute_kernel_api/matmul.h):

```cpp
// Initialize matmul operation
void mm_init(
    uint32_t in0_cb_id,  // Input A circular buffer
    uint32_t in1_cb_id,  // Input B circular buffer
    uint32_t out_cb_id   // Output circular buffer
);

// Perform tile multiplication
// C[dst_tile_index] += A[in0_tile_index] * B[in1_tile_index]
void matmul_tiles(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dst_index,
    bool transpose_B
);
```

**Simple Matmul Example:**
```cpp
void MAIN {
    uint32_t M = get_compile_time_arg_val(0);
    uint32_t K = get_compile_time_arg_val(1);
    uint32_t N = get_compile_time_arg_val(2);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            acquire_dst();

            // Accumulate over K dimension
            for (uint32_t k = 0; k < K; k++) {
                cb_wait_front(tt::CBIndex::c_0, 1);  // A[m,k]
                cb_wait_front(tt::CBIndex::c_1, 1);  // B[k,n]

                matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1,
                           0, 0, 0, false);

                cb_pop_front(tt::CBIndex::c_0, 1);
                cb_pop_front(tt::CBIndex::c_1, 1);
            }

            // Pack result
            cb_reserve_back(tt::CBIndex::c_16, 1);
            pack_tile(0, tt::CBIndex::c_16);
            cb_push_back(tt::CBIndex::c_16, 1);

            release_dst();
        }
    }
}
```

### Element-wise Operations

```cpp
// Binary operations
binary_op_init_common(cb_in0, cb_in1, cb_out);
ADD_TILES(cb_in0, cb_in1, dst_tile);
MUL_TILES(cb_in0, cb_in1, dst_tile);
SUB_TILES(cb_in0, cb_in1, dst_tile);

// Unary operations
unary_op_init_common(cb_in, cb_out);
copy_tile(cb_in, in_tile_index, dst_tile_index);
```

### SFPU (Special Function Processing Unit) Operations

```cpp
// Activation functions
exp_tile_init();      exp_tile(dst_index);      // e^x
log_tile_init();      log_tile(dst_index);      // ln(x)
sqrt_tile_init();     sqrt_tile(dst_index);     // √x
sigmoid_tile_init();  sigmoid_tile(dst_index);  // 1/(1+e^-x)
tanh_tile_init();     tanh_tile(dst_index);     // tanh(x)
gelu_tile_init();     gelu_tile(dst_index);     // GELU activation

// Trigonometric
sin_tile_init();      sin_tile(dst_index);
cos_tile_init();      cos_tile(dst_index);

// Each operation requires initialization before use
```

### Packing Operations

Transfer results from DST registers to output CBs:

```cpp
// Pack single tile from DST to CB
void pack_tile(uint32_t dst_index, uint32_t cb_id);

// Pack block of tiles
void pack_block(uint32_t num_tiles, uint32_t cb_id);

// Example
cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);  // Pack DST[0] to output CB
cb_push_back(cb_out, 1);
```

## 6. Kernel Synchronization

### Reader-Compute-Writer Pipeline

The three kernel types synchronize through circular buffers:

```
Reader (NCRISC)          Compute (TRISC)           Writer (NCRISC)
───────────────          ───────────────           ────────────────
cb_reserve_back(c_0)
noc_async_read → c_0
noc_async_read_barrier
cb_push_back(c_0)    →   cb_wait_front(c_0)
                         acquire_dst()
                         compute_operation()
                         cb_reserve_back(c_16)
                         pack_tile → c_16
                         cb_push_back(c_16)    →   cb_wait_front(c_16)
                         release_dst()              noc_async_write
                         cb_pop_front(c_0)          noc_async_write_barrier
                                                   cb_pop_front(c_16)
```

**Key Points:**
- All kernels run **concurrently** on different processors
- Circular buffers provide automatic blocking and backpressure
- Double buffering enables overlap of data movement and computation

### TRISC Thread Synchronization

Within compute kernels, three threads coordinate:

```cpp
// UNPACK thread
unpack_tiles_from_cb_to_src();

// MATH thread
tile_regs_acquire();        // Wait for DST available
perform_computation();       // Compute
tile_regs_commit();         // Signal PACK thread

// PACK thread
tile_regs_wait();           // Wait for MATH completion
cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);
cb_push_back(cb_out, 1);
tile_regs_release();        // Release for MATH thread
```

## 7. Advanced Patterns

### Block-Based Matmul with Spilling

For large matrices, use blocking and intermediate buffering:

```cpp
void MAIN {
    // Compile-time block dimensions
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(0);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);

    bool enable_reload = false;

    for (uint32_t block = 0; block < num_blocks; block++) {
        bool last_block = (block == num_blocks - 1);

        // Reload partial results if not first block
        if (enable_reload) {
            cb_wait_front(tt::CBIndex::c_24, out_subblock_tiles);
            for (uint32_t i = 0; i < out_subblock_tiles; i++) {
                copy_tile(tt::CBIndex::c_24, i, i);  // Load to DST
            }
            cb_pop_front(tt::CBIndex::c_24, out_subblock_tiles);
            mm_init_short(tt::CBIndex::c_0, tt::CBIndex::c_1);
        }

        // Compute block
        acquire_dst();
        for (uint32_t k = 0; k < block_k; k++) {
            cb_wait_front(tt::CBIndex::c_0, 1);
            cb_wait_front(tt::CBIndex::c_1, 1);

            matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1,
                        0, 0, 0, false);

            cb_pop_front(tt::CBIndex::c_0, 1);
            cb_pop_front(tt::CBIndex::c_1, 1);
        }

        // Pack to output or intermediate buffer
        uint32_t out_cb = last_block ? tt::CBIndex::c_16 : tt::CBIndex::c_24;
        cb_reserve_back(out_cb, out_subblock_tiles);
        for (uint32_t i = 0; i < out_subblock_tiles; i++) {
            pack_tile(i, out_cb);
        }
        cb_push_back(out_cb, out_subblock_tiles);

        release_dst();

        if (!last_block) {
            enable_reload = true;
        }
    }
}
```

### Double Buffering Pattern

```cpp
void kernel_main() {
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t tiles_per_buffer = 32;
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // CB sized for double buffering (2× block size)
    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve next buffer while compute processes current
        cb_reserve_back(cb_id, tiles_per_buffer);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        // Fetch next block
        uint64_t src_addr = get_block_addr(block);
        noc_async_read(src_addr, l1_write_addr, block_size);
        noc_async_read_barrier();

        cb_push_back(cb_id, tiles_per_buffer);
        // Compute kernel processes previous block while we fetch next
    }
}
```

### Batched NOC Barriers

```cpp
void kernel_main() {
    constexpr uint32_t batch_size = 8;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);

        noc_async_write(l1_addr, dst_addr + i * tile_size, tile_size);

        // Barrier every batch_size writes or at the end
        if (((i + 1) % batch_size == 0) || (i == num_tiles - 1)) {
            noc_async_write_barrier();
        }

        cb_push_back(cb_id, 1);
    }
}
```

## 8. Complete Example: Element-wise Operation

Here's a complete three-kernel example for element-wise addition:

### Reader Kernel
```cpp
// reader_add.cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t tile_size = get_tile_size(cb_id_in0);
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Read first operand
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_addr0 = get_write_ptr(cb_id_in0);
        uint64_t src0_noc = get_noc_addr_from_bank_id<src0_is_dram>(
            0, src0_addr + i * tile_size);
        noc_async_read(src0_noc, l1_addr0, tile_size);

        // Read second operand
        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_addr1 = get_write_ptr(cb_id_in1);
        uint64_t src1_noc = get_noc_addr_from_bank_id<src1_is_dram>(
            0, src1_addr + i * tile_size);
        noc_async_read(src1_noc, l1_addr1, tile_size);

        // Wait for both reads
        noc_async_read_barrier();

        // Push both tiles
        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_in1, 1);
    }
}
```

### Compute Kernel
```cpp
// compute_add.cpp
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 16;

    binary_op_init_common(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        acquire_dst();

        // Add tiles
        ADD_TILES(cb_in0, cb_in1, 0);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
}
```

### Writer Kernel
```cpp
// writer_add.cpp
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t tile_size = get_tile_size(cb_id_out);
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_out);

        uint64_t dst_noc = get_noc_addr_from_bank_id<dst_is_dram>(
            0, dst_addr + i * tile_size);
        noc_async_write(l1_addr, dst_noc, tile_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, 1);
    }
}
```

## 9. Best Practices

### Memory Management
1. **Never use stack arrays for NOC operations** - Always use L1 circular buffers
2. **Clear kernel cache after changes**: `rm -rf ~/.cache/tt-metal-cache/`
3. **Use compile-time args for constants** - Enables compiler optimizations

### NOC Operations
1. **Always call barriers** after async operations
2. **Batch operations** - Multiple reads/writes before single barrier
3. **Respect alignment** - 16-byte for L1, 32-byte for DRAM reads

### Circular Buffers
1. **Reserve before get_ptr** - Always `cb_reserve_back()` before `get_write_ptr()`
2. **Match wait/pop** - Every `cb_wait_front()` needs corresponding `cb_pop_front()`
3. **Double buffer inputs** - Size = 2× block size for overlap

### Compute Operations
1. **Pair acquire/release** - Always match `acquire_dst()` with `release_dst()`
2. **Initialize operations** - Call init functions before operations
3. **Use blocking** for large matrices - Improves data reuse

### Performance
1. **Hide latency** with double buffering
2. **Minimize barriers** - Batch NOC operations
3. **Exploit data reuse** with blocking
4. **Use both NOCs** for parallel transfers when possible

## Key References

- Dataflow API: [`dataflow_api.h`](../../../../../../tt_metal/hw/inc/dataflow_api.h)
- Compute API: [`compute_kernel_api.h`](../../../../../../tt_metal/include/compute_kernel_api.h)
- Matmul API: [`matmul.h`](../../../../../../tt_metal/include/compute_kernel_api/matmul.h)
- Example kernels: [`test_kernels/`](../../../../../../tests/tt_metal/tt_metal/test_kernels/)
- Production kernels: [`ttnn/cpp/ttnn/operations/`](../../../../../../ttnn/cpp/ttnn/operations/)