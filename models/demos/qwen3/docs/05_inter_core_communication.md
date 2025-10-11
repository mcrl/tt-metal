# Inter-Core Communication and Synchronization

## Overview

TT-Metal provides comprehensive mechanisms for cores to communicate and synchronize within and across Tensix cores. This includes semaphores for synchronization, multicast for efficient one-to-many communication, remote circular buffers for producer-consumer patterns, and barriers for transaction completion.

## 1. Semaphores

### Overview

Semaphores are 4-byte L1 memory locations used for synchronization between cores. They support both local operations and remote operations through NOC.

### Local Semaphore Operations

From [`tt_metal/hw/inc/dataflow_api.h`](../../../../../../tt_metal/hw/inc/dataflow_api.h):

```cpp
// Set local semaphore value
void noc_semaphore_set(
    volatile tt_l1_ptr uint32_t* sem_addr,
    uint32_t val
);

// Wait for semaphore to equal specific value
void noc_semaphore_wait(
    volatile tt_l1_ptr uint32_t* sem_addr,
    uint32_t val
);

// Wait for semaphore to be >= minimum value
void noc_semaphore_wait_min(
    volatile tt_l1_ptr uint32_t* sem_addr,
    uint32_t val
);
```

**Usage Pattern:**
```cpp
// Initialize semaphore
volatile tt_l1_ptr uint32_t* sem_addr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
noc_semaphore_set(sem_addr, 0);

// Wait for expected value
noc_semaphore_wait(sem_addr, EXPECTED_VALUE);
```

### Remote Semaphore Operations

```cpp
// Write semaphore value to remote core
void noc_semaphore_set_remote(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr,
    uint8_t noc = noc_index
);

// Atomic increment of remote semaphore
template <bool posted = false>
void noc_semaphore_inc(
    uint64_t addr,
    uint32_t incr,
    uint8_t noc_id = noc_index
);

// Multicast semaphore to multiple cores
void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index
);
```

### Producer-Consumer Pattern with Semaphores

Here's a complete example of multicast sender-receiver synchronization:

#### Sender Core

```cpp
// Initialize semaphores
volatile tt_l1_ptr uint32_t* receiver_semaphore =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
*receiver_semaphore = VALID;

volatile tt_l1_ptr uint32_t* sender_semaphore =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);

for (uint32_t block = 0; block < num_blocks; block++) {
    // 1. Load data from DRAM
    cb_reserve_back(cb_id_in, block_tiles);
    uint32_t l1_addr = get_write_ptr(cb_id_in);
    noc_async_read(dram_addr, l1_addr, block_size);
    noc_async_read_barrier();

    // 2. Wait for all receivers to be ready
    noc_semaphore_wait(sender_semaphore, num_receivers);
    noc_semaphore_set(sender_semaphore, 0);  // Reset for next block

    // 3. Multicast data to receivers
    uint64_t mcast_data_addr = get_noc_multicast_addr(
        start_x, start_y, end_x, end_y, data_l1_addr);
    noc_async_write_multicast(data_l1_addr, mcast_data_addr,
                             block_size, num_receivers);
    noc_async_write_barrier();

    // 4. Signal receivers that data is ready
    uint64_t mcast_sem_addr = get_noc_multicast_addr(
        start_x, start_y, end_x, end_y, receiver_sem_addr);
    noc_semaphore_set_multicast(receiver_sem_addr, mcast_sem_addr, num_receivers);

    cb_push_back(cb_id_in, block_tiles);
}
```

#### Receiver Cores

```cpp
volatile tt_l1_ptr uint32_t* receiver_semaphore =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

for (uint32_t block = 0; block < num_blocks; block++) {
    cb_reserve_back(cb_id_in, block_tiles);

    // 1. Mark local semaphore as not ready
    noc_semaphore_set(receiver_semaphore, INVALID);

    // 2. Signal sender we're ready (atomic increment)
    uint64_t sender_sem_noc_addr = get_noc_addr(
        sender_x, sender_y, sender_sem_addr);
    noc_semaphore_inc(sender_sem_noc_addr, 1);

    // 3. Wait for sender to multicast data
    noc_semaphore_wait(receiver_semaphore, VALID);

    // 4. Process received data
    cb_push_back(cb_id_in, block_tiles);
}
```

## 2. Barriers

### NOC Transaction Barriers

Different barrier types ensure completion of various NOC operations:

```cpp
// Wait for all async reads to complete
void noc_async_read_barrier(uint8_t noc = noc_index);

// Wait for all async writes to complete (with acknowledgments)
void noc_async_write_barrier(uint8_t noc = noc_index);

// Wait for all writes to be sent (without waiting for acks)
void noc_async_writes_flushed(uint8_t noc = noc_index);

// Wait for atomic operations to complete
void noc_async_atomic_barrier(uint8_t noc_idx = noc_index);

// Wait for ALL transaction types
void noc_async_full_barrier(uint8_t noc_idx = noc_index);
```

### Barrier Implementation Details

From [`tt_metal/hw/inc/wormhole/noc_nonblocking_api.h`](../../../../../../tt_metal/hw/inc/wormhole/noc_nonblocking_api.h):

Each processor maintains counters for different transaction types:

```cpp
enum class NocBarrierType : uint8_t {
    READS_NUM_ISSUED,
    NONPOSTED_WRITES_NUM_ISSUED,
    NONPOSTED_WRITES_ACKED,
    NONPOSTED_ATOMICS_ACKED,
    POSTED_WRITES_NUM_ISSUED
};

// Barrier implementation checks counters against status registers
bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) ==
            noc_reads_num_issued[noc]);
}
```

### Remote Circular Buffer Barriers

Ensure all receivers have acknowledged data:

```cpp
void remote_cb_sender_barrier(uint32_t cb_id) {
    RemoteSenderCBInterface& remote_cb = get_remote_sender_cb_interface(cb_id);

    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.pages_sent_ptr);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_cb.pages_acked_ptr);

    // Wait for each receiver to ack all sent data
    for (uint32_t i = 0; i < remote_cb.num_receivers; ++i) {
        while (*pages_acked_ptr != *pages_sent_ptr) {
            // Spin wait
        }
        pages_acked_ptr += alignment_offset;
        pages_sent_ptr += alignment_offset;
    }
}
```

## 3. Multicast and Broadcast

### Multicast Addressing

Create NOC addresses for rectangular grids of cores:

```cpp
// Create multicast address for rectangular region
uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start,  // Top-left X
    uint32_t noc_y_start,  // Top-left Y
    uint32_t noc_x_end,    // Bottom-right X (inclusive)
    uint32_t noc_y_end,    // Bottom-right Y (inclusive)
    uint32_t addr,         // L1 address on all destinations
    uint8_t noc = noc_index
);
```

### Multicast Write Operations

```cpp
// Standard multicast (sender NOT included)
void noc_async_write_multicast(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index
);

// Multicast with loopback (sender INCLUDED)
void noc_async_write_multicast_loopback_src(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,      // Includes sender
    bool linked = false,
    uint8_t noc = noc_index
);
```

### Linked Multicast Transactions

Optimize series of multicasts to same destinations:

```cpp
// Setup multicast address
uint64_t mcast_addr = get_noc_multicast_addr(
    start_x, start_y, end_x, end_y, l1_addr);

// Linked transactions maintain path reservation
for (uint32_t i = 0; i < num_transactions - 1; i++) {
    noc_async_write_multicast(
        src_addr, mcast_addr, size, num_dests,
        true);  // linked=true keeps path
}

// Last transaction releases path
noc_async_write_multicast(
    src_addr, mcast_addr, size, num_dests,
    false);  // linked=false releases path

noc_async_write_barrier();
```

## 4. Core-to-Core Communication Patterns

### Direct Unicast

Point-to-point communication between cores:

```cpp
// Get remote core address
uint64_t remote_addr = get_noc_addr(remote_x, remote_y, l1_addr);

// Read from remote core
noc_async_read(remote_addr, local_l1_addr, size);
noc_async_read_barrier();

// Write to remote core
noc_async_write(local_l1_addr, remote_addr, size);
noc_async_write_barrier();
```

### Scatter Pattern (One-to-Many)

Distribute data from one core to multiple cores:

```cpp
// Prepare data in local L1
cb_reserve_back(cb_id, num_tiles);
uint32_t local_addr = get_write_ptr(cb_id);
// ... populate data ...

// Scatter to rectangular grid
uint64_t mcast_addr = get_noc_multicast_addr(
    x_start, y_start, x_end, y_end, remote_l1_addr);
noc_async_write_multicast(local_addr, mcast_addr, size, num_dests);
noc_async_write_barrier();

cb_push_back(cb_id, num_tiles);
```

### Gather Pattern (Many-to-One)

Collect results from multiple cores:

```cpp
// Each worker core atomically increments accumulator
uint64_t accumulator_addr = get_noc_addr(
    collector_x, collector_y, accumulator_l1_addr);

// Atomic add to accumulator
noc_semaphore_inc<false>(accumulator_addr, partial_result);
noc_async_atomic_barrier();
```

### Pipeline Pattern

Chain cores for sequential processing:

```cpp
// STAGE 1: Reader Core
for (uint32_t block = 0; block < num_blocks; block++) {
    cb_reserve_back(cb_out, tiles);
    // Read from DRAM
    noc_async_read(dram_addr, l1_addr, size);
    noc_async_read_barrier();
    cb_push_back(cb_out, tiles);  // Signal Stage 2
}

// STAGE 2: Compute Core
for (uint32_t block = 0; block < num_blocks; block++) {
    cb_wait_front(cb_in, tiles);   // Wait for Stage 1
    cb_reserve_back(cb_out, tiles);

    acquire_dst();
    // Process data
    matmul_tiles(...);
    pack_tile(...);
    release_dst();

    cb_push_back(cb_out, tiles);   // Signal Stage 3
    cb_pop_front(cb_in, tiles);
}

// STAGE 3: Writer Core
for (uint32_t block = 0; block < num_blocks; block++) {
    cb_wait_front(cb_in, tiles);   // Wait for Stage 2
    // Write to DRAM
    noc_async_write(l1_addr, dram_addr, size);
    noc_async_write_barrier();
    cb_pop_front(cb_in, tiles);
}
```

## 5. Remote Circular Buffers

### Overview

Remote circular buffers enable producer-consumer patterns across cores with automatic flow control.

From [`tt_metal/hw/inc/remote_circular_buffer_api.h`](../../../../../../tt_metal/hw/inc/remote_circular_buffer_api.h):

### Sender Operations

```cpp
// Reserve space (blocks if receivers haven't consumed)
void remote_cb_reserve_back(uint32_t cb_id, uint32_t num_pages);

// Push data and write to receivers
template <bool skip_ptr_update = true>
void remote_cb_push_back_and_write_pages(
    uint32_t cb_id,
    uint32_t local_cb_addr,
    uint32_t num_pages,
    uint32_t num_rows,
    uint32_t coalesced_num_pages_per_row,
    uint32_t coalesced_page_size,
    uint8_t noc = noc_index
);

// Wait for all receivers to acknowledge
void remote_cb_sender_barrier(uint32_t cb_id);
```

### Receiver Operations

```cpp
// Wait for data from sender
void remote_cb_wait_front(uint32_t cb_id, uint32_t num_pages);

// Pop and acknowledge to sender
void remote_cb_pop_front(uint32_t cb_id, uint32_t num_pages, uint8_t noc = noc_index);
```

### Flow Control Mechanism

1. Sender maintains `pages_sent` counter per receiver
2. Receiver maintains `pages_acked` counter
3. `remote_cb_reserve_back()` blocks if buffer would overflow
4. `remote_cb_pop_front()` atomically increments sender's ack counter
5. Credit-based system prevents buffer overflow

## 6. Multi-Core Coordination

### Work Distribution

Distribute work across a grid of cores:

```cpp
// Host-side setup
uint32_t num_cores_x = 4, num_cores_y = 4;
uint32_t per_core_M = M / num_cores_y;
uint32_t per_core_N = N / num_cores_x;

for (uint32_t y = 0; y < num_cores_y; y++) {
    for (uint32_t x = 0; x < num_cores_x; x++) {
        CoreCoord core = {x, y};

        // Calculate per-core work assignment
        uint32_t start_row = y * per_core_M;
        uint32_t start_col = x * per_core_N;

        // Set unique runtime args
        SetRuntimeArgs(program, kernel, core, {
            start_row, start_col, per_core_M, per_core_N
        });
    }
}
```

### Barrier-Based Phases

Synchronize all cores between processing phases:

```cpp
// Phase 1: All cores read input
for (uint32_t i = 0; i < my_tiles; i++) {
    noc_async_read(src_addr[i], dst_addr[i], tile_size);
}
noc_async_read_barrier();

// Signal phase 1 complete
uint64_t phase_counter_addr = get_noc_addr(coord_x, coord_y, counter_l1);
noc_semaphore_inc(phase_counter_addr, 1);

// Wait for all cores to complete phase 1
volatile tt_l1_ptr uint32_t* phase_counter =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_l1);
noc_semaphore_wait(phase_counter, num_cores);

// Phase 2: All cores compute
// ... computation ...

// Phase 3: All cores write output
// ... similar barrier pattern ...
```

## 7. Avoiding Deadlocks and Race Conditions

### Best Practices

#### 1. Consistent Resource Ordering

```cpp
// ✅ CORRECT: All cores acquire in same order
noc_async_read(resource_A_addr, local_A, size_A);
noc_async_read_barrier();
noc_async_read(resource_B_addr, local_B, size_B);
noc_async_read_barrier();

// ❌ WRONG: Different ordering can deadlock
// Core 1: A then B
// Core 2: B then A
```

#### 2. Producer-Consumer Discipline

```cpp
// Producer ALWAYS:
cb_reserve_back(cb_id, tiles);  // Block until space
// ... write data ...
cb_push_back(cb_id, tiles);     // Make visible

// Consumer ALWAYS:
cb_wait_front(cb_id, tiles);    // Block until data
// ... read data ...
cb_pop_front(cb_id, tiles);     // Free space
```

#### 3. Asymmetric Communication

```cpp
// ✅ CORRECT: Initiator-responder pattern
// Core A (initiator):
noc_semaphore_set_remote(to_B_addr, SIGNAL);
noc_semaphore_wait(from_B_addr, RESPONSE);

// Core B (responder):
noc_semaphore_wait(from_A_addr, SIGNAL);
// ... process ...
noc_semaphore_set_remote(to_A_addr, RESPONSE);
```

#### 4. Atomic Operations for Shared Counters

```cpp
// ❌ WRONG: Race condition
uint32_t counter = *counter_addr;
counter++;
*counter_addr = counter;

// ✅ CORRECT: Atomic increment
noc_semaphore_inc<false>(counter_noc_addr, 1);
noc_async_atomic_barrier();
```

#### 5. Cache Invalidation for Polling

```cpp
// When polling remote-updated values
volatile tt_l1_ptr uint32_t* flag =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);

while (*flag != EXPECTED_VALUE) {
    // Compiler ensures volatile read
}
```

## 8. Complete Example: Multi-Core Matrix Multiply

Here's a complete example showing sender-receiver coordination for distributed matrix multiplication:

### Sender Core (broadcasts A matrix rows)

```cpp
void kernel_main() {
    // Runtime args
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);

    // Compile-time args
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_receivers = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = 0;

    // Semaphore setup
    volatile tt_l1_ptr uint32_t* ready_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(READY_SEM_ADDR);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Read A block from DRAM
        cb_reserve_back(cb_id, block_tiles);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read(a_addr + block * block_size, l1_addr, block_size);
        noc_async_read_barrier();

        // Wait for receivers
        noc_semaphore_wait(ready_sem, num_receivers);
        noc_semaphore_set(ready_sem, 0);

        // Multicast A block
        uint64_t mcast_addr = get_noc_multicast_addr(
            0, 0, 3, 0, DATA_ADDR);  // Row of 4 cores
        noc_async_write_multicast(l1_addr, mcast_addr, block_size, num_receivers);
        noc_async_write_barrier();

        // Signal completion
        uint64_t done_mcast = get_noc_multicast_addr(
            0, 0, 3, 0, DONE_SEM_ADDR);
        noc_semaphore_set_multicast(DONE_VALUE_ADDR, done_mcast, num_receivers);

        cb_push_back(cb_id, block_tiles);
    }
}
```

### Receiver Cores (receive A, compute with local B)

```cpp
void kernel_main() {
    // Runtime args
    uint32_t b_addr = get_arg_val<uint32_t>(0);
    uint32_t c_addr = get_arg_val<uint32_t>(1);
    uint32_t my_col = get_arg_val<uint32_t>(2);

    // Semaphores
    volatile tt_l1_ptr uint32_t* done_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DONE_SEM_ADDR);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Signal ready to sender
        noc_semaphore_set(done_sem, INVALID);
        uint64_t sender_ready = get_noc_addr(sender_x, sender_y, READY_SEM_ADDR);
        noc_semaphore_inc(sender_ready, 1);

        // Wait for multicast data
        noc_semaphore_wait(done_sem, VALID);

        // A data now in DATA_ADDR, compute C += A * B
        // ... computation with local B column ...
    }
}
```

## Performance Considerations

1. **Use multicast** instead of multiple unicasts for broadcasting
2. **Link multicast transactions** when sending series to same destinations
3. **Batch NOC operations** before barriers to amortize overhead
4. **Use posted writes** when acknowledgment not needed
5. **Minimize atomic operations** in tight loops (add latency)
6. **Double buffer** for overlapping communication and computation
7. **Use remote circular buffers** carefully - they add synchronization overhead

## Key References

- NOC APIs: [`dataflow_api.h`](../../../../../../tt_metal/hw/inc/dataflow_api.h)
- NOC Implementation: [`noc_nonblocking_api.h`](../../../../../../tt_metal/hw/inc/wormhole/noc_nonblocking_api.h)
- Address Generation: [`dataflow_api_addrgen.h`](../../../../../../tt_metal/hw/inc/dataflow_api_addrgen.h)
- Remote CBs: [`remote_circular_buffer_api.h`](../../../../../../tt_metal/hw/inc/remote_circular_buffer_api.h)
- Examples: [`test_kernels/dataflow/`](../../../../../../tests/tt_metal/tt_metal/test_kernels/dataflow/)