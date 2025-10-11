# TT-Metal Host Programming APIs and Patterns

## Overview

TT-Metal provides a comprehensive host API for controlling Tenstorrent AI accelerators. The programming model follows an explicit resource management approach similar to OpenCL/CUDA, giving developers direct control over device resources, memory allocation, and kernel execution.

## 1. Device Management

### Opening and Closing Devices

From [`tt_metal/api/tt-metalium/host_api.hpp`](../../../../../../tt_metal/api/tt-metalium/host_api.hpp):

```cpp
// Query available devices
size_t GetNumAvailableDevices();
size_t GetNumPCIeDevices();
bool IsGalaxyCluster();

// Create and initialize a device
IDevice* CreateDevice(
    chip_id_t device_id,
    uint8_t num_hw_cqs = 1,                          // Hardware command queues
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,    // L1 small region size
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE
);

// Close device
bool CloseDevice(IDevice* device);
```

**Basic Usage Example:**

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

int main() {
    // Open device 0
    int device_id = 0;
    IDevice* device = tt::tt_metal::CreateDevice(device_id);

    // Query device properties
    tt::ARCH arch = device->arch();  // GRAYSKULL, WORMHOLE_B0, BLACKHOLE
    CoreCoord grid_size = device->logical_grid_size();
    uint32_t l1_size = device->l1_size_per_core();
    int num_dram_channels = device->num_dram_channels();

    // Use device...

    // Cleanup
    tt::tt_metal::CloseDevice(device);
    return 0;
}
```

### Multi-Device and Mesh Configuration

For systems with multiple devices (Galaxy, T3000), use mesh configuration:

From [`models/demos/qwen3/conftest.py`](../conftest.py):

```python
import ttnn

# Auto-detect mesh configuration
device_ids = ttnn.get_device_ids()

if len(device_ids) == 32:  # Galaxy system
    mesh_shape = ttnn.MeshShape(4, 8)
elif len(device_ids) == 8:  # T3000 system
    mesh_shape = ttnn.MeshShape(1, 8)
else:  # Single device
    mesh_shape = ttnn.MeshShape(1, len(device_ids))

# Configure device parameters
device_params = {
    "mesh_shape": mesh_shape,
    "trace_region_size": 128 * 1024 * 1024,  # 128MB for trace
    "fabric_config": ttnn.FabricConfig.FABRIC_1D  # 1D fabric for communication
}

# Open mesh device
mesh_device = ttnn.open_mesh_device(**device_params)

# Use device...

# Cleanup
ttnn.close_mesh_device(mesh_device)
```

### Device Properties and Queries

```cpp
class IDevice {
    // Architecture and ID
    tt::ARCH arch() const;
    chip_id_t id() const;

    // Memory information
    int num_dram_channels() const;
    uint32_t l1_size_per_core() const;
    uint32_t dram_size_per_channel() const;

    // Grid dimensions
    CoreCoord compute_with_storage_grid_size() const;  // Full grid
    CoreCoord logical_grid_size() const;              // Compute cores only

    // Core type queries
    const CoreRangeSet& worker_cores(
        HalProgrammableCoreType type = HalProgrammableCoreType::TENSIX,
        SubDeviceId sub_device_id = SubDeviceId{0}) const;

    std::unordered_set<CoreCoord> get_active_ethernet_cores(
        bool skip_reserved = true) const;

    // Coordinate conversion
    CoreCoord worker_core_from_logical_core(const CoreCoord& logical) const;
    CoreCoord ethernet_core_from_logical_core(const CoreCoord& logical) const;
};
```

## 2. Programs and Kernels

### Creating Programs

A Program is a container for kernels, circular buffers, and semaphores:

```cpp
#include <tt-metalium/program.hpp>

Program program = CreateProgram();
```

### Kernel Types and Configuration

From [`tt_metal/api/tt-metalium/kernel_types.hpp`](../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp):

#### Data Movement Kernels (Reader/Writer)

```cpp
// Reader kernel configuration
ReaderDataMovementConfig reader_config{
    .processor = DataMovementProcessor::RISCV_1,  // NCRISC
    .noc = NOC::RISCV_1_default,
    .compile_args = {/* compile-time constants */}
};

// Writer kernel configuration
WriterDataMovementConfig writer_config{
    .processor = DataMovementProcessor::RISCV_0,  // BRISC
    .noc = NOC::RISCV_0_default,
    .compile_args = {/* compile-time constants */}
};
```

#### Compute Kernels

```cpp
ComputeConfig compute_config{
    .math_fidelity = MathFidelity::HiFi4,      // Accuracy level
    .fp32_dest_acc_en = false,                 // FP32 accumulation
    .math_approx_mode = false,                 // Approximation mode
    .compile_args = {/* compile-time constants */}
};
```

### Creating Kernels

```cpp
// Create kernel on single core
CoreCoord core = {0, 0};
KernelHandle reader_kernel = CreateKernel(
    program,
    "path/to/reader_kernel.cpp",
    core,
    reader_config
);

// Create kernel on multiple cores
CoreRange all_cores({0, 0}, {7, 7});  // 8×8 grid
KernelHandle compute_kernel = CreateKernel(
    program,
    "path/to/compute_kernel.cpp",
    all_cores,
    compute_config
);
```

### Runtime Arguments vs Compile-Time Arguments

#### Compile-Time Arguments
- Embedded in kernel binary at compilation
- Used for constants: dimensions, data formats, block sizes
- Accessed in kernel: `get_compile_time_arg_val(index)`
- Changing requires kernel recompilation

#### Runtime Arguments
- Set dynamically before each program execution
- Used for dynamic values: buffer addresses, tile IDs, strides
- Maximum 341 args per kernel (shared across unique and common)
- Accessed in kernel: `get_arg_val<uint32_t>(index)`

### Setting Runtime Arguments

```cpp
// Set unique args for specific core
std::vector<uint32_t> reader_args = {
    buffer->address(),      // Arg 0: buffer address
    start_tile_id,         // Arg 1: starting tile
    stride_w,              // Arg 2: width stride
    stride_h               // Arg 3: height stride
};
SetRuntimeArgs(program, reader_kernel, core, reader_args);

// Set different args for multiple cores
std::vector<CoreCoord> cores = {{0,0}, {1,0}, {2,0}};
std::vector<std::vector<uint32_t>> per_core_args = {
    {buffer->address(), 0, 1, 8},   // Core (0,0)
    {buffer->address(), 8, 1, 8},   // Core (1,0)
    {buffer->address(), 16, 1, 8}   // Core (2,0)
};
SetRuntimeArgs(program, reader_kernel, cores, per_core_args);

// Set common args (shared by all cores)
std::vector<uint32_t> common_args = {num_tiles, tile_size};
SetCommonRuntimeArgs(program, compute_kernel, common_args);
```

### Dynamic Runtime Arguments (Callbacks)

For operations where buffer addresses change between invocations:

```cpp
auto override_runtime_args_callback = [reader_kernel, writer_kernel](
    const void* operation,
    const Program& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>&,
    const std::vector<Tensor>& output_tensors
) {
    // Update buffer addresses when tensors are reallocated
    auto& reader_args = GetRuntimeArgs(program, reader_kernel, core);
    reader_args[0] = input_tensors[0].buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, writer_kernel, core);
    writer_args[0] = output_tensors[0].buffer()->address();
};

return {std::move(program), override_runtime_args_callback};
```

## 3. Buffer Management

### Buffer Types

From [`tt_metal/api/tt-metalium/buffer_types.hpp`](../../../../../../tt_metal/api/tt-metalium/buffer_types.hpp):

```cpp
enum class BufferType {
    DRAM,           // Device DRAM
    L1,             // L1 SRAM (large region)
    L1_SMALL,       // L1 SRAM (small region)
    SYSTEM_MEMORY,  // Host memory
    TRACE           // Trace buffer
};

enum class TensorMemoryLayout {
    INTERLEAVED = 0,     // Pages distributed across banks/cores
    HEIGHT_SHARDED = 2,  // Sharded along height dimension
    WIDTH_SHARDED = 3,   // Sharded along width dimension
    BLOCK_SHARDED = 4    // 2D block sharding
};
```

### Creating Interleaved Buffers

Interleaved buffers distribute data across DRAM banks for parallel access:

```cpp
uint32_t buffer_size = num_tiles * tile_size;
uint32_t page_size = tile_size;  // Page = unit of interleaving

InterleavedBufferConfig config{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::DRAM
};

auto buffer = CreateBuffer(config);
```

### Creating Sharded Buffers

Sharded buffers distribute data across compute cores:

```cpp
ShardedBufferConfig sharded_config{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::L1,
    .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
    .shard_parameters = ShardSpecBuffer{
        .core_set = worker_cores,
        .shape = {shard_height, shard_width},
        .orientation = ShardOrientation::ROW_MAJOR
    }
};

auto sharded_buffer = CreateBuffer(sharded_config);
```

### Buffer Operations

```cpp
// Query buffer properties
uint32_t address = buffer->address();
DeviceAddr size = buffer->size();
uint32_t page_size = buffer->page_size();
bool is_dram = buffer->is_dram();
bool is_l1 = buffer->is_l1();

// Deallocate buffer
DeallocateBuffer(*buffer);

// Assign buffer to program (for async execution)
AssignGlobalBufferToProgram(buffer, program);
```

## 4. Circular Buffers

Circular buffers provide synchronized communication between kernels:

From [`tt_metal/api/tt-metalium/circular_buffer_config.hpp`](../../../../../../tt_metal/api/tt-metalium/circular_buffer_config.hpp):

### Creating Circular Buffers

```cpp
// Standard CB indices
constexpr uint32_t cb_in0 = 0;      // Input buffer 0
constexpr uint32_t cb_in1 = 1;      // Input buffer 1
constexpr uint32_t cb_out = 16;     // Output buffer

// Create input CB with double buffering
uint32_t tiles_per_cb = 2;  // Double buffering
CircularBufferConfig cb_config(
    tiles_per_cb * tile_size,
    {{cb_in0, DataFormat::Float16_b}}  // CB index → data format
);
cb_config.set_page_size(cb_in0, tile_size);

CBHandle cb = CreateCircularBuffer(program, core, cb_config);

// Create CB with multiple indices sharing same memory
std::map<uint8_t, DataFormat> multi_cb_spec = {
    {16, DataFormat::Float16_b},  // Output
    {24, DataFormat::Float16_b}   // Intermediate
};
CircularBufferConfig multi_cb_config(total_size, multi_cb_spec);
multi_cb_config.set_page_size(16, tile_size);
multi_cb_config.set_page_size(24, tile_size);

CBHandle multi_cb = CreateCircularBuffer(program, core, multi_cb_config);
```

### Dynamic Circular Buffers

Share L1 address with an L1 buffer:

```cpp
// Create L1 buffer
auto l1_buffer = CreateBuffer(InterleavedBufferConfig{
    .device = device,
    .size = buffer_size,
    .page_size = page_size,
    .buffer_type = BufferType::L1
});

// Create CB sharing the L1 buffer's address
CircularBufferConfig dynamic_cb_config(
    buffer_size,
    {{0, DataFormat::Float16_b}},
    *l1_buffer  // Share address with L1 buffer
);

CBHandle dynamic_cb = CreateCircularBuffer(program, core, dynamic_cb_config);
```

## 5. Command Queue and Execution

### Getting the Command Queue

```cpp
// Get default command queue (CQ 0)
CommandQueue& cq = device->command_queue();

// Get specific command queue
CommandQueue& cq1 = device->command_queue(1);
```

### Writing Data to Device

```cpp
// Write std::vector to buffer (non-blocking)
std::vector<float> input_data(buffer_size);
EnqueueWriteBuffer(cq, buffer, input_data, /*blocking=*/false);

// Write raw pointer to buffer (blocking)
float* raw_data = new float[buffer_size];
EnqueueWriteBuffer(cq, buffer, raw_data, /*blocking=*/true);

// Write to sub-region of buffer
BufferRegion region{
    .offset = 1024,     // Byte offset
    .size = 512        // Bytes to write
};
EnqueueWriteSubBuffer(cq, buffer, data_ptr, region, /*blocking=*/false);
```

### Executing Programs

```cpp
// Enqueue program execution
EnqueueProgram(cq, program, /*blocking=*/false);

// Can enqueue multiple programs
EnqueueProgram(cq, program1, false);
EnqueueProgram(cq, program2, false);
EnqueueProgram(cq, program3, false);
```

### Reading Data from Device

```cpp
// Read buffer to std::vector (blocking)
std::vector<float> output_data(buffer_size);
EnqueueReadBuffer(cq, buffer, output_data, /*blocking=*/true);

// Read to raw pointer (non-blocking)
float* result = new float[buffer_size];
EnqueueReadBuffer(cq, buffer, result, /*blocking=*/false);

// Wait for read to complete
Finish(cq);
```

### Synchronization

```cpp
// Wait for all commands in queue to complete
Finish(cq);

// Synchronize specific device/queue
Synchronize(device);
Synchronize(device, /*cq_id=*/0);

// Events for fine-grained synchronization
auto event = std::make_shared<Event>(device);

// Record event after current commands
EnqueueRecordEvent(cq, event);

// Wait for event on device
EnqueueWaitForEvent(cq, event);

// Host waits for event
EventSynchronize(event);

// Query event status (non-blocking)
bool completed = EventQuery(event);
```

## 6. Complete Example: Matrix Multiplication

Here's a complete example showing all components together:

```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

void run_matmul(uint32_t M, uint32_t N, uint32_t K) {
    // 1. Initialize device
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // 2. Calculate sizes
    uint32_t tile_size = 32 * 32 * sizeof(bfloat16);
    uint32_t a_tiles = M * K;
    uint32_t b_tiles = K * N;
    uint32_t c_tiles = M * N;

    // 3. Create buffers
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = a_tiles * tile_size,
        .page_size = tile_size,
        .buffer_type = BufferType::DRAM
    };

    auto a_buffer = CreateBuffer(dram_config);
    auto b_buffer = CreateBuffer(dram_config);
    dram_config.size = c_tiles * tile_size;
    auto c_buffer = CreateBuffer(dram_config);

    // 4. Create circular buffers
    CoreCoord core = {0, 0};
    uint32_t cb_tiles = 2;  // Double buffering

    CircularBufferConfig cb_a_config(
        cb_tiles * tile_size,
        {{0, DataFormat::Float16_b}}
    );
    cb_a_config.set_page_size(0, tile_size);
    CreateCircularBuffer(program, core, cb_a_config);

    CircularBufferConfig cb_b_config(
        cb_tiles * tile_size,
        {{1, DataFormat::Float16_b}}
    );
    cb_b_config.set_page_size(1, tile_size);
    CreateCircularBuffer(program, core, cb_b_config);

    CircularBufferConfig cb_c_config(
        cb_tiles * tile_size,
        {{16, DataFormat::Float16_b}}
    );
    cb_c_config.set_page_size(16, tile_size);
    CreateCircularBuffer(program, core, cb_c_config);

    // 5. Create kernels
    std::vector<uint32_t> reader_compile_args = {
        (uint32_t)a_buffer->buffer_type() == BufferType::DRAM ? 1 : 0,
        (uint32_t)b_buffer->buffer_type() == BufferType::DRAM ? 1 : 0
    };

    KernelHandle reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_matmul.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_args)
    );

    std::vector<uint32_t> compute_compile_args = {M, N, K};
    KernelHandle compute_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/matmul.cpp",
        core,
        ComputeConfig{.compile_args = compute_compile_args}
    );

    KernelHandle writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_matmul.cpp",
        core,
        WriterDataMovementConfig()
    );

    // 6. Set runtime arguments
    SetRuntimeArgs(program, reader_kernel, core, {
        a_buffer->address(),
        b_buffer->address(),
        a_tiles,
        b_tiles
    });

    SetRuntimeArgs(program, compute_kernel, core, {
        M, N, K
    });

    SetRuntimeArgs(program, writer_kernel, core, {
        c_buffer->address(),
        c_tiles
    });

    // 7. Execute
    std::vector<bfloat16> a_data(a_tiles * 32 * 32);
    std::vector<bfloat16> b_data(b_tiles * 32 * 32);
    // Initialize a_data and b_data...

    EnqueueWriteBuffer(cq, a_buffer, a_data, false);
    EnqueueWriteBuffer(cq, b_buffer, b_data, false);
    EnqueueProgram(cq, program, false);

    std::vector<bfloat16> c_data(c_tiles * 32 * 32);
    EnqueueReadBuffer(cq, c_buffer, c_data, true);

    // 8. Cleanup
    CloseDevice(device);
}
```

## 7. Multi-Core Patterns

### Working with Core Ranges

```cpp
// Single core
CoreCoord single_core = {0, 0};

// Rectangular range
CoreRange range({0, 0}, {3, 3});  // 4×4 grid

// Multiple ranges
CoreRangeSet multi_range({
    CoreRange({0, 0}, {3, 3}),  // First 4×4 block
    CoreRange({4, 0}, {7, 3})   // Second 4×4 block
});

// Convert number of cores to CoreRangeSet
uint32_t num_cores = 32;
CoreCoord grid_size = device->compute_with_storage_grid_size();
CoreRangeSet cores = num_cores_to_corerangeset(
    num_cores, grid_size, /*row_major=*/true
);
```

### Distributing Work Across Cores

```cpp
void distribute_matmul(Device* device, uint32_t M, uint32_t N, uint32_t K) {
    Program program = CreateProgram();

    // Calculate work distribution
    uint32_t num_cores_x = 4, num_cores_y = 4;
    uint32_t per_core_M = M / num_cores_y;
    uint32_t per_core_N = N / num_cores_x;

    CoreRangeSet all_cores = num_cores_to_corerangeset(
        num_cores_x * num_cores_y,
        device->compute_with_storage_grid_size(),
        true
    );

    // Create kernels on all cores
    auto compute_kernel = CreateKernel(
        program,
        "matmul_kernel.cpp",
        all_cores,
        ComputeConfig{}
    );

    // Set per-core runtime args
    for (uint32_t y = 0; y < num_cores_y; y++) {
        for (uint32_t x = 0; x < num_cores_x; x++) {
            CoreCoord core = {x, y};

            uint32_t start_m = y * per_core_M;
            uint32_t start_n = x * per_core_N;

            SetRuntimeArgs(program, compute_kernel, core, {
                start_m, start_n, per_core_M, per_core_N, K
            });
        }
    }
}
```

## 8. Best Practices

### Program Factory Pattern

Encapsulate program creation for reuse:

```cpp
class MatmulProgramFactory {
    struct CachedProgram {
        Program program;
        KernelHandle reader, compute, writer;
        // Cache key parameters
        uint32_t M, N, K;
    };

    std::unordered_map<size_t, CachedProgram> cache_;

public:
    Program& get_or_create(Device* device, uint32_t M, uint32_t N, uint32_t K) {
        size_t key = hash_combine(M, N, K);

        if (cache_.find(key) == cache_.end()) {
            cache_[key] = create_matmul_program(device, M, N, K);
        }

        return cache_[key].program;
    }
};
```

### Double Buffering for Performance

```cpp
// Size CBs for double buffering
uint32_t single_buffer_tiles = 32;
uint32_t double_buffer_tiles = single_buffer_tiles * 2;

CircularBufferConfig double_buffer_config(
    double_buffer_tiles * tile_size,
    {{0, DataFormat::Float16_b}}
);

// Reader can fill buffer N+1 while compute processes buffer N
```

### Error Handling

```cpp
// Validate before execution
TT_FATAL(M % 32 == 0, "M must be divisible by 32");
TT_FATAL(buffer->is_allocated(), "Buffer not allocated");

// Check device properties
if (num_tiles > device->l1_size_per_core() / tile_size) {
    throw std::runtime_error("Data exceeds L1 capacity");
}
```

### Performance Tips

1. **Use non-blocking operations** - Queue multiple operations before synchronization
2. **Batch NOC transfers** - Issue multiple reads/writes before barriers in kernels
3. **Double buffer circular buffers** - Overlap computation and data movement
4. **Reuse programs** - Compile once, execute many times with different runtime args
5. **Use callbacks** - Update buffer addresses dynamically without recompilation

## Key References

- Host API: [`host_api.hpp`](../../../../../../tt_metal/api/tt-metalium/host_api.hpp)
- Device Interface: [`device.hpp`](../../../../../../tt_metal/api/tt-metalium/device.hpp)
- Program API: [`program.hpp`](../../../../../../tt_metal/api/tt-metalium/program.hpp)
- Buffer API: [`buffer.hpp`](../../../../../../tt_metal/api/tt-metalium/buffer.hpp)
- Kernel Types: [`kernel_types.hpp`](../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp)
- Examples: [`tests/tt_metal/`](../../../../../../tests/tt_metal/)
- Production Patterns: [`ttnn/cpp/ttnn/operations/`](../../../../../../ttnn/cpp/ttnn/operations/)