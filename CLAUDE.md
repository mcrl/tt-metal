# CLAUDE.md

Instructions for Claude Code when working with this repository.

## Core Rules

- **Follow ALL instructions** - not optional unless explicitly stated
- **Ask for clarification** if uncertain
- **Only make requested changes** - no refactoring, optimization, or "improvements" unless asked
- **Always use English** for comments and documentation
- **Always use `tt-base` conda environment** - prefix commands with `conda run -n tt-base`

## Repository Overview

TT-Metal (Metalium): Low-level programming framework for Tenstorrent AI accelerators
- **TT-Metalium**: Kernel development framework
- **TT-NN**: High-level neural network operations (PyTorch-like API)
- **Model Demos**: Production LLM/CNN implementations

See [METALIUM_GUIDE.md](METALIUM_GUIDE.md) for architecture, [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Qwen3 MoE Development

### Scope Restrictions

**ONLY modify code in these directories:**
- `models/demos/qwen3/` - Qwen3-MoE model implementation
- `ttnn/cpp/ttnn/operations/experimental/moe/` - Custom MoE TTNN operations

**Minimal external changes allowed:**
- `ttnn/cpp/ttnn/operations/experimental/experimental_pybind.cpp` - Python binding registration
- `ttnn/CMakeLists.txt` - Build system integration (3 locations: TTNN_SRC_PYBIND, target_link_libraries, add_subdirectory)

**Ask for clarification** before modifying any other framework code.

### Directory Structure

```
models/demos/qwen3/
├── tt/                    # TT-NN components (attention, moe, qwen, sdpa, rope, rms_norm, ccl_1d)
├── reference/             # PyTorch reference for validation
├── tests/                 # Component tests
├── common/                # loader, configuration
├── utils/                 # profiler, timer, device, memory_state, test_utils
├── simulator/             # hw specs, param, flops
├── test_dataset/          # dataset loader/maker
├── main.py               # Inference entry point
├── benchmark.py          # Benchmarking
├── generation.py         # Qwen3MoETT (TT) and Qwen3MoEReference (PyTorch)
└── conftest.py           # pytest device mesh fixtures

ttnn/cpp/ttnn/operations/experimental/moe/
├── CMakeLists.txt
├── prepare_moe_routing_tensors/       # Routing tensors op
│   ├── *.hpp, *.cpp, *_pybind.*
│   └── device/
│       ├── *_op.*, *_program_factory.*
│       └── kernels/dataflow/reader_writer_moe_routing.cpp
├── scatter_moe_input/                 # Scatter input tokens op
│   └── (same structure)
├── local_reduce_moe_output/           # Local reduce MoE output op
│   └── (same structure)
└── moe_bmm/                           # MoE batched matrix multiply op
    └── (same structure)
```

### Running Tests

**Always use timeout (default 30s)** to prevent device hangs:

```bash
cd models/demos/qwen3

# Component tests (30s timeout)
timeout 30 conda run -n tt-base pytest tests/test_moe.py -v
timeout 30 conda run -n tt-base pytest tests/test_attn.py -v

# MoE operation tests
timeout 30 conda run -n tt-base pytest tests/test_moe_routing_tensors.py -v
timeout 30 conda run -n tt-base pytest tests/test_scatter_moe_input.py -v
timeout 30 conda run -n tt-base pytest tests/test_local_reduce_moe_output.py -v

# All tests (60s)
timeout 60 conda run -n tt-base pytest tests/ -v

# Detailed output (disable buffering with -s)
timeout 30 conda run -n tt-base pytest tests/test_moe.py -vv -s

# Save output for analysis
timeout 30 conda run -n tt-base pytest tests/test_moe.py -vv -s 2>&1 | tee /tmp/test_output.log
```

**Timeout guidelines**: 30s (unit tests), 60s (component tests), 120s+ (integration/full suites)

### Running Inference

```bash
cd models/demos/qwen3
python main.py \
  --ckpt_dir=/shared/models/Qwen3-30B-A3B/ \
  --tokenizer_path=/shared/models/Qwen3-30B-A3B/tokenizer.json \
  --config_path=/shared/models/Qwen3-30B-A3B/config.json
```

### Device Configuration

Mesh configurations auto-detected in `conftest.py`:
- Galaxy (32 devices): 4x8 mesh
- T3000 (8 devices): 1x8 mesh
- Single device: 1x1 mesh

Common device params:
```python
device_params = {
    "trace_region_size": 128 * 1024 * 1024,  # 128MB
    "fabric_config": ttnn.FabricConfig.FABRIC_1D
}
```

### Development Workflow

1. Edit `tt/*.py` files
2. Run tests: `timeout 30 conda run -n tt-base pytest tests/test_moe.py -v`
3. Compare with PyTorch reference in `reference/`
4. Profile: `python benchmark.py --batch_sizes 32 --input_lengths 128`
5. Debug with utilities: `utils/timer.py`, `profiler.py`, `memory_state.py`

## Environment Setup

### Conda Environment

```bash
conda activate tt-base
conda env list  # Verify * next to tt-base
```

Pre-configured with Python dependencies, PyTorch, ML libraries, dev tools.

### Building

```bash
# Initial build (first time only)
conda run -n tt-base ./build_metal.sh -p  # REQUIRED for Tracy profiler (-p = --enable-profiler)

# Incremental builds (faster, use after C++ changes)
conda run -n tt-base cmake --build build -j16
```

**Note**: Always use incremental builds (`cmake --build`) after modifying C++ code. Only run `./build_metal.sh -p` for initial setup or when build configuration changes.

### Environment Variables

- `TT_METAL_HOME`: Repository root
- `TT_LOGGER_LEVEL`: Debug, Info, Warning, Error
- `TT_LOGGER_TYPES`: Op, Metal, Device, etc.
- `TT_METAL_WATCHER`: Enable device watcher

## TTNN API Development

### Architecture

```
Python API (ttnn.operation_name)
    ↓ pybind11
C++ Operation Registration (ttnn::register_operation)
    ↓
C++ Operation Implementation
    ↓
Device Programs & Kernels (TT-Metal)
```

### Operation Types

1. **Primitive**: Direct device programs with kernels (e.g., concat, binary ops)
2. **Composite**: Pure C++ calling other ops (e.g., to_dtype)
3. **Experimental**: New/unstable ops in `ttnn/cpp/ttnn/operations/experimental/`

### Adding New Operation

**Directory structure:**
```
ttnn/cpp/ttnn/operations/experimental/<op_name>/
├── CMakeLists.txt
├── <op_name>.hpp, .cpp              # Operation
├── <op_name>_pybind.hpp, .cpp       # Python binding
└── device/                           # Primitive ops only
    ├── <op_name>_op.hpp, .cpp
    ├── <op_name>_program_factory.hpp, .cpp
    └── kernels/
        ├── reader_*.cpp              # Data movement
        └── compute_*.cpp             # Compute (optional)
```

**Integration (3 files to modify):**
1. `experimental_pybind.cpp`: Add include + binding call
2. `ttnn/CMakeLists.txt`: Add to TTNN_SRC_PYBIND list (~line 200)
3. `ttnn/CMakeLists.txt`: Add to target_link_libraries (~line 494) + add_subdirectory (~line 647)

**Naming conventions:**
- C++ namespace: `ttnn::operations::<category>`
- Operation struct: `<OperationName>Operation` (PascalCase + "Operation")
- Registered operation: `ttnn::my_operation` (snake_case)
- CMake target: `ttnn_op_<category>_<operation_name>` (snake_case)
- CMake alias: `TTNN::Ops::<Category>::<OperationName>` (PascalCase)

**Argument ordering:**
- C++ API: `invoke(QueueId queue_id, <tensor_args>, <other_args>, <optional_args>)`
- Python API: `operation(<tensor_args>, <other_args>, *, <optional_args>, queue_id=0)`

**Key patterns:**
- Tensor args: `.noconvert()` to avoid copies
- Use `std::optional<MemoryConfig>` with default `std::nullopt`
- Use `py::kw_only()` for keyword-only Python args
- Accessor methods: `tensor.dtype()`, `tensor.padded_shape()`, `tensor.layout()` (no `get_` prefix)
- DataType → DataFormat: `tt_metal::datatype_to_dataformat_converter(dtype)`
- CircularBufferConfig: `CircularBufferConfig(size, {{cb_index, data_format}}).set_page_size(cb_index, tile_size)`
- Include paths: `<tt-metalium/...>` not `"tt_metal/..."`

**Reference operations:**
- Simple primitive: `ttnn/cpp/ttnn/operations/experimental/plusone/`
- Complex primitive: `ttnn/cpp/ttnn/operations/data_movement/concat/`
- Composite: `ttnn/cpp/ttnn/operations/core/to_dtype/`

### MoE Implementation Lessons

- Multi-output ops: Use `std::vector<Tensor>` and `compute_output_specs()` returning `std::vector<TensorSpec>`
- Use designated initializers: `{.field = value}`
- Namespace: Operation struct in `operations::experimental::moe`, helpers in `detail`
- Run device ops: `tt::tt_metal::operation::run()`
- Python bindings: Include `"ttnn-pybind/decorators.hpp"`
- Must copy `_ttnn.so` to package directory after build
- **Kernels**: Never use stack arrays for NOC ops; use L1 circular buffers
- **Interleaved writes**: Read → modify in L1 → write full row
- **Testing**: Import `tt_lock` for device sync; parametrize tests; validate vs PyTorch reference

## TT-Metal Kernel Development

### IMPORTANT: Required Documentation Review

**Before writing any kernels, ALWAYS read the documentation in `models/demos/qwen3/docs/`:**
- `01_tenstorrent_chip_architecture.md` - Chip architecture, coordinate systems, core types
- `02_tenstorrent_tensix_core.md` - Tensix core architecture and capabilities
- `03_host_programming_apis.md` - Host-side programming patterns
- `04_kernel_programming.md` - Kernel programming guide
- `05_inter_core_communication.md` - Inter-core communication patterns

These documents contain critical information about hardware constraints, best practices, and implementation patterns that must be understood before kernel development.

### Kernel Types

1. **Data Movement Kernels** (RISC-V processors)
   - **Reader kernels**: Read data from DRAM to L1 circular buffers
   - **Writer kernels**: Write data from L1 circular buffers to DRAM
   - Run on RISC-V cores, use NOC for data transfer
   - Example: `reader_bmm_tile_layout.cpp`, `writer_bmm_tile_layout.cpp`

2. **Compute Kernels** (Tensix compute cores)
   - Matrix multiplication (`matmul_tiles`), element-wise ops
   - SFPU operations (special functions: exp, log, etc.)
   - Acquire/release destination registers, pack results to circular buffers
   - Example: `bmm_large_block_zm.cpp`, `eltwise_binary_kernel.cpp`

3. **Ethernet Kernels** (multi-device communication)
   - Data transfer between devices on fabric

### Study Reference Kernels

**MUST review existing kernels before implementation:**

**Simple patterns (for learning):**
- Fill operation: `ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp`
- Binary eltwise: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- Simple matmul: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp` - outer product, single tile at a time

**High-performance matmul (study for optimization):**
- Reader: `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout.cpp`
- Writer: `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_tile_layout.cpp`
- Compute (large blocks): `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp`
- Program factory: `ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp`

**Advanced patterns (for complex scenarios):**
- Concat with multiple tensors: `reader_concat_interleaved_start_id.cpp` - shows multi-tensor accessor pattern
- Attention matmul: `transformer_attn_matmul.cpp` - tilize/untilize, data format reconfig
- Sharded inputs, broadcast, multi-core distribution in `ttnn/cpp/ttnn/operations/matmul/device/kernels/`

### Core Kernel Patterns

#### Data Movement Kernel Structure
```cpp
void kernel_main() {
    // 1. Get runtime arguments
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    // 2. Get compile-time arguments
    constexpr bool is_dram = get_compile_time_arg_val(0) == 1;

    // 3. Setup tensor accessor
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto accessor = TensorAccessor(accessor_args, src_addr, tile_size);

    // 4. Loop: read tiles from DRAM to CB
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, accessor, l1_addr);
        noc_async_read_barrier();  // CRITICAL
        cb_push_back(cb_id, 1);
    }
}
```

#### Compute Kernel Structure
```cpp
void MAIN {
    // 1. Get compile-time args
    uint32_t num_blocks = get_compile_time_arg_val(0);
    uint32_t block_size = get_compile_time_arg_val(1);

    // 2. Initialize compute operation
    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24);

    // 3. Main computation loop
    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_wait_front(tt::CBIndex::c_0, in0_tiles);
        cb_wait_front(tt::CBIndex::c_1, in1_tiles);

        acquire_dst();  // Acquire destination registers

        // Perform computation
        for (uint32_t i = 0; i < block_size; i++) {
            matmul_tiles(cb_in0, cb_in1, i, i, dst_idx, false);
        }

        // Pack results to output CB
        cb_reserve_back(tt::CBIndex::c_16, out_tiles);
        for (uint32_t i = 0; i < out_tiles; i++) {
            pack_tile(i, tt::CBIndex::c_16);
        }
        cb_push_back(tt::CBIndex::c_16, out_tiles);

        release_dst();  // Release destination registers

        cb_pop_front(tt::CBIndex::c_0, in0_tiles);
        cb_pop_front(tt::CBIndex::c_1, in1_tiles);
    }
}
```

### High-Performance Matmul Insights

**Comparison: Simple (`bmm.cpp`) vs Optimized (`bmm_large_block_zm.cpp`):**

**Simple matmul (`bmm.cpp`):**
- Outer product: for each output tile C[m,n], compute full K reduction
- Single tile at a time: `cb_wait_front(c_0, 1)`, `cb_wait_front(c_1, 1)`
- No intermediate buffering: accumulates in DST, packs directly to c_16
- Easy to understand but inefficient: doesn't exploit data reuse

**Optimized matmul (`bmm_large_block_zm.cpp`):**
1. **Block-based computation**: Loads blocks of tiles to exploit data reuse in L1
   - Block size: `per_core_M × in0_block_w` (A) and `in0_block_w × per_core_N` (B)
   - Reduces DRAM accesses by reusing loaded blocks

2. **Subblock tiling**: Divides output block into subblocks for optimal DST register utilization
   - Subblock size: `out_subblock_h × out_subblock_w` (e.g., 4×2 tiles)
   - Each subblock computed separately, packed immediately
   - Triple nested loops: `in0_num_subblocks` × `in1_num_subblocks` → per subblock: `out_subblock_h` × `out_subblock_w` → per output tile: `in0_block_w` K-tiles

3. **Spilling & reloading**: For large K (num_blocks > 1), accumulates partial results
   - First K-block: compute subblock → pack to c_24 (interm)
   - Later K-blocks: reload from c_24 → accumulate → pack to c_24
   - Last K-block: pack to c_16 (output)
   - Pattern: `enable_reload` flag controls accumulation vs fresh computation
   - Memory sharing: c_16 (output) and c_24 (interm) share same physical L1 space (different CB indices)
   - Reload mechanism:
     ```cpp
     if (enable_reload) {
         copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_1, tt::CBIndex::c_24);
         cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
         for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
             copy_tile(tt::CBIndex::c_24, i, i);  // Load partial result to DST
         }
         cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
         mm_init_short_with_dt(...);  // Re-init for accumulation
     }
     ```

4. **Double buffering**: Input CBs sized 2× block size
   - While compute processes block N, reader fetches block N+1 in parallel
   - Critical for hiding DRAM latency
   - Formula: `in0_CB_size = in0_block_tiles * 2 * tile_size`

5. **Batch processing**: Outer loop handles batch dimension with stride calculations
   - Batch broadcast support: `bcast_B` flag controls B tensor stride (0 = reuse same B across batch)

**Memory hierarchy:**
- c_0, c_1: Input CBs (double buffered: 2× block size)
- c_16: Output CB (single buffered: 1× block size)
- c_24: Intermediate buffer (spill/reload for multi-block K reduction)

**Reader kernel pattern:**
- Uses `InterleavedAddrGenFast` for efficient address generation
  ```cpp
  const InterleavedAddrGenFast<is_dram> s = {
      .bank_base_address = buffer_addr,
      .page_size = tile_size_bytes,
      .data_format = data_format
  };
  ```
- Nested loops: blocks × rows × cols with stride calculations
- 2D traversal: `in0_tensor_stride_w=1`, `in0_tensor_stride_h=K` for row-major tile layout
- Block stride: `in0_tensor_next_block_stride` advances to next K-block (typically `in0_block_w`)

**Writer kernel pattern:**
- Nested subblock loops: `out_num_subblocks_h` × `out_num_subblocks_w`
- Each subblock: `out_subblock_h` rows × `out_subblock_w` cols
- Stride calculations: `out_tensor_stride_w=1`, `out_tensor_stride_h=N`
- Subblock strides: `out_tensor_next_subblock_stride_w = out_subblock_w`, `out_tensor_next_subblock_stride_h = out_subblock_h * N`
- `cb_wait_front(c_16, out_subblock_tile_count)` waits for compute to complete
- Uses `InterleavedAddrGenFast` same as reader for consistent tile addressing

**Compute kernel indexing logic (critical for correctness):**
```cpp
// From bmm_large_block_zm.cpp - shows how to index tiles in CBs
int in0_index_subblock_offset = 0;
for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
    int in1_index_subblock_offset = 0;
    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
        int dst_index = 0;
        int in0_index_h_offset = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                int in1_index_inner_dim_offset = 0;
                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                    // Key formula: offset + row offset + inner dimension
                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, in0_index, in1_index, dst_index, false);
                    in1_index_inner_dim_offset += in1_per_core_w;  // Stride down K dimension
                }
                dst_index++;
            }
            in0_index_h_offset += in0_block_w;  // Move to next row
        }
        in1_index_subblock_offset += out_subblock_w;  // Move to next column subblock
    }
    in0_index_subblock_offset += in0_subblock_num_tiles;  // Move to next row subblock
}
```

**Key insight**: Tile indices in CBs are **linear offsets** from CB start, not tensor coordinates. Must manually calculate based on block layout in L1.

**Attention matmul specifics (`transformer_attn_matmul.cpp`):**
- Row-wise processing: computes one row of output at a time (32 rows per tile)
- Tilize/untilize for format conversion: RM ↔ tiled
- Data format reconfiguration: `reconfig_data_format_srca()`, `pack_reconfig_data_format()`
- Multiple intermediate CBs: c_2, c_3, c_4 for tilize/untilize stages
- Use case: when output needs different layout than standard tiled format

### Circular Buffers

**32 CBs per core** (CBIndex::c_0 to c_31):
- **c_0-c_7**: Input buffers
- **c_16-c_23**: Output buffers
- **c_24-c_31**: Intermediate/scratch buffers

**Producer-consumer pattern:**
```cpp
// Producer (reader/compute)
cb_reserve_back(cb_id, n_tiles);   // Reserve space
uint32_t addr = get_write_ptr(cb_id);
// ... write data to addr ...
cb_push_back(cb_id, n_tiles);      // Mark ready

// Consumer (compute/writer)
cb_wait_front(cb_id, n_tiles);     // Wait for data
uint32_t addr = get_read_ptr(cb_id);
// ... read data from addr ...
cb_pop_front(cb_id, n_tiles);      // Free space
```

**Double buffering for performance:**
- CB size = 2× block size
- While compute processes block N, reader fetches block N+1
- Overlaps data movement with computation

### Key Kernel APIs

**Circular buffer operations:**
```cpp
void cb_reserve_back(uint32_t cb_id, uint32_t num_tiles);
void cb_push_back(uint32_t cb_id, uint32_t num_tiles);
void cb_wait_front(uint32_t cb_id, uint32_t num_tiles);
void cb_pop_front(uint32_t cb_id, uint32_t num_tiles);
uint32_t get_write_ptr(uint32_t cb_id);
uint32_t get_read_ptr(uint32_t cb_id);
uint32_t get_tile_size(uint32_t cb_id);
```

**NOC data movement:**
```cpp
void noc_async_read_tile(uint32_t tile_id, TensorAccessor& accessor, uint32_t dst_addr);
void noc_async_read(uint32_t src_addr, uint32_t dst_addr, uint32_t size);
void noc_async_read_barrier();   // Wait for all reads to complete
void noc_async_write_tile(uint32_t tile_id, TensorAccessor& accessor, uint32_t src_addr);
void noc_async_write(uint32_t src_addr, uint32_t dst_addr, uint32_t size);
void noc_async_write_barrier();  // Wait for all writes to complete
```

**Compute operations:**
```cpp
// Destination register management
void acquire_dst();              // Acquire destination registers
void release_dst();              // Release destination registers

// Tile operations
void pack_tile(uint32_t dst_idx, CBIndex out_cb);
void copy_tile(CBIndex in_cb, uint32_t in_idx, uint32_t dst_idx);

// Matrix multiplication
void mm_init(CBIndex cb_in0, CBIndex cb_in1, CBIndex cb_interm);
void matmul_tiles(CBIndex cb_in0, CBIndex cb_in1, uint32_t in0_idx,
                  uint32_t in1_idx, uint32_t dst_idx, bool transpose);

// Element-wise operations
void binary_op_init_common(CBIndex cb_in0, CBIndex cb_in1, CBIndex cb_out);
// ADD_TILES, MUL_TILES, SUB_TILES, etc. - operation-specific macros
```

**Tensor accessor (for interleaved buffers):**
```cpp
// Setup
constexpr auto accessor_args = TensorAccessorArgs<0>();
const auto accessor = TensorAccessor(accessor_args, buffer_addr, tile_size);

// For ROW_MAJOR: page_size = row_size_bytes (NOT tile_size)
const auto accessor = TensorAccessor(accessor_args, buffer_addr, row_size_bytes);

// Get NOC address for tile/page
uint64_t noc_addr = get_noc_addr(tile_id, accessor);
```

### Program Factory Pattern

**Host-side program creation** (from `matmul_op_multi_core_reuse_program_factory.cpp`):

```cpp
tt_metal::operation::ProgramWithCallbacks create_program(...) {
    tt_metal::Program program{};

    // 1. Calculate CB sizes
    uint32_t in0_cb_size = in0_tiles * 2 * tile_size;  // Double buffer
    uint32_t in1_cb_size = in1_tiles * 2 * tile_size;
    uint32_t out_cb_size = out_tiles * tile_size;      // Single buffer

    // 2. Create circular buffers
    CircularBufferConfig cb_config =
        CircularBufferConfig(in0_cb_size, {{0, data_format}})
        .set_page_size(0, tile_size);
    auto cb = CreateCircularBuffer(program, all_cores, cb_config);

    // 3. Compile-time args
    std::vector<uint32_t> compute_compile_args = {
        in0_block_w, num_subblocks, ...
    };
    std::vector<uint32_t> reader_compile_args = {
        (uint32_t)in0_is_dram, (uint32_t)in1_is_dram
    };

    // 4. Create kernels
    auto reader_kernel = CreateKernel(
        program, "path/to/reader.cpp", all_cores,
        ReaderDataMovementConfig(reader_compile_args));

    auto compute_kernel = CreateKernel(
        program, "path/to/compute.cpp", all_cores,
        ComputeConfig{.math_fidelity = fidelity, .compile_args = compute_compile_args});

    // 5. Set runtime args per core
    for (each core) {
        std::vector<uint32_t> reader_runtime_args = {
            buffer->address(), start_tile_id, stride_w, stride_h, ...
        };
        SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
    }

    return {std::move(program), callbacks};
}
```

**Key concepts:**

**Compile-time args:**
- Constants embedded in kernel binary at compile time
- Accessed via `get_compile_time_arg_val(idx)` in kernel
- Use for: buffer types (DRAM/L1), data formats, block sizes, dimensions
- Cannot change between invocations - kernel must be recompiled
- Examples: `in0_is_dram`, `in1_is_dram`, `in0_block_w`, `num_subblocks`
- Enables compile-time optimizations (loop unrolling, constant folding)

**Runtime args:**
- Dynamic values set per invocation, stored in L1
- Accessed via `get_arg_val<T>(idx)` in kernel (unique per-core) or `get_common_arg_val<T>(idx)` (shared across cores)
- Use for: buffer addresses, start tile IDs, strides, loop counts
- Can change between invocations via `SetRuntimeArgs()` or callbacks
- Examples: `buffer->address()`, `start_tile_id`, `batch_size`
- Limit: 341 args per kernel (4-byte each)

**Callbacks (`override_runtime_args_callback`):**
- Lambda function that updates runtime args before each program execution
- Receives: operation ptr, program ref, input/output tensors
- Use case: buffer addresses change when tensors reallocated
- Pattern in matmul: updates addresses at indices 0 (in0), 8 (in1), writer index 0 (out)
- Called automatically by framework before `EnqueueProgram()`
- Essential for ops that reuse programs across different tensor instances
- Example from matmul program factory:
  ```cpp
  auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, ...](
      const void* operation,
      const tt::tt_metal::Program& program,
      const std::vector<ttnn::Tensor>& input_tensors,
      const std::vector<std::optional<const ttnn::Tensor>>&,
      const std::vector<ttnn::Tensor>& output_tensors) {
      auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
      runtime_args[0] = input_tensors.at(0).buffer()->address();
      runtime_args[8] = input_tensors.at(1).buffer()->address();
  };
  ```

**CoreRangeSet:**
- Defines which cores execute each kernel
- Created from grid dimensions: `num_cores_to_corerangeset(num_blocks, grid_size, row_major=true)`
- Each core gets unique runtime args via `SetRuntimeArgs(program, kernel, core, args)`
- Examples: all cores `CoreRangeSet({0, 0}, {num_cores_x-1, num_cores_y-1})`, specific range

**CircularBufferConfig:**
- Specifies CB size, data format, page size per CB index
- Multi-CB config: `{{cb_id, data_format}, ...}` for multiple CBs with same size
- Page size: `set_page_size(cb_id, tile_size)` - size of one "page" (tile) in CB
- Shared buffers: output+intermediate can share same physical memory with different indices

### Reader-Writer-Compute Synchronization

Understanding how the three kernel types synchronize via circular buffers:

```
Reader (RISC-V)          Compute (Tensix)           Writer (RISC-V)
───────────────          ────────────────           ───────────────
cb_reserve_back(c_0)
noc_async_read → c_0
noc_async_read_barrier
cb_push_back(c_0)    →   cb_wait_front(c_0)
                         acquire_dst()
                         matmul_tiles(...)
                         cb_reserve_back(c_16)
                         pack_tile → c_16
                         cb_push_back(c_16)    →    cb_wait_front(c_16)
                         release_dst()              noc_async_write from c_16
                         cb_pop_front(c_0)          noc_async_write_barrier
                                                    cb_pop_front(c_16)
```

**Key synchronization points:**
1. Reader fills CB → Compute waits for data
2. Compute fills output CB → Writer waits for results
3. All kernels run concurrently on different processors
4. CBs implement blocking producer-consumer queues (automatic backpressure)
5. Double buffering enables overlap: Reader fetches block N+1 while Compute processes block N

### Advanced Patterns

**Multi-tensor accessors (concat pattern):**
```cpp
// Setup multiple tensor accessors in one kernel
constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
constexpr auto tensor_accessor_args =
    make_tensor_accessor_args_tuple<num_tensors, page_size_base_idx + num_tensors>();

auto tensor_accessors_tuple =
    make_tensor_accessor_tuple(tensor_accessor_args, src_addr_base_idx, page_size_base_idx);
auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

// Use in loop: switch between tensors dynamically
auto read_addr = abstract_tensor_accessor_wrappers[curr_tensor].get_noc_addr(tile_id);
```

**Tile registers management (compute kernels):**
- `tile_regs_acquire()`: Acquire tile registers for computation
- `tile_regs_commit()`: Mark computation complete
- `tile_regs_wait()`: Wait for computation to finish
- `tile_regs_release()`: Release registers for next operation
- Pattern: acquire → compute → commit → wait → pack → release

**Data format reconfiguration:**
- `reconfig_data_format_srca(old_cb, new_cb)`: Change input data format for compute
- `pack_reconfig_data_format(old_cb, new_cb)`: Change output data format for packing
- Use case: switching between different precision formats (FP16, BF16, etc.)
- Critical for mixed-precision operations

**Tilize/Untilize operations:**
- `tilize_init()`, `tilize_block()`, `tilize_uninit()`: RM → tiled conversion
- `untilize_init()`, `untilize_block()`, `untilize_uninit()`: Tiled → RM conversion
- Use case: interfacing with ops that require different layouts
- Example: attention matmul processes row-by-row (RM) but inputs are tiled

**Sharded input handling:**
- Compile-time defines: `#ifdef IN0_SHARDED`, `#ifdef IN1_SHARDED`
- For sharded inputs: `cb_reserve_back()` + `cb_push_back()` without reading (data already in L1)
- For non-sharded: standard NOC read pattern
- Allows same kernel to handle both sharded and interleaved inputs

**Parameter constraints (matmul example):**
- `Mt % per_core_M == 0` - output rows evenly divisible across cores
- `Nt % per_core_N == 0` - output cols evenly divisible across cores
- `Kt % in0_block_w == 0` - K dimension evenly divisible into blocks
- `per_core_M % out_subblock_h == 0` - block rows divisible into subblocks
- `per_core_N % out_subblock_w == 0` - block cols divisible into subblocks
- Always validate dimensions in program factory before kernel dispatch

### Best Practices

1. **Always barrier after async NOC ops**: `noc_async_read_barrier()` / `noc_async_write_barrier()`
2. **Use compile-time args for constants**: Enables compiler optimizations, reduces L1 usage
3. **Double-buffer input CBs**: Size = 2× block size for compute-movement overlap
4. **Match tile strides to tensor layout**: Calculate 2D strides correctly for block traversal
5. **Acquire/release DST registers**: Always pair `acquire_dst()` with `release_dst()`
6. **Reserve before get_ptr**: Always `cb_reserve_back()` before `get_write_ptr()`
7. **Pop after processing**: Match every `cb_wait_front()` with `cb_pop_front()`
8. **Validate dimensions**: Check alignment constraints in program factory
9. **Use tile_regs API for complex compute**: Explicit control over register allocation timing
10. **CB indices are linear offsets**: Not tensor coordinates - calculate manually based on block layout

### Debugging

#### Kernel Cache

**CRITICAL**: Clear cache after kernel changes:
```bash
rm -rf ~/.cache/tt-metal-cache/
```

**Symptoms of stale cache:**
- Kernel modifications don't change behavior
- Tests unchanged after code edits

#### Memory Management

**Never use stack arrays for NOC operations:**
```cpp
// ❌ WRONG
uint32_t data[8];  // Stack array - UNRELIABLE
noc_async_read(src_addr, (uint32_t)data, size);

// ✅ CORRECT - Use L1 circular buffer
cb_reserve_back(cb_id, 1);
uint32_t l1_addr = get_write_ptr(cb_id);
noc_async_read(src_addr, l1_addr, size);
uint32_t* data = reinterpret_cast<uint32_t*>(l1_addr);
cb_push_back(cb_id, 1);
```

#### Interleaved Buffer Writes

**Pattern: Read → Modify in L1 → Write full row:**
```cpp
// Reserve L1 for row
cb_reserve_back(cb_id, 1);
uint32_t l1_row_addr = get_write_ptr(cb_id);

// Initialize and modify in L1
uint16_t* row = reinterpret_cast<uint16_t*>(l1_row_addr);
for (uint32_t i = 0; i < row_size; i++) row[i] = 0;
row[expert_idx] = weight;  // Scatter values

// Write complete row to DRAM
uint64_t row_noc_addr = get_noc_addr(row_idx, tensor_accessor);
noc_async_write(l1_row_addr, row_noc_addr, row_bytes);
noc_async_write_barrier();
```

#### TensorAccessor for ROW_MAJOR

```cpp
// Page size = row size in bytes (NOT tile size)
const auto accessor = TensorAccessor(accessor_args, buffer_addr, row_size_bytes);
uint64_t row_addr = get_noc_addr(N, accessor);  // Get row N
```

### Build System

#### Directory Structure

Build directory naming convention:
- Standard build with Tracy: `build_<BuildType>_tracy` (e.g., `build_Release_tracy`)
- Symlink: `build` → points to `build_Release_tracy` (or current build directory)

```bash
# Standard directory: build_Release_tracy/
build_Release_tracy/
├── lib/
│   ├── _ttnn.so          # Main TTNN library
│   └── _ttnncpp.so       # TTNN C++ library
├── ttnn/
│   └── _ttnncpp.so       # Intermediate build artifact
└── bin/                   # Executables

# Python package location (must copy here after build):
ttnn/ttnn/_ttnn.so
```

#### Build Workflow

**IMPORTANT: Kernel-only changes do NOT require rebuild**

Kernels are compiled at runtime, not build time. If you only modify kernel files (`.cpp` files in `kernels/` directories), you do NOT need to rebuild tt-metal. Just clear the kernel cache:

```bash
# For kernel-only changes (e.g., reader_*.cpp, writer_*.cpp, compute_*.cpp)
rm -rf ~/.cache/tt-metal-cache/

# Then run your test directly - kernels will be compiled on first run
timeout 30 conda run -n tt-base pytest tests/test_moe.py -v
```

**When rebuild IS required** (C++ operation code changes):

```bash
# 1. Incremental build (PREFERRED - fast, use this for most C++ changes)
conda run -n tt-base cmake --build build -j16

# 2. Verify the build completed successfully
ls -lh ttnn/ttnn/_ttnn.so
```

**When to use full rebuild** (only if incremental build fails or for configuration changes):

```bash
# Full rebuild (slower, only when necessary)
conda run -n tt-base ./build_metal.sh -p
```

**Summary:**
- **Kernel changes** (`kernels/*.cpp`): Clear cache only → `rm -rf ~/.cache/tt-metal-cache/`
- **Operation changes** (`.hpp`, `.cpp`, `*_op.*`): Incremental build → `cmake --build build -j16`
- **Program factory changes** (`*_program_factory.*`): Incremental build → `cmake --build build -j16`
- **Python binding changes** (`*_pybind.*`): Incremental build → `cmake --build build -j16`
- **Build configuration changes**: Full rebuild → `./build_metal.sh -p`
- **Always use tt-base conda environment** for building

#### Common Issues

1. **"Module not found"**: Library not copied → `cp build/lib/_ttnn.so ttnn/ttnn/_ttnn.so`
2. **Changes ignored**: File not detected → `touch` source file before build
3. **Kernel changes ignored**: Clear cache → `rm -rf ~/.cache/tt-metal-cache/`
4. **Build directory confusion**: Use symlink `build` or check actual directory with `ls -l build`

#### Symlinks

`/home/jinpyo/tt-metal-releases/tt-metal` → `tt-metal-mcrl` (same location, different paths in errors)

## Additional Resources

- [METALIUM_GUIDE.md](METALIUM_GUIDE.md): Architecture details
- [CONTRIBUTING.md](CONTRIBUTING.md): Git workflow, debugging, testing
- [README.md](README.md): Model demos, tech reports, installation
