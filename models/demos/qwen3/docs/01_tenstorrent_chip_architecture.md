# Tenstorrent Hardware: Chip Architecture

## Overview

TT-Metal provides a sophisticated multi-layer coordinate system to manage Tenstorrent AI accelerator chips. The framework abstracts hardware complexity through logical coordinates while maintaining efficient access to physical resources including compute cores, DRAM controllers, and Ethernet interfaces.

## 1. Coordinate Systems

TT-Metal uses multiple coordinate systems to represent core locations on the chip, defined in [`tt_metal/third_party/umd/device/api/umd/device/tt_core_coordinates.h`](../../../../../../tt_metal/third_party/umd/device/api/umd/device/tt_core_coordinates.h):

### CoordSystem Enum

```cpp
enum class CoordSystem : std::uint8_t {
    LOGICAL,      // User-facing, harvesting-aware coordinates (0,0) to (grid_x-1, grid_y-1)
    NOC0,         // Physical NOC0 routing coordinates
    VIRTUAL,      // Virtual coordinates used by UMD APIs (hides harvesting)
    TRANSLATED,   // Translation-table coordinates (architecture-specific)
    NOC1,         // Physical NOC1 routing coordinates
};
```

### Coordinate System Details

#### LOGICAL Coordinates
- **Purpose**: User-facing coordinate system for kernel placement and high-level operations
- **Characteristics**:
  - Dense grid with no gaps (harvested cores removed)
  - Always starts at (0,0) and extends to (grid_size_x-1, grid_size_y-1)
  - Used in all high-level APIs and kernel dispatch
- **Example**: On an 8×8 grid with 2 harvested rows, logical grid is 6×8

#### NOC0/NOC1 Coordinates (Physical)
- **Purpose**: Actual physical routing coordinates on the Network-on-Chip
- **Characteristics**:
  - May contain gaps due to harvested cores
  - NOC0 and NOC1 have different routing topologies on the same chip
  - Direct hardware addressing for data movement
- **Usage**: Internal framework operations, low-level NOC programming

#### VIRTUAL Coordinates
- **Purpose**: Interface between high-level APIs and UMD (Unified Metal Driver)
- **Characteristics**:
  - Provides consistent view abstracting harvesting details
  - Main interface for host-device communication
  - Used internally by the driver layer

#### TRANSLATED Coordinates
- **Purpose**: Architecture-specific coordinate mapping
- **Characteristics**:
  - Uses translation tables on newer architectures (Wormhole, Blackhole)
  - Enables dynamic harvesting remapping without firmware recompilation
  - Optimized for specific chip architectures

## 2. Harvested Cores Representation

### What is Harvesting?

Harvesting is the process of disabling defective cores to improve chip yield. TT-Metal transparently handles harvested cores, presenting users with a consistent logical view while managing the physical gaps internally.

### Harvesting Mask Implementation

From [`tt_metal/llrt/core_descriptor.cpp`](../../../../../../tt_metal/llrt/core_descriptor.cpp):

```cpp
// Get harvesting mask for a device
uint32_t harvesting_mask = tt::tt_metal::MetalContext::instance()
    .get_cluster()
    .get_harvesting_mask(device_id);

// Count harvested rows/columns
std::bitset<32> mask_bitset(harvesting_mask);
uint32_t num_harvested_on_axis = mask_bitset.count();
```

**Key Properties**:
- 32-bit bitmask where each bit represents a row or column
- Bit 0 corresponds to first row/column in NOC0 coordinates
- Maximum 2 rows/columns can be harvested (firmware limitation)
- Mask is reported in physical layout, then converted to NOC0 layout

### Product Classification by Harvesting

Different harvesting levels result in different product SKUs:

```cpp
inline const std::string& get_product_name(tt::ARCH arch, uint32_t num_harvested_on_axis) {
    const static std::map<tt::ARCH, std::map<uint32_t, std::string>> product_name = {
        {tt::ARCH::GRAYSKULL, {{0, "E150"}, {2, "E75"}}},
        {tt::ARCH::WORMHOLE_B0, {{0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"}}},
        {tt::ARCH::BLACKHOLE, {{0, "unharvested"}, {1, "1xharvested"}, {2, "2xharvested"}}}
    };
    return product_name.at(arch).at(num_harvested_on_axis);
}
```

## 3. Accessing the 2D Torus of Tensix Cores

### Core Grid Access APIs

The Device class provides convenient methods to access the compute grid:

```cpp
// Get the full compute grid size (including storage cores)
CoreCoord grid_size = device->compute_with_storage_grid_size();
// Returns: e.g., (8, 8) for an 8×8 grid

// Get logical grid size (compute cores only)
CoreCoord logical_size = device->logical_grid_size();

// Get all worker cores as a CoreRangeSet
const CoreRangeSet& worker_cores = device->worker_cores(
    HalProgrammableCoreType::TENSIX,
    sub_device_id
);

// Get number of worker cores
uint32_t num_workers = device->num_worker_cores(
    HalProgrammableCoreType::TENSIX,
    sub_device_id
);
```

### Iterating Through the Grid

Example from [`tt_metal/impl/device/device.cpp`](../../../../../../tt_metal/impl/device/device.cpp):

```cpp
// Iterate through all logical worker cores
for (uint32_t y = 0; y < logical_grid_size().y; y++) {
    for (uint32_t x = 0; x < logical_grid_size().x; x++) {
        CoreCoord logical_core(x, y);

        // Skip storage-only cores if needed
        if (!storage_only_cores_set.count(logical_core)) {
            // Process worker core
            process_worker_core(logical_core);
        }
    }
}

// Using CoreRangeSet iteration
for (const CoreCoord& core : worker_cores) {
    // Process each worker core
    dispatch_kernel_to_core(core);
}
```

### CoreRange and CoreRangeSet

For efficient specification of multiple cores:

```cpp
// Single rectangular region
CoreRange single_row(
    CoreCoord(0, 0),     // start
    CoreCoord(7, 0)      // end
);

// Multiple regions
std::vector<CoreRange> ranges = {
    CoreRange({0, 0}, {3, 3}),  // First 4×4 block
    CoreRange({4, 4}, {7, 7})   // Second 4×4 block
};
CoreRangeSet multi_region(ranges);

// Check if core is in set
if (multi_region.contains(CoreCoord(2, 2))) {
    // Core is in the set
}

// Get total number of cores
uint32_t total_cores = multi_region.num_cores();
```

## 4. Controlling Different Core Types Together

### Core Types Definition

From [`tt_metal/third_party/umd/device/api/umd/device/tt_core_coordinates.h`](../../../../../../tt_metal/third_party/umd/device/api/umd/device/tt_core_coordinates.h):

```cpp
enum class CoreType {
    TENSIX,        // Compute cores (main worker cores)
    DRAM,          // DRAM controller cores
    ACTIVE_ETH,    // Active ethernet cores (available for use)
    IDLE_ETH,      // Idle ethernet cores
    PCIE,          // PCIe interface cores
    ARC,           // ARC processor cores
    ROUTER_ONLY,   // Routing-only cores
    HARVESTED,     // Harvested/disabled cores
    // ... other types
};
```

### Accessing Different Core Types

```cpp
// Worker/Compute cores (TENSIX)
const CoreRangeSet& workers = device->worker_cores(
    HalProgrammableCoreType::TENSIX, sub_device_id);

// Ethernet cores
std::unordered_set<CoreCoord> active_eth =
    device->get_active_ethernet_cores(skip_reserved_tunnel_cores = true);
std::unordered_set<CoreCoord> inactive_eth =
    device->get_inactive_ethernet_cores();

// DRAM cores
for (uint32_t channel = 0; channel < device->num_dram_channels(); channel++) {
    CoreCoord dram_core = device->logical_core_from_dram_channel(channel);
    // Use DRAM core...
}

// Storage-only cores (cores with only L1 SRAM, no compute)
const std::unordered_set<CoreCoord>& storage_cores =
    device->storage_only_cores();
```

### Unified Control Example

Example showing how to initialize all core types together:

```cpp
void initialize_all_cores(Device* device) {
    // Reset all worker cores
    for (const CoreCoord& core : device->worker_cores()) {
        CoreCoord virtual_core = device->worker_core_from_logical_core(core);
        reset_core(device->id(), virtual_core);
    }

    // Reset all active ethernet cores
    for (const CoreCoord& core : device->get_active_ethernet_cores()) {
        CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
        reset_core(device->id(), virtual_core);
    }

    // Initialize DRAM cores
    for (uint32_t channel = 0; channel < device->num_dram_channels(); channel++) {
        CoreCoord dram_logical = device->logical_core_from_dram_channel(channel);
        CoreCoord dram_virtual = device->virtual_core_from_logical_core(
            dram_logical, CoreType::DRAM);
        initialize_dram_core(device->id(), dram_virtual);
    }
}
```

## 5. Logical vs Physical View APIs

### Coordinate Translation APIs

The Device class provides translation methods between coordinate systems:

```cpp
class IDevice {
    // Logical → Virtual translation for any core type
    virtual CoreCoord virtual_core_from_logical_core(
        const CoreCoord& logical_coord,
        const CoreType& core_type) const;

    // Specialized translations for common core types
    virtual CoreCoord worker_core_from_logical_core(
        const CoreCoord& logical_core) const;

    virtual CoreCoord ethernet_core_from_logical_core(
        const CoreCoord& logical_core) const;

    // Reverse translation: Virtual Ethernet → Logical
    virtual CoreCoord logical_core_from_ethernet_core(
        const CoreCoord& ethernet_core) const;

    // Batch conversion for efficiency
    virtual std::vector<CoreCoord> worker_cores_from_logical_cores(
        const std::vector<CoreCoord>& logical_cores) const;
};
```

### Translation Implementation Details

From [`tt_metal/impl/device/device.cpp`](../../../../../../tt_metal/impl/device/device.cpp):

```cpp
CoreCoord Device::virtual_core_from_logical_core(
    const CoreCoord& logical_coord,
    const CoreType& core_type) const {

    // Delegate to cluster manager for translation
    return tt::tt_metal::MetalContext::instance().get_cluster()
        .get_virtual_coordinate_from_logical_coordinates(
            this->id_, logical_coord, core_type);
}

// Worker cores use TENSIX type
CoreCoord Device::worker_core_from_logical_core(
    const CoreCoord& logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}
```

### Coordinate Flow in Practice

```
User Code (LOGICAL coordinates)
    ↓ Device::worker_core_from_logical_core()
Virtual Coordinates (for UMD)
    ↓ Cluster/UMD translation
Physical NOC Coordinates (hardware routing)
```

### Example: Writing to Cores Using Different Views

```cpp
// User provides logical coordinate
CoreCoord logical_worker(3, 4);

// Step 1: Convert to virtual for driver access
CoreCoord virtual_worker = device->worker_core_from_logical_core(logical_worker);

// Step 2: Use virtual coordinate for hardware operations
uint32_t data = 0x1234;
tt::tt_metal::MetalContext::instance().get_cluster().write_core(
    &data, sizeof(uint32_t),
    tt_cxy_pair(device->id(), virtual_worker),  // chip + virtual coord
    l1_address
);

// For debugging: Get physical coordinate (not recommended for normal use)
CoreCoord physical = cluster->get_physical_coordinate_from_logical_coordinates(
    device->id(), logical_worker, CoreType::WORKER,
    no_warn = false  // Will log warning
);
```

## 6. Practical Examples

### Example 1: Distribute Work Across Available Cores

```cpp
void distribute_workload(Device* device, uint32_t total_work_items) {
    // Get available worker cores
    const CoreRangeSet& workers = device->worker_cores();
    uint32_t num_cores = workers.num_cores();

    // Calculate work per core
    uint32_t items_per_core = total_work_items / num_cores;
    uint32_t remainder = total_work_items % num_cores;

    uint32_t item_offset = 0;
    uint32_t core_index = 0;

    for (const CoreCoord& logical_core : workers) {
        uint32_t items_for_this_core = items_per_core;
        if (core_index < remainder) {
            items_for_this_core++;
        }

        // Dispatch work to core
        dispatch_kernel(device, logical_core, item_offset, items_for_this_core);

        item_offset += items_for_this_core;
        core_index++;
    }
}
```

### Example 2: Create Sub-device with Mixed Core Types

```cpp
SubDevice create_mixed_subdevice(Device* device) {
    // Use half the compute cores
    const auto& grid = device->compute_with_storage_grid_size();
    CoreRangeSet half_workers(
        CoreRange({0, 0}, {grid.x/2 - 1, grid.y - 1})
    );

    // Use all active ethernet cores
    const auto& eth_cores = device->get_active_ethernet_cores();
    std::vector<CoreRange> eth_ranges;
    for (const auto& core : eth_cores) {
        eth_ranges.emplace_back(core, core);
    }
    CoreRangeSet eth_set(eth_ranges);

    // Create sub-device with both types
    return SubDevice(std::array{half_workers, eth_set});
}
```

### Example 3: Handle Harvesting Transparently

```cpp
void process_grid_with_harvesting(Device* device) {
    // User works with logical coordinates - harvesting is transparent
    CoreCoord logical_grid = device->logical_grid_size();

    std::cout << "Processing " << logical_grid.x << "×" << logical_grid.y
              << " logical grid\n";

    // Framework handles harvested cores automatically
    for (uint32_t y = 0; y < logical_grid.y; y++) {
        for (uint32_t x = 0; x < logical_grid.x; x++) {
            CoreCoord logical(x, y);

            // This works regardless of harvesting
            CoreCoord virtual_coord = device->worker_core_from_logical_core(logical);

            // Virtual coord is valid for hardware access
            program_core(device->id(), virtual_coord);
        }
    }

    // Check actual harvesting status
    uint32_t harvesting_mask = MetalContext::instance().get_cluster()
        .get_harvesting_mask(device->id());
    std::cout << "Harvesting mask: 0x" << std::hex << harvesting_mask << "\n";
}
```

## Key Takeaways

1. **Always use LOGICAL coordinates** in application code for portability and simplicity
2. **The framework handles harvesting transparently** - logical coordinates automatically skip harvested cores
3. **Different core types (TENSIX, DRAM, ETH) can be controlled together** through unified Device APIs
4. **Physical coordinates are internal** - avoid using them unless implementing low-level device drivers
5. **CoreRangeSet is the preferred way** to specify groups of cores for operations
6. **Translation between coordinate systems is automatic** when using Device class methods

## References

- Core coordinate systems: [`tt_core_coordinates.h`](../../../../../../tt_metal/third_party/umd/device/api/umd/device/tt_core_coordinates.h)
- Device interface: [`device.hpp`](../../../../../../tt_metal/api/tt-metalium/device.hpp)
- Device implementation: [`device.cpp`](../../../../../../tt_metal/impl/device/device.cpp)
- Coordinate manager: [`coordinate_manager.h`](../../../../../../tt_metal/third_party/umd/device/api/umd/device/coordinate_manager.h)
- Core descriptor: [`core_descriptor.hpp`](../../../../../../tt_metal/api/tt-metalium/core_descriptor.hpp)
- Cluster management: [`tt_cluster.cpp`](../../../../../../tt_metal/llrt/tt_cluster.cpp)