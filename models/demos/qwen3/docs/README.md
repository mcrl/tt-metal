# Qwen3 MoE Documentation

This directory contains technical documentation for the Qwen3 Mixture of Experts (MoE) implementation and comprehensive TT-Metal framework documentation.

## Documentation Files

### TT-Metal Framework Documentation
- **[01_tenstorrent_chip_architecture.md](01_tenstorrent_chip_architecture.md)** - Chip architecture, coordinate systems, harvested cores
- **[02_tenstorrent_tensix_core.md](02_tenstorrent_tensix_core.md)** - Tensix core architecture, 5 RISC-V cores, L1 memory
- **[03_host_programming_apis.md](03_host_programming_apis.md)** - Host APIs, device management, programs, buffers
- **[04_kernel_programming.md](04_kernel_programming.md)** - Kernel programming, NOC operations, compute APIs
- **[05_inter_core_communication.md](05_inter_core_communication.md)** - Semaphores, multicast, synchronization patterns

### MoE Implementation Documentation
- **[MoE_implementation_plan.md](MoE_implementation_plan.md)** - Original implementation planning document

## Quick Links

### Source Code
- MoE TT-NN Implementation: `../tt/moe.py`
- MoE Tests: `../tests/test_moe*.py`
- TTNN Operations: `/ttnn/cpp/ttnn/operations/experimental/moe/`

### Key Operations

#### 1. `ttnn.prepare_moe_routing_tensors()`
Creates efficient routing tensors for expert-parallel computation.

## Status Summary

| Component | Status | Test Coverage |
|-----------|--------|---------------|
| prepare_moe_routing_tensors | ✅ Complete | 60 tests passing |
| pytest fixtures | ✅ Optimized | Module-scoped caching |

## Running Tests

```bash
# Run all MoE tests
cd models/demos/qwen3
pytest tests/test_moe*.py -v

# Run specific operation tests
pytest tests/test_moe_mapping.py -v
pytest tests/test_moe_routing_tensors.py -v
```