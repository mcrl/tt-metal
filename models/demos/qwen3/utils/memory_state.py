import ttnn


def get_memory_state(device):
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)

    used_per_bank = mv.total_bytes_allocated_per_bank
    total_per_bank = mv.total_bytes_per_bank
    free_per_bank = mv.total_bytes_free_per_bank
    banks = mv.num_banks

    used_total = used_per_bank * banks
    total = total_per_bank * banks
    free_total = free_per_bank * banks

    return used_total, free_total, total


def print_memory_state(mesh_device):
    used_total, free_total, total = get_memory_state(mesh_device)
    print(f"DRAM used: {used_total / (1024**3):.2f} / {total / (1024**3):.2f} GB ({used_total / total:.2%})")
