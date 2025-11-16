#pragma once

constexpr uint32_t WH_LOGICAL_TO_VIRTUALL_OFFSET = 18;
constexpr uint32_t GRID_SIZE = 8;

void logical_to_virtual(uint32_t logical_x, uint32_t logical_y,
    uint32_t *virtual_x, uint32_t *virtual_y) {
    *virtual_x = logical_x + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    *virtual_y = logical_y + WH_LOGICAL_TO_VIRTUALL_OFFSET;
}

void virtual_to_logical(uint32_t virtual_x, uint32_t virtual_y,
    uint32_t *logical_x, uint32_t *logical_y) {
    *logical_x = virtual_x - WH_LOGICAL_TO_VIRTUALL_OFFSET;
    *logical_y = virtual_y - WH_LOGICAL_TO_VIRTUALL_OFFSET;
}

void get_logical_and_virtual(uint32_t *logical_x, uint32_t *logical_y,
         uint32_t *virtual_x, uint32_t *virtual_y) {
    *virtual_x = my_x[noc_index];
    *virtual_y = my_y[noc_index];
    virtual_to_logical(*virtual_x, *virtual_y, logical_x, logical_y);
}

void get_virtual_coord(uint32_t *virtual_x, uint32_t *virtual_y) {
    *virtual_x = my_x[noc_index];
    *virtual_y = my_y[noc_index];
}


// Generic broadcast function for rectangular (including 1D) multicast
FORCE_INLINE void broadcast(uint32_t l1_addr, uint32_t nbytes,
                            uint32_t sender_x, uint32_t sender_y, uint32_t x0,
                            uint32_t x1, uint32_t y0, uint32_t y1,
                            uint32_t sender_sem_id, uint32_t receiver_sem_id,
                            uint8_t noc) {
    // Assert valid coordinate ranges
    ASSERT(x0 <= x1);
    ASSERT(y0 <= y1);

    // Translate semaphore IDs to addresses and pointers
    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t *sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t *receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(receiver_sem_addr);

    // Get current core's logical coordinates
    uint32_t x = (uint32_t)my_x[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t y = (uint32_t)my_y[noc] - WH_LOGICAL_TO_VIRTUALL_OFFSET;

    // Convert logical coordinates to virtual
    uint32_t sender_virtual_x = sender_x + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t sender_virtual_y = sender_y + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t x0_virtual = x0 + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t x1_virtual = x1 + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t y0_virtual = y0 + WH_LOGICAL_TO_VIRTUALL_OFFSET;
    uint32_t y1_virtual = y1 + WH_LOGICAL_TO_VIRTUALL_OFFSET;

    // Check if sender is in destination range
    bool sender_in_range =
        (sender_x >= x0 && sender_x <= x1 && sender_y >= y0 && sender_y <= y1);

    uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);

    if (x == sender_x && y == sender_y) {
        // SENDER: Wait for all receivers to signal ready
        noc_semaphore_wait(sender_sem_ptr,
                           sender_in_range ? num_dests - 1 : num_dests);
        noc_semaphore_set(sender_sem_ptr, 0);

        // Multicast data
        uint64_t mcast_data_addr = get_noc_multicast_addr(
            x0_virtual, y0_virtual, x1_virtual, y1_virtual, l1_addr, noc);
        if (sender_in_range) {
            noc_async_write_multicast_loopback_src(
                l1_addr, mcast_data_addr, nbytes, num_dests, false, noc);
        } else {
            noc_async_write_multicast(l1_addr, mcast_data_addr, nbytes,
                                      num_dests, false, noc);
        }

        // Multicast VALID signal to all receivers
        *receiver_sem_ptr = VALID;
        uint64_t mcast_signal_addr =
            get_noc_multicast_addr(x0_virtual, y0_virtual, x1_virtual,
                                   y1_virtual, receiver_sem_addr, noc);
        if (sender_in_range) {
            noc_semaphore_set_multicast_loopback_src(
                receiver_sem_addr, mcast_signal_addr, num_dests, false, noc);
        } else {
            noc_semaphore_set_multicast(receiver_sem_addr, mcast_signal_addr,
                                        num_dests, false, noc);
        }
        noc_async_write_barrier();
    } else if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        // RECEIVER: This core is in destination range
        noc_semaphore_set(receiver_sem_ptr, INVALID);

        // Signal sender that we're ready
        uint64_t sender_sem_noc_addr = get_noc_addr(
            sender_virtual_x, sender_virtual_y, sender_sem_addr, noc);
        noc_semaphore_inc(sender_sem_noc_addr, 1, noc);

        // Wait for sender to signal data is ready
        noc_semaphore_wait(receiver_sem_ptr, VALID);
    }
}
