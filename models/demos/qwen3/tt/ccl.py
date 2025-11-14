import ttnn
import torch
import os

class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.shape[0] * mesh_device.shape[1]

        if mesh_device.shape[0] == 2 or mesh_device.shape[1] == 2:
            self.topology = ttnn.Topology.Linear
        else:
            self.topology = ttnn.Topology.Ring

        self.is_galaxy = self.num_devices == 32

        if self.is_galaxy:
            self.sub_device_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
        else:
            grid = mesh_device.compute_with_storage_grid_size()
            num_cores = grid.x * grid.y
            self.sub_device_crs = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        self.num_cbs = 2
        self.from_remote_semaphore_handles = []
        self.to_remote_semaphore_handles = []

        self.gather_semaphore_handles = [[], []]
        self.barrier_semaphore_handles = [[], []]

        self.from_semaphore_handles = [[], []]
        self.to_semaphore_handles = [[], []]
        self.reduce_semaphore_handles = [[], []]
    
        for i in range(2):
            for _ in range(self.num_cbs):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
                self.gather_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                )
                self.reduce_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                )
        
        self.gather_idx = [0, 0]
        self.barrier_semaphore_idx = [0, 0]

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis):
        semaphore_index = cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % self.num_cbs
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def reduce_scatter(
        self,
        input_tensor_mesh,
        dim,
        cluster_axis,
        persistent_output_buffers=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        num_links = 4 if self.is_galaxy else 1
        ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor=input_tensor_mesh,
            persistent_output_buffers=persistent_output_buffers,
            dim=dim,
            multi_device_global_semaphore=self.reduce_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_links,
            memory_config=memory_config,
            topology=self.topology,
            cluster_axis=cluster_axis,
            num_workers_per_link=1,
        )

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def all_gather(
        self, 
        input_tensor_mesh,
        dim, 
        cluster_axis, 
        persistent_output_buffer=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, 
    ):
        num_links = 4 if self.is_galaxy else 1
        ttnn_tensor_out = ttnn.experimental.all_gather_async(
            input_tensor=input_tensor_mesh,
            persistent_output_buffer=persistent_output_buffer,
            dim=dim,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            memory_config=memory_config,
            topology=self.topology,
            cluster_axis=cluster_axis,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out