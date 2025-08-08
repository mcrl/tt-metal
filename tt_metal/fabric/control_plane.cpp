// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <enchantum/enchantum.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <tt_stl/assert.hpp>

#include "control_plane.hpp"
#include "core_coord.hpp"
#include "compressed_routing_table.hpp"
#include "compressed_routing_path.hpp"
#include "hostdevcommon/fabric_common.h"
#include "distributed_context.hpp"
#include "fabric_types.hpp"
#include "hal_types.hpp"
#include "host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/common/env_lib.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_coord.hpp"
#include "mesh_graph.hpp"
#include "metal_soc_descriptor.h"
#include "routing_table_generator.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/cluster.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"
#include "tt_stl/small_vector.hpp"

namespace tt::tt_fabric {

namespace {

// Get the physical chip ids for a mesh
std::unordered_map<ChipId, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(ChipId chip_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

template <typename CONNECTIVITY_MAP_T>
void build_golden_link_counts(
    CONNECTIVITY_MAP_T const& golden_connectivity_map,
    std::unordered_map<MeshId, std::unordered_map<ChipId, std::unordered_map<RoutingDirection, size_t>>>&
        golden_link_counts_out) {
    static_assert(
        std::is_same_v<CONNECTIVITY_MAP_T, IntraMeshConnectivity> ||
            std::is_same_v<CONNECTIVITY_MAP_T, InterMeshConnectivity>,
        "Invalid connectivity map type");
    for (std::uint32_t mesh_id = 0; mesh_id < golden_connectivity_map.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < golden_connectivity_map[mesh_id].size(); chip_id++) {
            for (const auto& [remote_connected_id, router_edge] : golden_connectivity_map[mesh_id][chip_id]) {
                TT_FATAL(
                    golden_link_counts_out[MeshId{mesh_id}][chip_id][router_edge.port_direction] == 0,
                    "Golden link counts already set for chip {} in mesh {}",
                    chip_id,
                    mesh_id);
                golden_link_counts_out[MeshId{mesh_id}][chip_id][router_edge.port_direction] =
                    router_edge.connected_chip_ids.size();
            }
        }
    }
}

std::uint64_t encode_mesh_id_and_rank(MeshId mesh_id, MeshHostRankId host_rank) {
    return (static_cast<uint64_t>(mesh_id.get()) << 32) | static_cast<uint64_t>(host_rank.get());
}

std::pair<MeshId, MeshHostRankId> decode_mesh_id_and_rank(std::uint64_t encoded_value) {
    return {
        MeshId{static_cast<std::uint32_t>(encoded_value >> 32)},
        MeshHostRankId{static_cast<std::uint32_t>(encoded_value & 0xFFFFFFFF)}};
}

bool check_connection_requested(
    MeshId my_mesh_id,
    MeshId neighbor_mesh_id,
    const RequestedIntermeshConnections& requested_intermesh_connections,
    const RequestedIntermeshPorts& requested_intermesh_ports) {
    if (!requested_intermesh_ports.empty()) {
        return requested_intermesh_ports.find(*my_mesh_id) != requested_intermesh_ports.end() &&
               requested_intermesh_ports.at(*my_mesh_id).find(*neighbor_mesh_id) !=
                   requested_intermesh_ports.at(*my_mesh_id).end();
    } else {
        return requested_intermesh_connections.find(*my_mesh_id) != requested_intermesh_connections.end() &&
               requested_intermesh_connections.at(*my_mesh_id).find(*neighbor_mesh_id) !=
                   requested_intermesh_connections.at(*my_mesh_id).end();
    }
}

[[maybe_unused]] std::string create_port_tag(port_id_t port_id) {
    return std::string(enchantum::to_string(port_id.first)) + std::to_string(port_id.second);
}

}  // namespace

const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
    {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
    {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
};

uint16_t get_bus_id(tt::umd::Cluster& cluster, ChipId chip_id) {
    // Prefer cached value from cluster descriptor (available for silicon and our simulator/mock descriptors)
    auto cluster_desc = cluster.get_cluster_description();
    uint16_t bus_id = cluster_desc->get_bus_id(chip_id);
    return bus_id;
}

UbbId get_ubb_id(tt::umd::Cluster& cluster, ChipId chip_id) {
    auto cluster_desc = cluster.get_cluster_description();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster_desc->get_arch());
    const auto bus_id = get_bus_id(cluster, chip_id);
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = bus_id & 0x0F;
        return UbbId{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id};
    }
    return UbbId{0, 0};  // Invalid UBB ID if not found
}

void ControlPlane::initialize_dynamic_routing_plane_counts(
    const IntraMeshConnectivity& intra_mesh_connectivity,
    tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode) {
    if (fabric_config == tt_fabric::FabricConfig::CUSTOM || fabric_config == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->router_port_directions_to_num_routing_planes_map_.clear();

    auto topology = FabricContext::get_topology_from_config(fabric_config);

    // For TG need to skip the direction on the remote devices directly connected to the MMIO devices as we have only
    // one outgoing eth chan to the mmio device
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    auto skip_direction = [&](const FabricNodeId& node_id, const RoutingDirection direction) -> bool {
        const auto& neighbors = this->get_chip_neighbors(node_id, direction);
        if (neighbors.empty()) {
            return false;
        }

        // The remote devices connected directly to the mmio will have both intra-mesh and inter-mesh neighbors
        if (neighbors.size() > 1 || neighbors.begin()->first != node_id.mesh_id) {
            return true;
        }

        return false;
    };

    auto apply_min =
        [&](FabricNodeId fabric_node_id,
            const std::unordered_map<tt::tt_fabric::RoutingDirection, std::vector<tt::tt_fabric::chan_id_t>>&
                port_direction_eth_chans,
            tt::tt_fabric::RoutingDirection direction,
            const std::unordered_map<tt::tt_fabric::RoutingDirection, size_t>& /*golden_link_counts*/,
            size_t& val) {
            if (skip_direction(fabric_node_id, direction)) {
                return;
            }
            if (auto it = port_direction_eth_chans.find(direction); it != port_direction_eth_chans.end()) {
                val = std::min(val, it->second.size());
            }
        };

    // For each mesh in the system
    auto user_meshes = this->get_user_physical_mesh_ids();
    if (reliability_mode == tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
        for (const auto& [fabric_node_id, directions_and_eth_chans] :
             this->router_port_directions_to_physical_eth_chan_map_) {
            for (const auto& [direction, eth_chans] : directions_and_eth_chans) {
                this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] = eth_chans.size();
            }
        }
    }

    std::unordered_map<MeshId, std::unordered_map<ChipId, std::unordered_map<RoutingDirection, size_t>>>
        golden_link_counts;
    TT_FATAL(
        this->routing_table_generator_ != nullptr && this->routing_table_generator_->mesh_graph != nullptr,
        "Routing table generator not initialized");
    build_golden_link_counts(
        this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity(), golden_link_counts);
    build_golden_link_counts(
        this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity(), golden_link_counts);

    auto apply_count = [&](FabricNodeId fabric_node_id, RoutingDirection direction, size_t count) {
        if (skip_direction(fabric_node_id, direction)) {
            return;
        }
        if (this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id) &&
            this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).contains(direction) &&
            !this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).at(direction).empty()) {
            this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] = count;
        }
    };

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    // For each mesh in the system
    for (auto mesh_id : user_meshes) {
        const auto& mesh_shape = this->get_physical_mesh_shape(MeshId{mesh_id});
        TT_FATAL(mesh_shape.dims() == 2, "ControlPlane: Only 2D meshes are supported");
        TT_FATAL(mesh_shape[0] > 0, "ControlPlane: Mesh width must be greater than 0");
        TT_FATAL(mesh_shape[1] > 0, "ControlPlane: Mesh height must be greater than 0");

        std::vector<size_t> row_min_planes(mesh_shape[0], std::numeric_limits<size_t>::max());
        std::vector<size_t> col_min_planes(mesh_shape[1], std::numeric_limits<size_t>::max());

        // First pass: Calculate minimums for each row/column
        size_t num_chips_in_mesh = intra_mesh_connectivity[mesh_id.get()].size();
        bool is_single_chip = num_chips_in_mesh == 1 && user_meshes.size() == 1;
        bool may_have_intra_mesh_connectivity = !is_single_chip;

        if (may_have_intra_mesh_connectivity) {
            const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
            for (const auto& mesh_coord : local_mesh_coord_range) {
                auto fabric_chip_id =
                    this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                auto mesh_coord_x = mesh_coord[0];
                auto mesh_coord_y = mesh_coord[1];

                const auto& port_directions = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);

                const auto& golden_counts = golden_link_counts.at(MeshId{mesh_id}).at(fabric_chip_id);
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::E,
                    golden_counts,
                    row_min_planes.at(mesh_coord_x));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::W,
                    golden_counts,
                    row_min_planes.at(mesh_coord_x));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::N,
                    golden_counts,
                    col_min_planes.at(mesh_coord_y));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::S,
                    golden_counts,
                    col_min_planes.at(mesh_coord_y));
            }

            // Collect row and column mins from all hosts in a BigMesh
            auto rows_min = *std::min_element(row_min_planes.begin(), row_min_planes.end());
            auto cols_min = *std::min_element(col_min_planes.begin(), col_min_planes.end());
            std::vector<size_t> rows_min_buf(*distributed_context.size());
            std::vector<size_t> cols_min_buf(*distributed_context.size());
            distributed_context.all_gather(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&rows_min), sizeof(size_t)),
                tt::stl::as_writable_bytes(tt::stl::Span<size_t>{rows_min_buf.data(), rows_min_buf.size()}));
            distributed_context.all_gather(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&cols_min), sizeof(size_t)),
                tt::stl::as_writable_bytes(tt::stl::Span<size_t>{cols_min_buf.data(), cols_min_buf.size()}));
            distributed_context.barrier();
            const auto global_rows_min = std::min_element(rows_min_buf.begin(), rows_min_buf.end());
            const auto global_cols_min = std::min_element(cols_min_buf.begin(), cols_min_buf.end());
            // TODO: specialize by topology for better perf
            if (topology == Topology::Mesh || topology == Topology::Torus) {
                auto global_mesh_min = std::min(*global_rows_min, *global_cols_min);
                std::fill(row_min_planes.begin(), row_min_planes.end(), global_mesh_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), global_mesh_min);
            } else {
                std::fill(row_min_planes.begin(), row_min_planes.end(), *global_rows_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), *global_cols_min);
            }

            // Second pass: Apply minimums to each device
            for (const auto& mesh_coord : local_mesh_coord_range) {
                auto fabric_chip_id =
                    this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                auto mesh_coord_x = mesh_coord[0];
                auto mesh_coord_y = mesh_coord[1];

                apply_count(fabric_node_id, RoutingDirection::E, row_min_planes.at(mesh_coord_x));
                apply_count(fabric_node_id, RoutingDirection::W, row_min_planes.at(mesh_coord_x));
                apply_count(fabric_node_id, RoutingDirection::N, col_min_planes.at(mesh_coord_y));
                apply_count(fabric_node_id, RoutingDirection::S, col_min_planes.at(mesh_coord_y));
            }
        }
    }
}

LocalMeshBinding ControlPlane::initialize_local_mesh_binding() {
    // When unset, assume host rank 0.
    const char* host_rank_str = std::getenv("TT_MESH_HOST_RANK");
    const MeshHostRankId host_rank =
        (host_rank_str == nullptr) ? MeshHostRankId{0} : MeshHostRankId{std::stoi(host_rank_str)};

    // If TT_MESH_ID is unset, assume this host is the only host in the system and owns all Meshes in
    // the MeshGraphDescriptor. Single Host Multi-Mesh is only used for testing purposes.
    if (mesh_id_str == nullptr && host_rank_str == nullptr) {
        auto& ctx = tt::tt_metal::MetalContext::instance().global_distributed_context();
        auto mpi_rank = *ctx.rank();
        std::vector<MeshId> local_mesh_ids;
        for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
            // TODO: #24528 - Move this to use TopologyMapper once Topology mapper works for multi-mesh systems
            const auto& host_ranks = this->routing_table_generator_->mesh_graph->get_host_ranks(mesh_id);
            TT_FATAL(
                host_ranks.size() == 1 && *host_ranks.values().front() == 0,
                "Mesh {} has {} host ranks, expected 1",
                *mesh_id,
                host_ranks.size());
            local_mesh_ids.push_back(mesh_id);
        }
        TT_FATAL(!local_mesh_ids.empty(), "No local meshes found.");
        return LocalMeshBinding{.mesh_ids = std::move(local_mesh_ids), .host_rank = MeshHostRankId{0}};
    }

    // Otherwise, use the value from the environment variable.
    auto local_mesh_binding = LocalMeshBinding{.mesh_ids = {MeshId{std::stoi(mesh_id_str)}}, .host_rank = host_rank};

    log_debug(
        tt::LogDistributed,
        "Local mesh binding: mesh_id: {}, host_rank: {}",
        local_mesh_binding.mesh_ids[0],
        local_mesh_binding.host_rank);

    // Validate the local mesh binding exists in the mesh graph descriptor
    const auto mesh_ids = this->routing_table_generator_->mesh_graph->get_mesh_ids();
    TT_FATAL(
        std::find(mesh_ids.begin(), mesh_ids.end(), local_mesh_binding.mesh_ids[0]) != mesh_ids.end(),
        "Invalid TT_MESH_ID: Local mesh binding mesh_id {} not found in mesh graph descriptor",
        *local_mesh_binding.mesh_ids[0]);

    // Validate host rank (only if mesh_id is valid)
    const auto& host_ranks =
        this->routing_table_generator_->mesh_graph->get_host_ranks(local_mesh_binding.mesh_ids[0]).values();
    if (host_rank_str == nullptr) {
        TT_FATAL(
            host_ranks.size() == 1 && *host_ranks.front() == 0,
            "TT_MESH_HOST_RANK must be set when multiple host ranks are present in the mesh graph descriptor for mesh "
            "ID {}",
            *local_mesh_binding.mesh_ids[0]);
    } else {
        TT_FATAL(
            std::find(host_ranks.begin(), host_ranks.end(), local_mesh_binding.host_rank) != host_ranks.end(),
            "Invalid TT_MESH_HOST_RANK: Local mesh binding host_rank {} not found in mesh graph descriptor",
            *local_mesh_binding.host_rank);
    }

    return local_mesh_binding;
}

void ControlPlane::init_control_plane(
    const std::string& mesh_graph_desc_file,
    std::optional<std::reference_wrapper<const std::map<FabricNodeId, chip_id_t>>>
        logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    this->local_mesh_binding_ = this->initialize_local_mesh_binding();

    const auto& global_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::transform(
        this->local_mesh_binding_.mesh_ids.begin(),
        this->local_mesh_binding_.mesh_ids.end(),
        std::inserter(this->distributed_contexts_, this->distributed_contexts_.end()),
        [&](const MeshId& mesh_id) { return std::make_pair(mesh_id, global_context); });
    if (*global_context->size() > 1) {
        std::array this_host = {*global_context->rank()};
        this->host_local_context_ =
            tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->create_sub_context(
                this_host);
    } else {
        this->host_local_context_ = global_context;
    }

    // Printing, only enabled with log_debug
    this->routing_table_generator_->mesh_graph->print_connectivity();
}

    if (logical_mesh_chip_id_to_physical_chip_id_mapping.has_value()) {
        this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping->get());
    } else {
        this->load_physical_chip_mapping(get_logical_chip_to_physical_chip_mapping(mesh_graph_desc_file));
    }
    this->initialize_intermesh_eth_links();
    this->generate_local_intermesh_link_table();
}

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    init_control_plane(mesh_graph_desc_file, std::nullopt);
}

ControlPlane::ControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    init_control_plane(mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void ControlPlane::load_physical_chip_mapping(
    const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    this->validate_mesh_connections();
}

void ControlPlane::validate_mesh_connections(MeshId mesh_id) const {
    MeshShape mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
    auto get_physical_chip_id = [&](const MeshCoordinate& mesh_coord) {
        auto fabric_chip_id = this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
        return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(FabricNodeId(mesh_id, fabric_chip_id));
    };
    auto validate_chip_connections = [&](const MeshCoordinate& mesh_coord, const MeshCoordinate& other_mesh_coord) {
        ChipId physical_chip_id = get_physical_chip_id(mesh_coord);
        ChipId physical_chip_id_other = get_physical_chip_id(other_mesh_coord);
        auto eth_links = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
        auto eth_links_to_other = eth_links.find(physical_chip_id_other);
        TT_FATAL(
            eth_links_to_other != eth_links.end(),
            "Chip {} not connected to chip {}",
            physical_chip_id,
            physical_chip_id_other);
    };
    const auto& mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
    for (const auto& mesh_coord : mesh_coord_range) {
        auto mode = mesh_coord_range.get_boundary_mode();

        auto col_neighbor = mesh_coord.get_neighbor(mesh_shape, 1, 1, mode);
        auto row_neighbor = mesh_coord.get_neighbor(mesh_shape, 1, 0, mode);

        if (col_neighbor.has_value() && mesh_coord_range.contains(*col_neighbor)) {
            validate_chip_connections(mesh_coord, *col_neighbor);
        }
        if (row_neighbor.has_value() && mesh_coord_range.contains(*row_neighbor)) {
            validate_chip_connections(mesh_coord, *row_neighbor);
        }
    }
}

void ControlPlane::validate_mesh_connections() const {
    for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
        if (this->is_local_mesh(mesh_id)) {
            this->validate_mesh_connections(mesh_id);
        }
    }
}

routing_plane_id_t ControlPlane::get_routing_plane_id(
    chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const {
    auto it = std::find(eth_chans_in_direction.begin(), eth_chans_in_direction.end(), eth_chan_id);
    return std::distance(eth_chans_in_direction.begin(), it);
}

routing_plane_id_t ControlPlane::get_routing_plane_id(FabricNodeId fabric_node_id, chan_id_t eth_chan_id) const {
    TT_FATAL(
        this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id),
        "Mesh {} Chip {} out of bounds",
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);

    std::optional<std::vector<chan_id_t>> eth_chans_in_direction;
    const auto& chip_eth_chans_map = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
    for (const auto& [_, eth_chans] : chip_eth_chans_map) {
        if (std::find(eth_chans.begin(), eth_chans.end(), eth_chan_id) != eth_chans.end()) {
            eth_chans_in_direction = eth_chans;
            break;
        }
    }
    TT_FATAL(
        eth_chans_in_direction.has_value(),
        "Could not find Eth chan ID {} for Chip ID {}, Mesh ID {}",
        eth_chan_id,
        fabric_node_id.chip_id,
        fabric_node_id.mesh_id);

    return get_routing_plane_id(eth_chan_id, eth_chans_in_direction.value());
}

chan_id_t ControlPlane::get_downstream_eth_chan_id(
    routing_plane_id_t src_routing_plane_id, const std::vector<chan_id_t>& candidate_target_chans) const {
    if (candidate_target_chans.empty()) {
        return eth_chan_magic_values::INVALID_DIRECTION;
    }

    for (const auto& target_chan_id : candidate_target_chans) {
        if (src_routing_plane_id == this->get_routing_plane_id(target_chan_id, candidate_target_chans)) {
            return target_chan_id;
        }
    }

    // TODO: for now disable collapsing routing planes until we add the corresponding logic for
    //     connecting the routers on these planes
    // If no match found, return a channel from candidate_target_chans
    // Enabled for TG Dispatch on Fabric
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG) {
        while (src_routing_plane_id >= candidate_target_chans.size()) {
            src_routing_plane_id = src_routing_plane_id % candidate_target_chans.size();
        }
        return candidate_target_chans[src_routing_plane_id];
    }

    return eth_chan_magic_values::INVALID_DIRECTION;
};

void ControlPlane::convert_fabric_routing_table_to_chip_routing_table() {
    // Routing tables contain direction from chip to chip
    // Convert it to be unique per ethernet channel

    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    // Get the number of ports per chip from a local mesh
    std::uint32_t num_ports_per_chip = 0;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < router_intra_mesh_routing_table.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        if (this->is_local_mesh(mesh_id)) {
            // TODO: Remove this once Topology mapper works for multi-mesh systems
            // Get the number of ports per chip from any chip in the local mesh
            auto local_mesh_chip_id_container =
                (this->topology_mapper_ == nullptr)
                    ? this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id)
                    : this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

            for (const auto& [_, src_fabric_chip_id] : local_mesh_chip_id_container) {
                const auto src_fabric_node_id = FabricNodeId(mesh_id, src_fabric_chip_id);
                auto physical_chip_id = get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
                num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                        .get_cluster()
                                        .get_soc_desc(physical_chip_id)
                                        .get_cores(CoreType::ETH)
                                        .size();
                break;
            }
        }
    }
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < router_intra_mesh_routing_table.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        const auto& global_mesh_chip_id_container = this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id);
        for (const auto& [_, src_fabric_chip_id] : global_mesh_chip_id_container) {
            const auto src_fabric_node_id = FabricNodeId(mesh_id, src_fabric_chip_id);
            this->intra_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed, this size is for all eth channels on chip
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of chips in the mesh
                this->intra_mesh_routing_tables_[src_fabric_node_id][i].resize(
                    router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id].size());
            }
            // Dst is looped over all chips in the mesh, regardless of whether they are local or not
            for (ChipId dst_fabric_chip_id = 0;
                 dst_fabric_chip_id < router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id].size();
                 dst_fabric_chip_id++) {
                // Target direction is the direction to the destination chip for all ethernet channesl
                const auto& target_direction =
                    router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id][dst_fabric_chip_id];
                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination chip as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_fabric_chip_id == dst_fabric_chip_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "Expecting same direction for intra mesh routing");
                            // This entry represents chip to itself, should not be used by FW
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                src_chan_id;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                this->get_downstream_eth_chan_id(src_routing_plane_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    for (std::uint32_t src_mesh_id_val = 0; src_mesh_id_val < router_inter_mesh_routing_table.size();
         src_mesh_id_val++) {
        MeshId src_mesh_id{src_mesh_id_val};
        const auto& global_mesh_chip_id_container =
            this->routing_table_generator_->mesh_graph->get_chip_ids(src_mesh_id);
        for (const auto& [_, src_fabric_chip_id] : global_mesh_chip_id_container) {
            const auto src_fabric_node_id = FabricNodeId(src_mesh_id, src_fabric_chip_id);
            this->inter_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of meshes
                this->inter_mesh_routing_tables_[src_fabric_node_id][i].resize(
                    router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id].size());
            }
            for (ChipId dst_mesh_id_val = 0;
                 dst_mesh_id_val < router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id].size();
                 dst_mesh_id_val++) {
                // Target direction is the direction to the destination mesh for all ethernet channesl
                const auto& target_direction =
                    router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id][dst_mesh_id_val];

                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination mesh as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_mesh_id_val == dst_mesh_id_val) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "ControlPlane: Expecting same direction for inter mesh routing");
                            // This entry represents mesh to itself, should not be used by FW
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                src_chan_id;
                        } else if (target_direction == RoutingDirection::NONE) {
                            // This entry represents a mesh to mesh connection that is not reachable
                            // Set to an invalid channel id
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                eth_chan_magic_values::INVALID_DIRECTION;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                this->get_downstream_eth_chan_id(src_routing_plane_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }

    // Printing, only enabled with log_debug
    this->print_routing_tables();
}

// order ethernet channels using translated coordinates
void ControlPlane::order_ethernet_channels() {
    for (auto& [fabric_node_id, eth_chans_by_dir] : this->router_port_directions_to_physical_eth_chan_map_) {
        for (auto& [_, eth_chans] : eth_chans_by_dir) {
            auto phys_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(phys_chip_id);

            std::sort(eth_chans.begin(), eth_chans.end(), [&soc_desc](const auto& a, const auto& b) {
                auto translated_coords_a = soc_desc.get_eth_core_for_channel(a, CoordSystem::TRANSLATED);
                auto translated_coords_b = soc_desc.get_eth_core_for_channel(b, CoordSystem::TRANSLATED);
                return translated_coords_a.x < translated_coords_b.x;
            });
        }
    }
}

void ControlPlane::trim_ethernet_channels_not_mapped_to_live_routing_planes() {
    auto user_mesh_ids = this->get_user_physical_mesh_ids();
    std::unordered_set<MeshId> user_mesh_ids_set(user_mesh_ids.begin(), user_mesh_ids.end());
    if (tt::tt_metal::MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::CUSTOM) {
        for (auto& [fabric_node_id, directional_eth_chans] : this->router_port_directions_to_physical_eth_chan_map_) {
            if (user_mesh_ids_set.count(fabric_node_id.mesh_id) == 0) {
                continue;
            }
            for (auto direction :
                 {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}) {
                if (directional_eth_chans.find(direction) != directional_eth_chans.end()) {
                    size_t num_available_routing_planes = this->get_num_live_routing_planes(fabric_node_id, direction);
                    TT_FATAL(
                        directional_eth_chans.at(direction).size() >= num_available_routing_planes,
                        "Expected {} eth channels on M{}D{} in direction {}, but got {}",
                        num_available_routing_planes,
                        fabric_node_id.mesh_id,
                        fabric_node_id.chip_id,
                        direction,
                        directional_eth_chans.at(direction).size());
                    bool trim = directional_eth_chans.at(direction).size() > num_available_routing_planes;
                    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
                    if (trim) {
                        log_warning(
                            tt::LogFabric,
                            "phys {} M{}D{} in direction {} has {} eth channels, but only {} routing planes are "
                            "available",
                            physical_chip_id,
                            fabric_node_id.mesh_id,
                            fabric_node_id.chip_id,
                            direction,
                            directional_eth_chans.at(direction).size(),
                            num_available_routing_planes);
                    }
                    directional_eth_chans.at(direction).resize(num_available_routing_planes);
                }
            }
        }
    }
}

size_t ControlPlane::get_num_live_routing_planes(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    TT_FATAL(
        this->router_port_directions_to_num_routing_planes_map_.find(fabric_node_id) !=
            this->router_port_directions_to_num_routing_planes_map_.end(),
        "Fabric node id (mesh={}, chip={}) not found in router port directions to num routing planes map",
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);
    TT_FATAL(
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).find(routing_direction) !=
            this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).end(),
        "Routing direction {} not found in router port directions to num routing planes map for fabric node id "
        "(mesh={}, chip={})",
        routing_direction,
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);
    return this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
}

// Only builds the routing table representation, does not actually populate the routing tables in memory of the
// fabric routers on device
void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels(
    tt::tt_fabric::FabricConfig fabric_config, tt_fabric::FabricReliabilityMode reliability_mode) {
    this->intra_mesh_routing_tables_.clear();
    this->inter_mesh_routing_tables_.clear();
    this->router_port_directions_to_physical_eth_chan_map_.clear();

    const auto& intra_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity();
    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    for (const auto& [fabric_node_id, _] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (!this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
            this->router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = {};
        }
    }

    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& my_host = physical_system_descriptor_->my_host_name();
    const auto& neighbor_hosts = physical_system_descriptor_->get_host_neighbors(my_host);

    for (std::uint32_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); mesh_id_val++) {
        // run for all meshes. intra_mesh_connectivity.size() == number of meshes in the system
        // TODO: we can probably remove this check, in general should update these loops to iterate over local meshes
        MeshId mesh_id{mesh_id_val};
        if (!this->is_local_mesh(mesh_id)) {
            continue;
        }
        const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);

        // TODO: Remove this once Topology mapper works for multi-mesh systems
        MeshContainer<ChipId> local_mesh_chip_id_container =
            (this->topology_mapper_ == nullptr)
                ? this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id)
                : this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

        for (const auto& [_, fabric_chip_id] : local_mesh_chip_id_container) {
            const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            auto physical_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);

            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[*mesh_id][fabric_chip_id]) {
                auto connected_mesh_coord =
                    this->routing_table_generator_->mesh_graph->chip_to_coordinate(mesh_id, logical_connected_chip_id);
                if (local_mesh_coord_range.contains(connected_mesh_coord)) {
                    // This is a local chip, so we can use the logical chip id directly
                    TT_ASSERT(
                        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(
                            FabricNodeId(mesh_id, logical_connected_chip_id)),
                        "Mesh {} Chip {} not found in logical mesh chip id to physical chip id mapping",
                        mesh_id,
                        logical_connected_chip_id);
                    const auto& physical_connected_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(
                        FabricNodeId(mesh_id, logical_connected_chip_id));

                    const auto& connected_chips_and_eth_cores =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                            physical_chip_id);

                    // If connected_chips_and_eth_cores contains physical_connected_chip_id then atleast one connection
                    // exists to physical_connected_chip_id
                    bool connections_exist = connected_chips_and_eth_cores.find(physical_connected_chip_id) !=
                                             connected_chips_and_eth_cores.end();
                    TT_FATAL(
                        connections_exist ||
                            reliability_mode != tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                        "Expected connections to exist for M{}D{} to D{}",
                        mesh_id,
                        fabric_chip_id,
                        logical_connected_chip_id);
                    if (!connections_exist) {
                        continue;
                    }

                    const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                    if (reliability_mode == tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
                        TT_FATAL(
                            connected_eth_cores.size() >= edge.connected_chip_ids.size(),
                            "Expected {} eth links from physical chip {} to physical chip {}",
                            edge.connected_chip_ids.size(),
                            physical_chip_id,
                            physical_connected_chip_id);
                    }

                    for (const auto& eth_core : connected_eth_cores) {
                        // There could be an optimization here to create entry for both chips here, assuming links are
                        // bidirectional
                        this->assign_direction_to_fabric_eth_core(fabric_node_id, eth_core, edge.port_direction);
                    }
                } else {
                    auto host_rank_for_chip =
                        (this->topology_mapper_ == nullptr)
                            ? this->routing_table_generator_->mesh_graph->get_host_rank_for_chip(
                                  mesh_id, logical_connected_chip_id)
                            : this->topology_mapper_->get_host_rank_for_chip(mesh_id, logical_connected_chip_id);
                    TT_ASSERT(
                        host_rank_for_chip.has_value(),
                        "Mesh {} Chip {} does not have a host rank associated with it",
                        mesh_id,
                        logical_connected_chip_id);
                    auto connected_host_rank_id = host_rank_for_chip.value();

                    auto unique_chip_id =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(physical_chip_id);
                    // Iterate over all neighboring hosts
                    // Check if the neighbor belongs to the same mesh and owns the connected chip
                    // If so, iterate over all cross host connections between the neighbors
                    // Assign this edge to all links on the local chip part of this intramesh connection
                    for (const auto& neighbor_host : neighbor_hosts) {
                        auto neighbor_host_rank = physical_system_descriptor_->get_rank_for_hostname(neighbor_host);
                        auto neighbor_mesh_id = this->global_logical_bindings_
                                                    .at(tt::tt_metal::distributed::multihost::Rank{neighbor_host_rank})
                                                    .first;
                        auto neighbor_mesh_host_rank =
                            this->global_logical_bindings_
                                .at(tt::tt_metal::distributed::multihost::Rank{neighbor_host_rank})
                                .second;
                        if (neighbor_mesh_id == mesh_id && neighbor_mesh_host_rank == connected_host_rank_id) {
                            const auto& neighbor_exit_nodes =
                                physical_system_descriptor_->get_connecting_exit_nodes(my_host, neighbor_host);
                            for (const auto& exit_node : neighbor_exit_nodes) {
                                if (*exit_node.src_exit_node == unique_chip_id) {
                                    this->assign_direction_to_fabric_eth_chan(
                                        fabric_node_id, exit_node.eth_conn.src_chan, edge.port_direction);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < inter_mesh_connectivity.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        if (this->is_local_mesh(mesh_id)) {
            const auto& local_mesh_chip_id_container =
                this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);
            for (const auto& [_, fabric_chip_id] : local_mesh_chip_id_container) {
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                if (*(distributed_context.size()) > 1) {
                    this->assign_intermesh_link_directions_to_remote_host(fabric_node_id);
                } else {
                    this->assign_intermesh_link_directions_to_local_host(fabric_node_id);
                }
            }
        }
    }

    this->initialize_dynamic_routing_plane_counts(intra_mesh_connectivity, fabric_config, reliability_mode);

    // Order the ethernet channels so that when we use them for deciding connections, indexing into ports per direction
    // is consistent for each each neighbouring chip.
    this->order_ethernet_channels();

    // Trim the ethernet channels that don't map to live fabric routing planes.
    // NOTE: This MUST be called after ordering ethernet channels
    this->trim_ethernet_channels_not_mapped_to_live_routing_planes();

    this->collect_and_merge_router_port_directions_from_all_hosts();

    this->convert_fabric_routing_table_to_chip_routing_table();
    // After this, router_port_directions_to_physical_eth_chan_map_, intra_mesh_routing_tables_,
    // inter_mesh_routing_tables_ should be populated for all hosts in BigMesh
}

void ControlPlane::write_routing_tables_to_eth_cores(MeshId mesh_id, ChipId chip_id) const {
    FabricNodeId fabric_node_id{mesh_id, chip_id};
    const auto& chip_intra_mesh_routing_tables = this->intra_mesh_routing_tables_.at(fabric_node_id);
    const auto& chip_inter_mesh_routing_tables = this->inter_mesh_routing_tables_.at(fabric_node_id);
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    // Loop over ethernet channels to only write to cores with ethernet links
    // Looping over chip_intra/inter_mesh_routing_tables will write to all cores, even if they don't have ethernet links
    const auto& chip_eth_chans_map = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
    for (const auto& [direction, eth_chans] : chip_eth_chans_map) {
        for (const auto& eth_chan : eth_chans) {
            // eth_chans are the active ethernet channels on this chip
            const auto& eth_chan_intra_mesh_routing_table = chip_intra_mesh_routing_tables[eth_chan];
            const auto& eth_chan_inter_mesh_routing_table = chip_inter_mesh_routing_tables[eth_chan];
            tt::tt_fabric::fabric_router_l1_config_t fabric_router_config{};
            std::fill_n(
                fabric_router_config.intra_mesh_table.dest_entry,
                tt::tt_fabric::MAX_MESH_SIZE,
                eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
            std::fill_n(
                fabric_router_config.inter_mesh_table.dest_entry,
                tt::tt_fabric::MAX_NUM_MESHES,
                eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
            for (uint32_t i = 0; i < eth_chan_intra_mesh_routing_table.size(); i++) {
                fabric_router_config.intra_mesh_table.dest_entry[i] = eth_chan_intra_mesh_routing_table[i];
            }
            for (uint32_t i = 0; i < eth_chan_inter_mesh_routing_table.size(); i++) {
                fabric_router_config.inter_mesh_table.dest_entry[i] = eth_chan_inter_mesh_routing_table[i];
            }

            const auto src_routing_plane_id = this->get_routing_plane_id(eth_chan, eth_chans);
            if (chip_eth_chans_map.find(RoutingDirection::N) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::N));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::S) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::S));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::E) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::E));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::W) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::W));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }

            fabric_router_config.my_mesh_id = *mesh_id;
            fabric_router_config.my_device_id = chip_id;
            MeshShape fabric_mesh_shape = this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
            fabric_router_config.north_dim = fabric_mesh_shape[0];
            fabric_router_config.east_dim = fabric_mesh_shape[1];

            // Write data to physical eth core
            CoreCoord virtual_eth_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                    physical_chip_id, eth_chan);

            TT_ASSERT(
                tt_metal::MetalContext::instance().hal().get_dev_size(
                    tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::FABRIC_ROUTER_CONFIG) ==
                    sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                "ControlPlane: Fabric router config size mismatch");
            log_debug(
                tt::LogFabric,
                "ControlPlane: Writing routing table to on M{}D{} eth channel {}",
                mesh_id,
                chip_id,
                eth_chan);
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                (void*)&fabric_router_config,
                sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                tt_cxy_pair(physical_chip_id, virtual_eth_core),
                tt_metal::MetalContext::instance().hal().get_dev_addr(
                    tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::FABRIC_ROUTER_CONFIG));
        }
    }
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_chip_id);
}

FabricNodeId ControlPlane::get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const {
    for (const auto& [fabric_node_id, mapped_physical_chip_id] :
         this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (mapped_physical_chip_id == physical_chip_id) {
            return fabric_node_id;
        }
    }
    TT_FATAL(
        false,
        "Physical chip id {} not found in control plane chip mapping. You are calling for a chip outside of the fabric "
        "cluster. Check that your mesh graph descriptor specifies the correct topology",
        physical_chip_id);
    return FabricNodeId(MeshId{0}, 0);
}

ChipId ControlPlane::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    TT_ASSERT(logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(fabric_node_id));
    return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
}

std::pair<FabricNodeId, chan_id_t> ControlPlane::get_connected_mesh_chip_chan_ids(
    FabricNodeId fabric_node_id, chan_id_t chan_id) const {
    // TODO: simplify this and use Global Physical Desc in ControlPlane soon
    const auto& intra_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();
    RoutingDirection port_direction = RoutingDirection::NONE;
    routing_plane_id_t routing_plane_id = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (eth_chan == chan_id) {
                port_direction = direction;
                routing_plane_id = this->get_routing_plane_id(eth_chan, eth_chans);
                break;
            }
        }
    }

    // Try to find the connected mesh chip chan ids for the given port direction in intra mesh connectivity
    const auto& intra_mesh_node = intra_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id];
    for (const auto& [dst_fabric_chip_id, edge] : intra_mesh_node) {
        if (edge.port_direction == port_direction) {
            // Get reverse port direction
            TT_ASSERT(
                intra_mesh_connectivity[*fabric_node_id.mesh_id][dst_fabric_chip_id].contains(fabric_node_id.chip_id),
                "Intra mesh connectivity from {} to {} not found",
                dst_fabric_chip_id,
                fabric_node_id.chip_id);
            RoutingDirection reverse_port_direction =
                intra_mesh_connectivity[*fabric_node_id.mesh_id][dst_fabric_chip_id]
                    .at(fabric_node_id.chip_id)
                    .port_direction;
            // Find the eth chan on connected dst_fabric_chip_id based on routing_plane_id
            const auto& dst_fabric_node = FabricNodeId(fabric_node_id.mesh_id, dst_fabric_chip_id);
            const auto& dst_fabric_chip_eth_chans =
                this->router_port_directions_to_physical_eth_chan_map_.at(dst_fabric_node);
            for (const auto& [direction, eth_chans] : dst_fabric_chip_eth_chans) {
                if (direction == reverse_port_direction) {
                    return std::make_pair(dst_fabric_node, eth_chans[routing_plane_id]);
                }
            }
        }
    }

    // Try to find the connected mesh chip chan ids for the given port direction in inter mesh connectivity
    const auto& inter_mesh_node = inter_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id];
    for (const auto& [dst_fabric_mesh_id, edge] : inter_mesh_node) {
        if (edge.port_direction == port_direction) {
            // Get reverse port direction
            const auto& dst_connected_fabric_chip_id = edge.connected_chip_ids[0];
            TT_ASSERT(
                inter_mesh_connectivity[*dst_fabric_mesh_id][dst_connected_fabric_chip_id].contains(
                    fabric_node_id.mesh_id),
                "Inter mesh connectivity from {} to {} not found",
                dst_fabric_mesh_id,
                fabric_node_id.mesh_id);
            RoutingDirection reverse_port_direction =
                inter_mesh_connectivity[*dst_fabric_mesh_id][dst_connected_fabric_chip_id]
                    .at(fabric_node_id.mesh_id)
                    .port_direction;
            // Find the eth chan on connected dst_fabric_mesh_id based on routing_plane_id
            const auto& dst_fabric_node = FabricNodeId(dst_fabric_mesh_id, dst_connected_fabric_chip_id);
            const auto& dst_fabric_chip_eth_chans =
                this->router_port_directions_to_physical_eth_chan_map_.at(dst_fabric_node);
            for (const auto& [direction, eth_chans] : dst_fabric_chip_eth_chans) {
                if (direction == reverse_port_direction) {
                    if (routing_plane_id >= eth_chans.size()) {
                        // Only TG non-standard intermesh connections hits this
                        return std::make_pair(dst_fabric_node, eth_chans[0]);
                    }
                    return std::make_pair(dst_fabric_node, eth_chans[routing_plane_id]);
                }
            }
        }
    }
    TT_FATAL(false, "Could not find connected mesh chip chan ids for {} on chan {}", fabric_node_id, chan_id);
    return std::make_pair(FabricNodeId(MeshId{0}, 0), 0);
}

std::vector<chan_id_t> ControlPlane::get_valid_eth_chans_on_routing_plane(
    FabricNodeId fabric_node_id, routing_plane_id_t routing_plane_id) const {
    std::vector<chan_id_t> valid_eth_chans;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (this->get_routing_plane_id(eth_chan, eth_chans) == routing_plane_id) {
                valid_eth_chans.push_back(eth_chan);
            }
        }
    }
    return valid_eth_chans;
}

eth_chan_directions ControlPlane::routing_direction_to_eth_direction(RoutingDirection direction) const {
    eth_chan_directions dir;
    switch (direction) {
        case RoutingDirection::N: dir = eth_chan_directions::NORTH; break;
        case RoutingDirection::S: dir = eth_chan_directions::SOUTH; break;
        case RoutingDirection::E: dir = eth_chan_directions::EAST; break;
        case RoutingDirection::W: dir = eth_chan_directions::WEST; break;
        default: TT_FATAL(false, "Invalid Routing Direction");
    }
    return dir;
}

std::set<std::pair<chan_id_t, eth_chan_directions>> ControlPlane::get_active_fabric_eth_channels(
    FabricNodeId fabric_node_id) const {
    std::set<std::pair<chan_id_t, eth_chan_directions>> active_fabric_eth_channels;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            active_fabric_eth_channels.insert({eth_chan, this->routing_direction_to_eth_direction(direction)});
        }
    }
    return active_fabric_eth_channels;
}

eth_chan_directions ControlPlane::get_eth_chan_direction(FabricNodeId fabric_node_id, int chan) const {
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (chan == eth_chan) {
                return this->routing_direction_to_eth_direction(direction);
            }
        }
    }
    TT_THROW("Cannot Find Ethernet Channel Direction");
}

std::vector<std::pair<FabricNodeId, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
    // Query the mesh coord range owned by the current host
    auto host_local_coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    auto src_mesh_coord = this->routing_table_generator_->mesh_graph->chip_to_coordinate(
        src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    auto dst_mesh_coord = this->routing_table_generator_->mesh_graph->chip_to_coordinate(
        dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);

    std::vector<std::pair<FabricNodeId, chan_id_t>> route;
    int i = 0;
    while (src_fabric_node_id != dst_fabric_node_id) {
        i++;
        auto src_mesh_id = src_fabric_node_id.mesh_id;
        auto src_chip_id = src_fabric_node_id.chip_id;
        auto dst_mesh_id = dst_fabric_node_id.mesh_id;
        auto dst_chip_id = dst_fabric_node_id.chip_id;
        if (i >= tt::tt_fabric::MAX_MESH_SIZE * tt::tt_fabric::MAX_NUM_MESHES) {
            log_warning(
                tt::LogFabric, "Could not find a route between {} and {}", src_fabric_node_id, dst_fabric_node_id);
            return {};
        }
        chan_id_t next_chan_id = 0;
        if (src_mesh_id != dst_mesh_id) {
            // Inter-mesh routing
            next_chan_id = this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][*dst_mesh_id];
        } else if (src_chip_id != dst_chip_id) {
            // Intra-mesh routing
            next_chan_id = this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id];
        }
        if (next_chan_id == eth_chan_magic_values::INVALID_DIRECTION) {
            // The complete route b/w src and dst not found, probably some eth cores are reserved along the path
            log_warning(
                tt::LogFabric, "Could not find a route between {} and {}", src_fabric_node_id, dst_fabric_node_id);
            return {};
        }
        if (src_chan_id != next_chan_id) {
            // Chan to chan within chip
            route.push_back({src_fabric_node_id, next_chan_id});
        }
        std::tie(src_fabric_node_id, src_chan_id) =
            this->get_connected_mesh_chip_chan_ids(src_fabric_node_id, next_chan_id);
        route.push_back({src_fabric_node_id, src_chan_id});
    }
    return route;
}

std::optional<RoutingDirection> ControlPlane::get_forwarding_direction(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const {
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    auto dst_mesh_id = dst_fabric_node_id.mesh_id;
    auto dst_chip_id = dst_fabric_node_id.chip_id;
    // TODO: remove returning of std::nullopt, and just return NONE value
    // Tests and usage should check for NONE value
    if (src_mesh_id != dst_mesh_id) {
        const auto& inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
        if (inter_mesh_routing_table[*src_mesh_id][src_chip_id][*dst_mesh_id] != RoutingDirection::NONE) {
            return inter_mesh_routing_table[*src_mesh_id][src_chip_id][*dst_mesh_id];
        }
    } else if (src_chip_id != dst_chip_id) {
        const auto& intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
        if (intra_mesh_routing_table[*src_mesh_id][src_chip_id][dst_chip_id] != RoutingDirection::NONE) {
            return intra_mesh_routing_table[*src_mesh_id][src_chip_id][dst_chip_id];
        }
    }
    return std::nullopt;
}

std::vector<chan_id_t> ControlPlane::get_forwarding_eth_chans_to_chip(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const {
    const auto& forwarding_direction = get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_direction.has_value()) {
        return {};
    }

    return this->get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, *forwarding_direction);
}

std::vector<chan_id_t> ControlPlane::get_forwarding_eth_chans_to_chip(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, RoutingDirection forwarding_direction) const {
    std::vector<chan_id_t> forwarding_channels;
    const auto& active_channels =
        this->get_active_fabric_eth_channels_in_direction(src_fabric_node_id, forwarding_direction);
    for (const auto& src_chan_id : active_channels) {
        // check for end-to-end route before accepting this channel
        if (this->get_fabric_route(src_fabric_node_id, dst_fabric_node_id, src_chan_id).empty()) {
            continue;
        }
        forwarding_channels.push_back(src_chan_id);
    }

    return forwarding_channels;
}

stl::Span<const ChipId> ControlPlane::get_intra_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    for (const auto& [_, routing_edge] :
         this->routing_table_generator_->mesh_graph
             ->get_intra_mesh_connectivity()[*src_fabric_node_id.mesh_id][src_fabric_node_id.chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            return routing_edge.connected_chip_ids;
        }
    }
    return {};
}

std::unordered_map<MeshId, std::vector<ChipId>> ControlPlane::get_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    std::unordered_map<MeshId, std::vector<ChipId>> neighbors;
    auto intra_neighbors = this->get_intra_chip_neighbors(src_fabric_node_id, routing_direction);
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    if (!intra_neighbors.empty()) {
        neighbors[src_mesh_id].insert(neighbors[src_mesh_id].end(), intra_neighbors.begin(), intra_neighbors.end());
    }
    for (const auto& [mesh_id, routing_edge] :
         this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity()[*src_mesh_id][src_chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            neighbors[mesh_id] = routing_edge.connected_chip_ids;
        }
    }
    return neighbors;
}

size_t ControlPlane::get_num_active_fabric_routers(FabricNodeId fabric_node_id) const {
    // Return the number of active fabric routers on the chip
    // Not always all the available FABRIC_ROUTER cores given by Cluster, since some may be disabled
    size_t num_routers = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        num_routers += eth_chans.size();
    }
    return num_routers;
}

std::vector<chan_id_t> ControlPlane::get_active_fabric_eth_channels_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        if (routing_direction == direction) {
            return eth_chans;
        }
    }
    return {};
}

void write_to_worker_or_fabric_tensix_cores(
    const void* worker_data,
    const void* dispatcher_data,
    const void* tensix_extension_data,
    size_t size,
    tt::tt_metal::HalL1MemAddrType addr_type,
    ChipId physical_chip_id) {
    TT_FATAL(
        size ==
            tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type),
        "ControlPlane: Tensix core data size mismatch expected {} but got {}",
        size,
        tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type));

    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);
    const std::vector<tt::umd::CoreCoord>& all_tensix_cores =
        soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);

    // Check if tensix config is enabled
    bool tensix_config_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                 tt::tt_fabric::FabricTensixConfig::DISABLED;

    // Get pre-computed translated fabric mux cores from tensix config
    std::unordered_set<CoreCoord> fabric_mux_cores_translated;
    std::unordered_set<CoreCoord> dispatch_mux_cores_translated;
    if (tensix_config_enabled) {
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const auto& tensix_config = fabric_context.get_tensix_config();
        fabric_mux_cores_translated = tensix_config.get_translated_fabric_mux_cores();
        dispatch_mux_cores_translated = tensix_config.get_translated_dispatch_mux_cores();
    }

    enum class CoreType { Worker, FabricTensixExtension, DispatcherMux };

    auto get_core_type = [&](const CoreCoord& core_coord) -> CoreType {
        if (fabric_mux_cores_translated.find(core_coord) != fabric_mux_cores_translated.end()) {
            return CoreType::FabricTensixExtension;
        }
        if (dispatch_mux_cores_translated.find(core_coord) != dispatch_mux_cores_translated.end()) {
            return CoreType::DispatcherMux;
        }
        return CoreType::Worker;
    };

    auto select_data = [&](CoreType core_type) -> const void* {
        if (tensix_config_enabled) {
            switch (core_type) {
                case CoreType::FabricTensixExtension: return worker_data;
                case CoreType::DispatcherMux: return dispatcher_data;
                case CoreType::Worker: return tensix_extension_data;
                default: TT_THROW("unknown core type: {}", core_type);
            }
        } else {
            return worker_data;
        }
    };

    for (const auto& tensix_core : all_tensix_cores) {
        CoreCoord core_coord(tensix_core.x, tensix_core.y);
        CoreType core_type = get_core_type(core_coord);
        const void* data_to_write = select_data(core_type);

        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            data_to_write,
            size,
            tt_cxy_pair(physical_chip_id, core_coord),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, addr_type));
    }
}

static void write_to_all_cores(
    const void* data,
    size_t size,
    tt::tt_metal::HalL1MemAddrType addr_type,
    ChipId physical_chip_id,
    tt::tt_metal::HalProgrammableCoreType core_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const char* type_label = "Unknown";
    switch (core_type) {
        case tt::tt_metal::HalProgrammableCoreType::TENSIX: type_label = "Tensix"; break;
        case tt::tt_metal::HalProgrammableCoreType::IDLE_ETH: type_label = "Idle ETH"; break;
        case tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH: type_label = "Active ETH"; break;
        default: break;
    }

    TT_FATAL(
        size == tt::tt_metal::MetalContext::instance().hal().get_dev_size(core_type, addr_type),
        "ControlPlane: {} core data size mismatch expected {} but got {}",
        type_label,
        tt::tt_metal::MetalContext::instance().hal().get_dev_size(core_type, addr_type),
        size);

    switch (core_type) {
        case tt::tt_metal::HalProgrammableCoreType::TENSIX: {
            const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
            const std::vector<tt::umd::CoreCoord>& tensix_cores =
                soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
            for (const auto& tensix_core : tensix_cores) {
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    data,
                    size,
                    tt_cxy_pair(physical_chip_id, CoreCoord(tensix_core.x, tensix_core.y)),
                    tt::tt_metal::MetalContext::instance().hal().get_dev_addr(core_type, addr_type));
            }
            break;
        }
        case tt::tt_metal::HalProgrammableCoreType::IDLE_ETH:
        case tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH: {
            std::unordered_set<CoreCoord> logical_eth_cores =
                (core_type == tt::tt_metal::HalProgrammableCoreType::IDLE_ETH)
                    ? tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(
                          physical_chip_id)
                    : tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
                          physical_chip_id);
            for (const CoreCoord& logical_eth_core : logical_eth_cores) {
                CoreCoord virtual_eth_core = cluster.get_virtual_coordinate_from_logical_coordinates(
                    physical_chip_id, logical_eth_core, CoreType::ETH);
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    data,
                    size,
                    tt_cxy_pair(physical_chip_id, CoreCoord(virtual_eth_core.x, virtual_eth_core.y)),
                    tt::tt_metal::MetalContext::instance().hal().get_dev_addr(core_type, addr_type));
            }
            break;
        }
        default: TT_THROW("Unsupported core type {}", static_cast<int>(core_type));
    }
}

// Helper functions to compute and embed routing path tables
void ControlPlane::compute_and_embed_1d_routing_path_table(
    MeshId mesh_id, tensix_routing_l1_info_t& tensix_routing_info) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& local_mesh_chip_id_container =
        (this->topology_mapper_ == nullptr)
            ? this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id)
            : this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);
    uint16_t num_chips = MAX_CHIPS_LOWLAT_1D < local_mesh_chip_id_container.size()
                             ? MAX_CHIPS_LOWLAT_1D
                             : static_cast<uint16_t>(local_mesh_chip_id_container.size());

    intra_mesh_routing_path_t<1, false> routing_path_1d;
    routing_path_1d.calculate_chip_to_all_routing_fields(FabricNodeId(mesh_id, 0), num_chips);

    std::memcpy(
        &tensix_routing_info.routing_path_table_1d, &routing_path_1d, sizeof(intra_mesh_routing_path_t<1, false>));
}

void ControlPlane::compute_and_embed_2d_routing_path_table(
    MeshId mesh_id, ChipId chip_id, tensix_routing_l1_info_t& tensix_routing_info) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    auto local_mesh_chip_id_container =
        (this->topology_mapper_ == nullptr)
            ? this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id)
            : this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

    bool chip_is_local_to_host = false;
    for (const auto& [_, local_chip_id] : local_mesh_chip_id_container) {
        if (local_chip_id == chip_id) {
            chip_is_local_to_host = true;
            break;
        }
    }
    TT_ASSERT(
        chip_is_local_to_host,
        "2D routing path: chip {} is not owned by local host_rank {} for mesh {}",
        chip_id,
        *host_rank_id,
        *mesh_id);

    // Calculate routing using global mesh geometry (device tables are indexed by global chip ids)
    MeshShape mesh_shape = this->get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    TT_ASSERT(num_chips <= 256, "Number of chips exceeds 256 for mesh {}", *mesh_id);
    TT_ASSERT(
        mesh_shape[0] <= 16 && mesh_shape[1] <= 16,
        "One or both of mesh axis exceed 16 for mesh {}: {}x{}",
        *mesh_id,
        mesh_shape[0],
        mesh_shape[1]);

    intra_mesh_routing_path_t<2, true> routing_path_2d;
    routing_path_2d.calculate_chip_to_all_routing_fields(FabricNodeId(mesh_id, chip_id), num_chips);

    std::memcpy(
        &tensix_routing_info.routing_path_table_2d, &routing_path_2d, sizeof(intra_mesh_routing_path_t<2, true>));

    // Build per-dst-mesh exit node table (1 byte per mesh) for this src chip
    exit_node_table_t exit_table{};
    std::fill_n(exit_table.nodes, MAX_NUM_MESHES, eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    const auto& inter_mesh_table = this->routing_table_generator_->get_inter_mesh_table();
    for (const auto& dst_mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
        auto direction = inter_mesh_table[*mesh_id][chip_id][*dst_mesh_id];
        if (direction == RoutingDirection::NONE) {
            continue;
        }
        auto exit_node = this->routing_table_generator_->get_exit_node_from_mesh_to_mesh(mesh_id, chip_id, dst_mesh_id);
        exit_table.nodes[*dst_mesh_id] = static_cast<std::uint8_t>(exit_node.chip_id);
    }
    std::memcpy(&tensix_routing_info.exit_node_table, &exit_table, sizeof(exit_node_table_t));
}

// Write routing table to Tensix cores' L1 on a specific chip
void ControlPlane::write_routing_tables_to_tensix_cores(MeshId mesh_id, ChipId chip_id) const {
    FabricNodeId src_fabric_node_id{mesh_id, chip_id};
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

    tensix_routing_l1_info_t tensix_routing_info = {};
    tensix_routing_info.my_mesh_id = *mesh_id;
    tensix_routing_info.my_device_id = chip_id;

    // Build intra-mesh routing entries (chip-to-chip routing)
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    TT_FATAL(
        router_intra_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_MESH_SIZE,
        "ControlPlane: Intra mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_MESH_SIZE; i++) {
        tensix_routing_info.intra_mesh_routing_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (ChipId dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[*mesh_id][chip_id].size();
         dst_chip_id++) {
        if (chip_id == dst_chip_id) {
            tensix_routing_info.intra_mesh_routing_table.set_original_direction(
                dst_chip_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_intra_mesh_routing_table[*mesh_id][chip_id][dst_chip_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        tensix_routing_info.intra_mesh_routing_table.set_original_direction(dst_chip_id, direction_value);
    }

    // Build inter-mesh routing entries (mesh-to-mesh routing)
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    TT_FATAL(
        router_inter_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_NUM_MESHES,
        "ControlPlane: Inter mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_NUM_MESHES; i++) {
        tensix_routing_info.inter_mesh_routing_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (std::uint32_t dst_mesh_id = 0; dst_mesh_id < router_inter_mesh_routing_table[*mesh_id][chip_id].size();
         dst_mesh_id++) {
        if (*mesh_id == dst_mesh_id) {
            tensix_routing_info.inter_mesh_routing_table.set_original_direction(
                dst_mesh_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_inter_mesh_routing_table[*mesh_id][chip_id][dst_mesh_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        tensix_routing_info.inter_mesh_routing_table.set_original_direction(dst_mesh_id, direction_value);
    }

    if (this->get_fabric_context().is_2D_routing_enabled()) {
        // Compute and embed 2D routing path table and exit node table (per src chip id)
        compute_and_embed_2d_routing_path_table(mesh_id, chip_id, tensix_routing_info);
    } else {
        // Compute and embed 1D routing path table (independent of src chip id)
        compute_and_embed_1d_routing_path_table(mesh_id, tensix_routing_info);
    }

    // Finally, write the full routing info to all Tensix cores and mirror to IDLE_ETH routing table
    write_to_all_cores(
        &tensix_routing_info,
        sizeof(tensix_routing_info),
        tt::tt_metal::HalL1MemAddrType::TENSIX_ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::TENSIX);
    write_to_all_cores(
        &tensix_routing_info,
        sizeof(tensix_routing_info),
        tt::tt_metal::HalL1MemAddrType::FABRIC_ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
    write_to_all_cores(
        &tensix_routing_info,
        // TODO: https://github.com/tenstorrent/tt-metal/issues/27881
        //      Active ETH doesn't have enough space (yet)
        sizeof(tensix_routing_info) - sizeof(exit_node_table_t),
        tt::tt_metal::HalL1MemAddrType::FABRIC_ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
}

// Write connection info to Tensix cores' L1 on a specific chip
void ControlPlane::write_fabric_connections_to_tensix_cores(MeshId mesh_id, ChipId chip_id) const {
    if (this->fabric_context_ == nullptr) {
        log_warning(
            tt::LogFabric,
            "ControlPlane: Fabric context is not set, cannot write fabric connections to Tensix cores for M%dD%d",
            *mesh_id,
            chip_id);
        return;
    }
    FabricNodeId src_fabric_node_id{mesh_id, chip_id};
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_worker_connections = {};
    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_dispatcher_connections = {};
    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_tensix_connections = {};

    // Get all physically connected ethernet channels directly from the cluster
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& connected_chips_and_eth_cores = cluster.get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);

    size_t num_eth_endpoint = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
        for (auto eth_channel_id : eth_chans) {
            eth_chan_directions router_direction = this->routing_direction_to_eth_direction(direction);
            if (num_eth_endpoint >= tt::tt_fabric::tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS) {
                log_warning(
                    tt::LogFabric,
                    "ControlPlane: Maximum number of fabric endpoints exceeded for M%dD%d, skipping further "
                    "connections",
                    *mesh_id,
                    chip_id);
                break;
            }

            // Populate connection info for regular fabric connections (for tensix mux cores)
            auto& worker_connection_info = fabric_worker_connections.read_only[eth_channel_id];
            worker_connection_info.edm_direction = router_direction;

            // Populate connection info for dispatcher fabric connections
            auto& dispatcher_connection_info = fabric_dispatcher_connections.read_only[eth_channel_id];
            dispatcher_connection_info.edm_direction = router_direction;

            // Populate connection info for tensix mux connections (for normal worker cores)
            auto& tensix_connection_info = fabric_tensix_connections.read_only[eth_channel_id];
            tensix_connection_info.edm_direction = router_direction;

            // Use helper function to populate both connection types
            this->populate_fabric_connection_info(
                worker_connection_info,
                dispatcher_connection_info,
                tensix_connection_info,
                physical_chip_id,
                eth_channel_id,
                router_direction);

            // Mark this connection as valid for fabric communication
            fabric_worker_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_dispatcher_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_tensix_connections.valid_connections_mask |= (1u << eth_channel_id);
            num_eth_endpoint++;
        }
    }

    // Write fabric connections (fabric router config) to mux cores and tensix connections (tensix config) to worker
    // cores
    write_to_worker_or_fabric_tensix_cores(
        &fabric_worker_connections,      // worker_data - goes to mux cores
        &fabric_dispatcher_connections,  // dispatcher_data - goes to dispatcher cores
        &fabric_tensix_connections,      // tensix_extension_data - goes to worker cores
        sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t),
        tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS,
        physical_chip_id);
}

std::vector<chan_id_t> ControlPlane::get_active_fabric_eth_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    auto eth_chans = get_active_fabric_eth_channels_in_direction(fabric_node_id, routing_direction);
    size_t num_routing_planes = 0;
    if (this->router_port_directions_to_num_routing_planes_map_.contains(fabric_node_id) &&
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).contains(routing_direction)) {
        num_routing_planes =
            this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
        TT_FATAL(
            eth_chans.size() >= num_routing_planes,
            "Not enough active fabric eth channels for node {} in direction {}. Requested {} routing planes but only "
            "have {} eth channels",
            fabric_node_id,
            routing_direction,
            num_routing_planes,
            eth_chans.size());
        eth_chans.resize(num_routing_planes);
    }
    return eth_chans;
}

size_t ControlPlane::get_num_available_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    if (this->router_port_directions_to_num_routing_planes_map_.contains(fabric_node_id) &&
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).contains(routing_direction)) {
        return this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
    }
    return 0;
}

void ControlPlane::write_routing_tables_to_all_chips() const {
    // Configure the routing tables on the chips
    TT_ASSERT(
        this->intra_mesh_routing_tables_.size() == this->inter_mesh_routing_tables_.size(),
        "Intra mesh routing tables size mismatch with inter mesh routing tables");
    auto user_meshes = this->get_user_physical_mesh_ids();
    for (auto mesh_id : user_meshes) {
        const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
        for (const auto& mesh_coord : local_mesh_coord_range) {
            auto fabric_chip_id = this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
            auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            TT_ASSERT(
                this->inter_mesh_routing_tables_.contains(fabric_node_id),
                "Intra mesh routing tables keys mismatch with inter mesh routing tables");
            this->write_routing_tables_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_fabric_connections_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_routing_tables_to_eth_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        }
    }
}

// TODO: remove this after TG is deprecated
std::vector<MeshId> ControlPlane::get_user_physical_mesh_ids() const {
    std::vector<MeshId> physical_mesh_ids;
    const auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    for (const auto& [fabric_node_id, physical_chip_id] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (user_chips.find(physical_chip_id) != user_chips.end() and
            std::find(physical_mesh_ids.begin(), physical_mesh_ids.end(), fabric_node_id.mesh_id) ==
                physical_mesh_ids.end()) {
            physical_mesh_ids.push_back(fabric_node_id.mesh_id);
        }
    }
    return physical_mesh_ids;
}

MeshShape ControlPlane::get_physical_mesh_shape(MeshId mesh_id, MeshScope scope) const {
    std::optional<MeshHostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;

    // TODO: Remove this once Topology mapper works for multi-mesh systems
    if (this->topology_mapper_ == nullptr) {
        return this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id, local_host_rank_id);
    }

    return this->topology_mapper_->get_mesh_shape(mesh_id, local_host_rank_id);
}

void ControlPlane::print_routing_tables() const {
    this->print_ethernet_channels();

    std::stringstream ss;
    ss << "Control Plane: IntraMesh Routing Tables" << std::endl;
    for (const auto& [fabric_node_id, chip_routing_table] : this->intra_mesh_routing_tables_) {
        ss << fabric_node_id << ":" << std::endl;
        for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
            ss << "   Eth Chan " << eth_chan << ": ";
            for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                ss << (std::uint16_t)dst_chan_id << " ";
            }
            ss << std::endl;
        }
    }

    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Control Plane: InterMesh Routing Tables" << std::endl;

    for (const auto& [fabric_node_id, chip_routing_table] : this->inter_mesh_routing_tables_) {
        ss << fabric_node_id << ":" << std::endl;
        for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
            ss << "   Eth Chan " << eth_chan << ": ";
            for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                ss << (std::uint16_t)dst_chan_id << " ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::print_ethernet_channels() const {
    std::stringstream ss;
    ss << "Control Plane: Physical eth channels in each direction" << std::endl;
    for (const auto& [fabric_node_id, fabric_eth_channels] : this->router_port_directions_to_physical_eth_chan_map_) {
        ss << fabric_node_id << ": " << std::endl;
        for (const auto& [direction, eth_chans] : fabric_eth_channels) {
            ss << "   " << enchantum::to_string(direction) << ":";
            for (const auto& eth_chan : eth_chans) {
                ss << " " << (std::uint16_t)eth_chan;
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::set_routing_mode(uint16_t mode) {
    if (!(this->routing_mode_ == 0 || this->routing_mode_ == mode)) {
        log_warning(
            tt::LogFabric,
            "Control Plane: Routing mode already set to {}. Setting to {}",
            (uint16_t)this->routing_mode_,
            (uint16_t)mode);
    }
    this->routing_mode_ = mode;
}

uint16_t ControlPlane::get_routing_mode() const { return this->routing_mode_; }

void ControlPlane::initialize_fabric_context(tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(this->fabric_context_ == nullptr, "Trying to re-initialize fabric context");
    this->fabric_context_ = std::make_unique<FabricContext>(fabric_config);
}

FabricContext& ControlPlane::get_fabric_context() const {
    TT_FATAL(this->fabric_context_ != nullptr, "Trying to get un-initialized fabric context");
    return *this->fabric_context_;
}

void ControlPlane::clear_fabric_context() { this->fabric_context_.reset(nullptr); }

void ControlPlane::initialize_fabric_tensix_datamover_config() {
    TT_FATAL(this->fabric_context_ != nullptr, "Fabric context must be initialized first");
    this->fabric_context_->initialize_tensix_config();
}

bool ControlPlane::is_cross_host_eth_link(ChipId chip_id, chan_id_t chan_id) const {
    auto asic_id = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(chip_id);
    return this->physical_system_descriptor_->is_cross_host_eth_link(tt::tt_metal::AsicID{asic_id}, chan_id);
}

std::unordered_set<CoreCoord> ControlPlane::get_active_ethernet_cores(ChipId chip_id, bool skip_reserved_cores) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    std::unordered_set<CoreCoord> active_ethernet_cores;
    const auto& cluster_desc = cluster.get_cluster_desc();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);

    // Check if there are any ethernet cores available on this chip
    if (soc_desc.logical_eth_core_to_chan_map.empty()) {
        return active_ethernet_cores;  // Return empty set if no ethernet cores
    }

    if (cluster.arch() == ARCH::BLACKHOLE) {
        // Can't just use `get_ethernet_cores_grouped_by_connected_chips` because there are some active ethernet cores
        // without links. Only risc1 on these cores is available for Metal and should not be classified as idle
        // to ensure that Metal does not try to program both riscs.
        std::set<uint32_t> logical_active_eth_channels = cluster_desc->get_active_eth_channels(chip_id);
        for (auto logical_active_eth_channel : logical_active_eth_channels) {
            tt::umd::CoreCoord logical_active_eth =
                soc_desc.get_eth_core_for_channel(logical_active_eth_channel, CoordSystem::LOGICAL);
            active_ethernet_cores.insert(CoreCoord(logical_active_eth.x, logical_active_eth.y));
        }
    } else {
        std::set<uint32_t> logical_active_eth_channels = cluster_desc->get_active_eth_channels(chip_id);
        const auto& freq_retrain_eth_cores = cluster.get_eth_cores_with_frequent_retraining(chip_id);
        const auto& eth_routing_info = cluster.get_eth_routing_info(chip_id);
        for (const auto& eth_channel : logical_active_eth_channels) {
            tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
            const auto& routing_info = eth_routing_info.at(eth_core);
            if (routing_info == EthRouterMode::FABRIC_ROUTER && skip_reserved_cores) {
                continue;
            }
            if (freq_retrain_eth_cores.find(eth_core) != freq_retrain_eth_cores.end()) {
                continue;
            }

            active_ethernet_cores.insert(eth_core);
        }
        // WH has a special case where mmio chips with remote connections must always have certain channels active
        if (cluster.arch() == tt::ARCH::WORMHOLE_B0 && cluster_desc->is_chip_mmio_capable(chip_id) &&
            !cluster.get_tunnels_from_mmio_device(chip_id).empty()) {
            // UMD routing FW uses these cores for base routing
            // channel 15 is used by syseng tools
            std::unordered_set<int> channels_to_skip = {};
            if (cluster.is_galaxy_cluster()) {
                // TODO: This may need to change, if we need additional eth cores for dispatch on Galaxy
                channels_to_skip = {0, 1, 2, 3, 15};
            } else {
                channels_to_skip = {15};
            }
            for (const auto& eth_channel : channels_to_skip) {
                if (logical_active_eth_channels.find(eth_channel) == logical_active_eth_channels.end()) {
                    tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                    active_ethernet_cores.insert(eth_core);
                }
            }
        }
    }
    return active_ethernet_cores;
}

std::unordered_set<CoreCoord> ControlPlane::get_inactive_ethernet_cores(ChipId chip_id) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;

    for (const auto& [eth_core, chan] : cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (active_ethernet_cores.find(eth_core) == active_ethernet_cores.end()) {
            inactive_ethernet_cores.insert(eth_core);
        }
    }
    return inactive_ethernet_cores;
}

void ControlPlane::generate_local_intermesh_link_table() {
    // Populate the local to remote mapping for all intermesh links
    // This cannot be done by UMD, since it has no knowledge of links marked
    // for intermesh routing (these links are hidden from UMD).
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    intermesh_link_table_.local_mesh_id = local_mesh_binding_.mesh_ids[0];
    intermesh_link_table_.local_host_rank_id = this->get_local_host_rank_id_binding();
    const uint32_t remote_config_base_addr = tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::ETH_LINK_REMOTE_INFO);
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        if (this->has_intermesh_links(chip_id)) {
            for (const auto& [eth_core, chan_id] : this->get_intermesh_eth_links(chip_id)) {
                // TODO: remove below logic, should at very least be using UMD apis to get ids
                // But all this data can be provided by UMD
                tt_cxy_pair virtual_eth_core(
                    chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                uint64_t local_board_id = 0;
                uint64_t remote_board_id = 0;
                uint32_t remote_chan_id = 0;
                cluster.read_core(
                    &local_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::LOCAL_BOARD_ID_OFFSET);
                cluster.read_core(
                    &remote_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::REMOTE_BOARD_ID_OFFSET);
                cluster.read_core(
                    &remote_chan_id,
                    sizeof(uint32_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::REMOTE_ETH_CHAN_ID_OFFSET);
                auto local_eth_chan_desc = EthChanDescriptor{
                    .board_id = local_board_id,
                    .chan_id = chan_id,
                };
                auto remote_eth_chan_desc = EthChanDescriptor{
                    .board_id = remote_board_id,
                    .chan_id = remote_chan_id,
                };
                intermesh_link_table_.intermesh_links[local_eth_chan_desc] = remote_eth_chan_desc;
                chip_id_to_asic_id_[chip_id] = local_board_id;
            }
        } else if (cluster.arch() != ARCH::BLACKHOLE) {
            // For chips without intermesh links, we still need to populate the asic IDs
            // for consistency.
            // Skip this on Blackhole for now.
            if (this->get_active_ethernet_cores(chip_id).size() == 0) {
                // No Active Ethernet Cores found. Not querying the board id off ethernet cores.
                chip_id_to_asic_id_[chip_id] = chip_id;
            } else {
                auto first_eth_core = *(this->get_active_ethernet_cores(chip_id).begin());
                tt_cxy_pair virtual_eth_core(
                    chip_id,
                    cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, first_eth_core, CoreType::ETH));
                uint64_t local_board_id = 0;
                cluster.read_core(
                    &local_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::LOCAL_BOARD_ID_OFFSET);
                chip_id_to_asic_id_[chip_id] = local_board_id;
            }
        }
    }
}

void ControlPlane::exchange_intermesh_link_tables() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    if (*distributed_context.size() == 1) {
        // No need to exchange intermesh link tables when running a single process
        return;
    }

    auto serialized_table = tt::tt_fabric::serialize_to_bytes(intermesh_link_table_);
    std::vector<uint8_t> serialized_remote_table;
    auto my_rank = *(distributed_context.rank());

    for (std::size_t bcast_root = 0; bcast_root < *(distributed_context.size()); ++bcast_root) {
        if (my_rank == bcast_root) {
            // Issue the broadcast from the current process to all other processes in the world
            int local_table_size_bytes = serialized_table.size();  // Send txn size first
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&local_table_size_bytes), sizeof(local_table_size_bytes)),
                distributed_context.rank());

            distributed_context.broadcast(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
                distributed_context.rank());
        } else {
            // Acknowledge the broadcast issued by the root
            int remote_table_size_bytes = 0;  // Receive the size of the serialized descriptor
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&remote_table_size_bytes), sizeof(remote_table_size_bytes)),
                tt::tt_metal::distributed::multihost::Rank{bcast_root});
            serialized_remote_table.clear();
            serialized_remote_table.resize(remote_table_size_bytes);
            distributed_context.broadcast(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_remote_table.data(), serialized_remote_table.size())),
                tt::tt_metal::distributed::multihost::Rank{bcast_root});
            tt_fabric::IntermeshLinkTable deserialized_remote_table =
                tt::tt_fabric::deserialize_from_bytes(serialized_remote_table);
            peer_intermesh_link_tables_[deserialized_remote_table.local_mesh_id]
                                       [deserialized_remote_table.local_host_rank_id] =
                                           std::move(deserialized_remote_table.intermesh_links);
        }
        // Barrier here for safety - Ensure that all ranks have completed the bcast op before proceeding to the next
        // root
        distributed_context.barrier();
    }
}

void ControlPlane::assign_direction_to_fabric_eth_core(
    const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction) {
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    // TODO: get_fabric_ethernet_channels accounts for down links, but we should manage down links in control plane
    auto fabric_router_channels_on_chip =
        tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_channels(physical_chip_id);

    // TODO: add logic here to disable unsed routers, e.g. Mesh on Torus system
    if (fabric_router_channels_on_chip.contains(chan_id)) {
        this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)[direction].push_back(chan_id);
    } else {
        log_debug(
            tt::LogFabric,
            "Control Plane: Disabling router on M{}D{} eth channel {}",
            fabric_node_id.mesh_id,
            fabric_node_id.chip_id,
            chan_id);
    }
}

void ControlPlane::assign_direction_to_fabric_eth_core(
    const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction) {
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    auto chan_id = tt::tt_metal::MetalContext::instance()
                       .get_cluster()
                       .get_soc_desc(physical_chip_id)
                       .logical_eth_core_to_chan_map.at(eth_core);
    this->assign_direction_to_fabric_eth_chan(fabric_node_id, chan_id, direction);
}

const MeshGraph& ControlPlane::get_mesh_graph() const { return *routing_table_generator_->mesh_graph; }

std::vector<MeshId> ControlPlane::get_local_mesh_id_bindings() const {
    const auto& mesh_id_bindings = this->local_mesh_binding_.mesh_ids;
    const auto& user_mesh_ids = this->get_user_physical_mesh_ids();
    std::vector<MeshId> local_mesh_ids;
    for (const auto& mesh_id : mesh_id_bindings) {
        if (std::find(user_mesh_ids.begin(), user_mesh_ids.end(), mesh_id) != user_mesh_ids.end()) {
            local_mesh_ids.push_back(mesh_id);
        }
    }
    TT_FATAL(!local_mesh_ids.empty(), "No local mesh ids found");
    return local_mesh_ids;
}

MeshHostRankId ControlPlane::get_local_host_rank_id_binding() const { return this->local_mesh_binding_.host_rank; }

MeshCoordinate ControlPlane::get_local_mesh_offset() const {
    auto coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    return coord_range.start_coord();
}

MeshCoordinateRange ControlPlane::get_coord_range(MeshId mesh_id, MeshScope scope) const {
    std::optional<MeshHostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;

    // TODO: Remove this once Topology mapper works for multi-mesh systems
    if (this->topology_mapper_ == nullptr) {
        return this->routing_table_generator_->mesh_graph->get_coord_range(mesh_id, local_host_rank_id);
    }

    return this->topology_mapper_->get_coord_range(mesh_id, local_host_rank_id);
}

bool ControlPlane::is_local_mesh(MeshId mesh_id) const {
    const auto& local_mesh_ids = local_mesh_binding_.mesh_ids;
    return std::find(local_mesh_ids.begin(), local_mesh_ids.end(), mesh_id) != local_mesh_ids.end();
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_distributed_context(
    MeshId mesh_id) const {
    auto distributed_context = distributed_contexts_.find(mesh_id);
    TT_FATAL(distributed_context != distributed_contexts_.end(), "Unknown mesh id: {}", mesh_id);
    return distributed_context->second;
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_host_local_context()
    const {
    return host_local_context_;
}

ControlPlane::~ControlPlane() = default;

}  // namespace tt::tt_fabric
