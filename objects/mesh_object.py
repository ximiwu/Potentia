import imp
from typing import List, Optional, Tuple

import taichi as ti
from taichi.math import vec3

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from energies.distance_energy import DistanceEnergy
from energies.pd_bending_energy import PDBendingEnergy
from energies.pd_strain_energy import PDStrainEnergy
from mesh.base import IEdgeDataProvider, IMesh
from mesh.transforms import rotate_positions, scale_positions
from .base import IMeshObject

@ti.data_oriented
class MeshObject(IMeshObject):
    """
    A concrete implementation of IMeshObject.

    This class binds a static mesh (IMesh) to a dynamic slice of simulation data
    (ISimulationData) and serves as a factory for creating potential energy terms
    that act upon that data.
    """

    def __init__(
        self,
        mesh: IMesh,
        data: ISimulationData,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        face_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        edge_color: Optional[Tuple[float, float, float]] = None,
        vertex_color: Optional[Tuple[float, float, float]] = None,
        mass: float = 1.0,
    ):
        # Interface adherence via capability checks rather than isinstance
        required_mesh_methods = ['get_rest_positions', 'get_num_vertices']
        for method_name in required_mesh_methods:
            if not hasattr(mesh, method_name):
                raise TypeError(f"Argument 'mesh' must implement IMesh capability '{method_name}'.")
        required_data_methods = [
            'allocate_dofs', 'get_dofs', 'get_inv_masses', 'get_masses'
        ]
        for method_name in required_data_methods:
            if not hasattr(data, method_name):
                raise TypeError(f"Argument 'data' must implement ISimulationData capability '{method_name}'.")

        # Cache DistanceEnergy singleton for use inside Taichi kernels via ti.static
        self._distance_energy = DistanceEnergy.get_instance()
        self._pd_bending_energy = PDBendingEnergy.get_instance()
        self._pd_strain_energy = PDStrainEnergy.get_instance()
        self._energy_container = GlobalEnergyContainer.get_instance()

        self._mesh = mesh
        self._data = data
        self._mass = mass
        self._face_color = face_color
        self._edge_color = edge_color
        self._vertex_color = vertex_color
        
        # Automatically allocate space from the simulation data container
        self._data_offset = self._data.allocate_dofs(self._mesh.get_num_vertices())

        # --- Apply transformations ---
        # The transformation pipeline is Scale -> Rotate -> Translate
        transformed_positions = self._mesh.get_rest_positions()

        # 1. Scale
        if scale != (1.0, 1.0, 1.0):
            transformed_positions = scale_positions(transformed_positions, scale)
        
        # 2. Rotate
        if rotation != (0.0, 0.0, 0.0):
            transformed_positions = rotate_positions(transformed_positions, rotation)
        
        # self._initialize_positions_kernel()
        self._initialize_positions_kernel(
            transformed_positions,
            self._data.get_dofs(),
            self._data.get_inv_masses(),
            self._data.get_masses(),
            self._data_offset,
            vec3(*translation),
            self._mass,
        )

        # --- Append vertex adjacency (CSR) from mesh into global data (with global offset) ---
        # Capability checks via hasattr; do nothing if data does not expose adjacency container.
        if hasattr(self._data, 'get_vertex_adjacency') and hasattr(self._data, 'get_vertex_adjacency_write_ptr'):
            offsets_global, info_global = self._data.get_vertex_adjacency()
            write_ptr = self._data.get_vertex_adjacency_write_ptr()

            V = int(self._mesh.get_num_vertices())

            if hasattr(self._mesh, 'get_vertex_adjacency_offsets') \
               and hasattr(self._mesh, 'get_vertex_adjacency_indices') \
               and hasattr(self._mesh, 'get_vertex_adjacency_cotan_weights'):
                # Require max_degree capability on data for strict degree checking
                if not hasattr(self._data, 'get_max_degree'):
                    raise TypeError("ISimulationData must provide get_max_degree() when using global adjacency CSR.")

                mesh_offsets = self._mesh.get_vertex_adjacency_offsets()
                mesh_indices = self._mesh.get_vertex_adjacency_indices()
                mesh_weights = self._mesh.get_vertex_adjacency_cotan_weights()

                max_degree = int(self._data.get_max_degree())
                max_entries = int(info_global.shape[0])

                overflow = self._append_vertex_adjacency_kernel(
                    offsets_global,
                    info_global,
                    write_ptr,
                    mesh_offsets,
                    mesh_indices,
                    mesh_weights,
                    self._data_offset,
                    V,
                    max_degree,
                    max_entries,
                )
                if overflow == 1:
                    raise RuntimeError(
                        f"顶点度数超过 max_degree={max_degree}。请调大 MassPointData(max_degree)。")
                if overflow == 2:
                    raise RuntimeError(
                        "全局邻接信息容量不足 (max_point_num * max_degree)。请增大 max_degree 或 max_point_num。")
            else:
                # Mesh 无邻接能力时，为此对象写入零度邻接（offsets 指向同一 write_ptr，info 不追加）
                self._append_zero_degree_adjacency_kernel(
                    offsets_global,
                    write_ptr,
                    self._data_offset,
                    V,
                )


    @ti.kernel
    def _set_mass_kernel(
        self,
        inv_masses: ti.template(),
        masses: ti.template(),
        idx: ti.i32,
        inv_mass_val: ti.f32,
        mass_val: ti.f32,
    ):
        inv_masses[idx] = inv_mass_val
        masses[idx] = mass_val


    @ti.kernel
    def _initialize_positions_kernel(
        self,
        mesh_positions: ti.template(),
        sim_data_dofs: ti.template(),
        inv_masses: ti.template(),
        masses: ti.template(),
        offset: ti.i32,
        translation: ti.types.vector(3, ti.f32),
        mass: ti.f32,
    ):
        for i in range(mesh_positions.shape[0]):
            sim_data_dofs[offset + i] = mesh_positions[i] + translation
            inv_masses[offset + i] = 1.0 / mass if mass != 0.0 else -1.0
            masses[offset + i] = mass

    @ti.kernel
    def _append_vertex_adjacency_kernel(
        self,
        offsets_global: ti.template(),
        info_global: ti.template(),
        write_ptr: ti.template(),
        mesh_offsets: ti.template(),
        mesh_indices: ti.template(),
        mesh_weights: ti.template(),
        data_offset: ti.i32,
        V: ti.i32,
        max_degree: ti.i32,
        max_entries: ti.i32,
    ) -> ti.i32:
        overflow = 0
        ti.loop_config(serialize=True)
        for i in range(V):
            start = mesh_offsets[i]
            end = mesh_offsets[i + 1]
            deg = end - start
            if deg > max_degree:
                overflow = 1
            base = write_ptr[None]
            if base + deg > max_entries:
                overflow = 2
            offsets_global[data_offset + i] = base
            for k in range(deg):
                info_idx = base + k
                info_global[info_idx].vertex_adj_indices = data_offset + mesh_indices[start + k]
                info_global[info_idx].vertex_adj_cotan_weights = mesh_weights[start + k]
            write_ptr[None] = base + deg
        offsets_global[data_offset + V] = write_ptr[None]
        return overflow

    @ti.kernel
    def _append_zero_degree_adjacency_kernel(
        self,
        offsets_global: ti.template(),
        write_ptr: ti.template(),
        data_offset: ti.i32,
        V: ti.i32,
    ):
        base = write_ptr[None]
        for i in range(V):
            offsets_global[data_offset + i] = base
        offsets_global[data_offset + V] = base


    def get_color(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        return self._face_color, self._edge_color, self._vertex_color

    def get_mesh(self) -> IMesh:
        return self._mesh

    def get_data_offset(self) -> int:
        return self._data_offset

    def get_num_dofs(self) -> int:
        return self._mesh.get_num_vertices()

    def set_mass(self, local_index: int, mass: float) -> None:
        if local_index < 0 or local_index >= self._mesh.get_num_vertices():
            raise IndexError(f"local_index {local_index} out of range [0, {self._mesh.get_num_vertices()-1}]")

        global_idx = self._data_offset + local_index

        if mass == -1.0:
            # Pinned vertex per spec: inv_mass = 0, mass = -1
            inv_mass_val = 0.0
            mass_val = 1e9
        else:
            if mass <= 0.0:
                raise ValueError("mass must be positive, or -1 for pinned vertex")
            inv_mass_val = 1.0 / float(mass)
            mass_val = float(mass)

        self._set_mass_kernel(
            self._data.get_inv_masses(),
            self._data.get_masses(),
            global_idx,
            inv_mass_val,
            mass_val,
        )

    def add_xpbd_distance_energy(self, stiffness: float):
        """
        Factory method to create and add PBD-style distance constraints (edge springs)
        to the global DistanceEnergy singleton.

        This method requires the associated mesh to provide edge connectivity data.

        Args:
            stiffness (float): The stiffness parameter for this batch of constraints.
        """
        # --- Type and Capability Checking ---
        if not hasattr(self._mesh, 'get_edge_indices'):
            raise AttributeError("The mesh does not provide edge data (IEdgeDataProvider). Cannot create distance energy.")

        # --- Constraint Preparation ---
        edge_indices = self._mesh.get_edge_indices()
        assert edge_indices.shape[0] % 2 == 0
        num_edges = edge_indices.shape[0] // 2
        if num_edges == 0:
            return

        start_idx = self._energy_container.reserve_constraints(num_edges)

        self._add_distance_constraints_kernel(
            self._data.get_dofs(),
            edge_indices,
            self._data_offset,
            stiffness,
            start_idx,
            self._distance_energy,
            self._energy_container,
            num_edges,

        )


    @ti.kernel
    def _add_distance_constraints_kernel(
        self,
        dofs: ti.template(),
        mesh_edge_indices: ti.template(),
        data_offset: ti.i32,
        stiffness: ti.f32,
        start_idx: ti.i32,
        distance_energy: ti.template(),
        energy_container: ti.template(),
        num_edges: ti.i32
    ):
        """
        Taichi kernel to compute rest lengths from the current deformed shape and
        add distance constraints directly to the global energy container.
        """
        for i in range(num_edges):
            idx1_local = mesh_edge_indices[2 * i + 0]
            idx2_local = mesh_edge_indices[2 * i + 1]

            p1_idx_global = idx1_local + data_offset
            p2_idx_global = idx2_local + data_offset

            # Calculate rest distance from the CURRENT positions in the global dofs array
            p1_current = dofs[p1_idx_global]
            p2_current = dofs[p2_idx_global]
            rest_dist = (p1_current - p2_current).norm()
            
            constraint_idx = start_idx + i
            distance_energy.add_one_constraint_func(
                energy_container,
                constraint_idx,
                p1_idx_global,
                p2_idx_global,
                rest_dist,
                stiffness
            )

    def add_pd_strain_energy(self, stiffness: float, singular_min: float, singular_max: float) -> None:
        """
        为每个三角面添加一条 PD Strain 约束：
        - v_indices 来自 `get_surface_indices()`（局部 -> 全局偏移）
        - local_edge(2x2 展平为 vec4) 来自 `get_surface_local_edge_2x2()`
        - stiffness 为该批约束的统一刚度
        """
        if not hasattr(self._mesh, 'get_surface_indices'):
            raise AttributeError("IMesh 必须提供 get_surface_indices() 以创建 PD Strain 约束。")
        if not hasattr(self._mesh, 'get_surface_local_edge_2x2'):
            raise AttributeError("IMesh 必须提供 get_surface_local_edge_2x2() 以创建 PD Strain 约束。")
        if not hasattr(self._mesh, 'get_surface_areas'):
            raise AttributeError("IMesh 必须提供 get_surface_areas() 以创建 PD Strain 约束。")

        surface_indices = self._mesh.get_surface_indices()
        num_faces = surface_indices.shape[0] // 3
        if num_faces == 0:
            return
        surface_areas = self._mesh.get_surface_areas()
        local_edge_2x2 = self._mesh.get_surface_local_edge_2x2()

        start_idx = self._energy_container.reserve_constraints(num_faces)

        self._add_pd_strain_energy_kernel(
            surface_indices,
            local_edge_2x2,
            surface_areas,
            self._data_offset,
            stiffness,
            singular_min,
            singular_max,
            start_idx,
            self._pd_strain_energy,
            self._energy_container,
            num_faces,
        )

    def add_pd_bending_energy(self, stiffness: float):
        V = int(self._mesh.get_num_vertices())
        if V == 0:
            return

        if not hasattr(self._data, 'get_vertex_adjacency'):
            raise AttributeError("ISimulationData must provide get_vertex_adjacency() for PD Bending energy.")
        if not hasattr(self._mesh, 'get_vertex_mixed_voronoi_areas'):
            raise AttributeError("IMesh must provide get_vertex_mixed_voronoi_areas() for PD Bending energy.")

        start_idx = self._energy_container.reserve_constraints(V)
        offsets_global, info_global = self._data.get_vertex_adjacency()
        vertex_areas = self._mesh.get_vertex_mixed_voronoi_areas()
        self._add_pd_bending_energy_kernel(
            self._data.get_dofs(),
            offsets_global,
            info_global,
            self._data_offset,
            stiffness,
            start_idx,
            vertex_areas,
            self._pd_bending_energy,
            self._energy_container,
            V,
        )


    @ti.kernel
    def _add_pd_bending_energy_kernel(
        self,
        dofs: ti.template(),
        offsets_global: ti.template(),
        info_global: ti.template(),
        data_offset: ti.i32,
        stiffness: ti.f32,
        start_idx: ti.i32,
        vertex_areas: ti.template(),
        pd_bending_energy: ti.template(),
        energy_container: ti.template(),
        V: ti.i32,
    ):
        for i in range(V):
            idx_global = data_offset + i
            constraint_idx = start_idx + i
            pd_bending_energy.add_one_constraint_func(
                energy_container,
                constraint_idx,
                idx_global,
                stiffness,
                vertex_areas[i],
                dofs,
                offsets_global,
                info_global.vertex_adj_indices,
                info_global.vertex_adj_cotan_weights,
            )



    @ti.kernel
    def _add_pd_strain_energy_kernel(
        self,
        surface_indices: ti.template(),
        surface_local_edge_2x2: ti.template(),
        surface_areas: ti.template(),
        data_offset: ti.i32,
        stiffness: ti.f32,
        singular_min: ti.f32,
        singular_max: ti.f32,
        start_idx: ti.i32,
        pd_strain_energy: ti.template(),
        energy_container: ti.template(),
        num_faces: ti.i32,
    ):
        for i in range(num_faces):
            a_local = surface_indices[3 * i + 0]
            b_local = surface_indices[3 * i + 1]
            c_local = surface_indices[3 * i + 2]

            a = a_local + data_offset
            b = b_local + data_offset
            c = c_local + data_offset

            v_idx = ti.Vector([a, b, c])
            local_edge = surface_local_edge_2x2[i]
            surface_area = surface_areas[i]

            pd_strain_energy.add_one_constraint_func(
                energy_container,
                start_idx + i,
                v_idx,
                local_edge,
                surface_area,
                stiffness,
                singular_min,
                singular_max
            )
