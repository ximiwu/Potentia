import imp
import math
from typing import List, Optional, Tuple

import taichi as ti
from taichi.math import vec3

from data.base import ISimulationData
from energies.global_energy_container import GlobalEnergyContainer
from energies.distance_energy import DistanceEnergy
from energies.pd_spring_energy import PDSpringEnergy
from energies.pd_bending_energy import PDBendingEnergy
from energies.pd_strain_energy import PDStrainEnergy
from energies.attachment_energy import AttachmentEnergy

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
        self._pd_spring_energy = PDSpringEnergy.get_instance()
        self._pd_bending_energy = PDBendingEnergy.get_instance()
        self._pd_strain_energy = PDStrainEnergy.get_instance()
        self._attachment_energy = AttachmentEnergy.get_instance()
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
        if mass_val >= 0.0:
            masses[idx] = mass_val
        else:
            # masses[idx] = masses[idx] * 10000
            masses[idx] = masses[idx] * 10000



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
            inv_mass_val = 0.0
            mass_val = -1.0
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

    @ti.kernel
    def _add_attachment_constraint_kernel(
        self,
        dofs: ti.template(),
        data_offset: ti.i32,
        idx_local: ti.i32,
        stiffness: ti.f32,
        start_idx: ti.i32,
        attach_energy: ti.template(),
        energy_container: ti.template(),
        attach_pos: ti.types.vector(3, ti.f32),
        use_current: ti.i32,
    ):
        idx_global = data_offset + idx_local
        pos = attach_pos
        if use_current == 1:
            pos = dofs[idx_global]
        attach_energy.add_one_constraint_func(
            energy_container,
            start_idx,
            idx_global,
            pos,
            stiffness,
        )

    def add_attachment_energy(self, stiffness: float, idx: int, attach_pos: ti.Vector([float, float, float]) = None):
        """
        添加固定点约束，将 idx 点固定在 attach_pos 位置。
        如果 attach_pos 为 None，则默认固定在当前位置。
        """
        V = int(self._mesh.get_num_vertices())
        if idx < 0 or idx >= V:
            raise IndexError(f"local_index {idx} out of range [0, {V - 1}]")

        start_idx = self._energy_container.reserve_constraints(1)

        use_current = 1 if attach_pos is None else 0
        pos_vec = vec3(0.0, 0.0, 0.0) if attach_pos is None else vec3(*attach_pos)

        self._add_attachment_constraint_kernel(
            self._data.get_dofs(),
            self._data_offset,
            idx,
            stiffness,
            start_idx,
            self._attachment_energy,
            self._energy_container,
            pos_vec,
            use_current,
        )
        return start_idx

    def add_pd_spring_energy(self, stiffness: float, single_diag: bool = True):
        """
        按点遍历构建 PD 弹簧（结构/剪切/弯曲），适用于 n×n 规则网格：
        - 结构：(i,j)-(i,j+1), (i,j)-(i+1,j)，静长 r
        - 剪切：(i,j)-(i+1,j+1), (i+1,j)-(i,j+1)，静长 sqrt(2)*r
        - 弯曲（跨两格）：对每个点尝试 (i,j)-(i,j+2) 与 (i,j)-(i+2,j)，静长 2*r

        计数：
        - 结构：2 * n * (n - 1)
        - 剪切：2 * (n - 1) * (n - 1)
        - 弯曲：2 * n * (n - 2)
        """
        V = int(self._mesh.get_num_vertices())
        if V == 0:
            return

        # 推断 n（规则方格网），若不满足则提示
        n_f = math.sqrt(float(V))
        n = int(round(n_f))
        if n * n != V or n < 2:
            raise ValueError(
                f"PD 网格弹簧仅支持 n×n 规则网格顶点，当前 V={V} 无法推断有效 n。")

        # 预留约束空间（按实际将要添加的数量，避免过/欠预留）
        struct_cnt = 2 * n * (n - 1)
        shear_cnt = (n - 1) * (n - 1) if single_diag else 2 * (n - 1) * (n - 1)
        bend_cnt = max(2 * n * (n - 2), 0)
        n_springs = struct_cnt + shear_cnt + bend_cnt
        start_idx = self._energy_container.reserve_constraints(n_springs)

        # 读取静止位置用于估计 r（邻接均匀间距），使用 (0,0)-(0,1) 与 (0,0)-(1,0) 的平均
        rest_positions = self._mesh.get_rest_positions()

        # 按点遍历添加三类弹簧（静长以 r 为基准）
        self._add_pd_spring_constraints_on_grid_kernel(
            self._data.get_dofs(),
            rest_positions,
            self._data_offset,
            stiffness,
            start_idx,
            self._pd_spring_energy,
            self._energy_container,
            n,
            1 if single_diag else 0,
        )


    @ti.kernel
    def _add_pd_spring_constraints_on_grid_kernel(
        self,
        dofs: ti.template(),
        rest_positions: ti.template(),
        data_offset: ti.i32,
        stiffness: ti.f32,
        start_idx: ti.i32,
        pd_spring_energy: ti.template(),
        energy_container: ti.template(),
        n: ti.i32,
        single_diag: ti.i32,
    ):
        """
        在 n×n 网格上按点遍历创建 PD 弹簧，索引布局：
        - 结构（水平 -> 垂直）： 2 * n * (n - 1)
          * 水平偏移 base_struct_h = 0，编号 i*(n-1)+j
          * 垂直偏移 base_struct_v = n*(n-1)，编号 j*(n-1)+i
        - 剪切（每格一条对角）： (n - 1)**2
          * base_shear = 2*n*(n-1)
          * diag1: cell_id = i*(n-1)+j
        - 弯曲（跨点连接）： 2 * n * (n - 2)
          * base_bend = base_shear + (n-1)**2
          * 水平：base_bend_h = base_bend，偏移 i*(n-2)+j
          * 垂直：base_bend_v = base_bend + n*(n-2)，偏移 j*(n-2)+i
        """
        base_struct_h = 0
        base_struct_v = n * (n - 1)
        base_shear = 2 * n * (n - 1)
        shear_cnt = (n - 1) * (n - 1) if single_diag == 1 else 2 * (n - 1) * (n - 1)
        base_bend = base_shear + shear_cnt
        base_bend_h = base_bend
        base_bend_v = base_bend + n * (n - 2)

        # 估计基础静长 r（均匀网格），水平与垂直的平均
        r_h = (rest_positions[0] - rest_positions[1]).norm() if n > 1 else 0.0
        r_v = (rest_positions[0] - rest_positions[n]).norm() if n > 1 else 0.0
        r = 0.5 * (r_h + r_v)
        r_shear = ti.sqrt(2.0) * r
        r_bend = 2.0 * r

        for i in range(n):
            for j in range(n):
                p_ij = data_offset + i * n + j

                # 结构：水平 (i,j)-(i,j+1)
                if j < n - 1:
                    p_right = data_offset + i * n + (j + 1)
                    idx_h = start_idx + base_struct_h + i * (n - 1) + j
                    v_idx = ti.Vector([p_ij, p_right])
                    rest_h = r
                    pd_spring_energy.add_one_constraint_func(
                        energy_container,
                        idx_h,
                        v_idx,
                        rest_h,
                        stiffness,
                    )

                # 结构：垂直 (i,j)-(i+1,j)
                if i < n - 1:
                    p_down = data_offset + (i + 1) * n + j
                    idx_v = start_idx + base_struct_v + j * (n - 1) + i
                    v_idx = ti.Vector([p_ij, p_down])
                    rest_v = r
                    pd_spring_energy.add_one_constraint_func(
                        energy_container,
                        idx_v,
                        v_idx,
                        rest_v,
                        stiffness,
                    )

                # 剪切：根据 single_diag 模式生成对角弹簧
                if single_diag == 0:
                    if i < n - 1 and j < n - 1:
                        cell_id = i * (n - 1) + j
                        p_dr = data_offset + (i + 1) * n + (j + 1)
                        idx_d1 = start_idx + base_shear + cell_id
                        v_idx = ti.Vector([p_ij, p_dr])
                        pd_spring_energy.add_one_constraint_func(
                            energy_container,
                            idx_d1,
                            v_idx,
                            r_shear,
                            stiffness * 0.5,
                        )

                        p_dl = data_offset + (i + 1) * n + j
                        p_ur = data_offset + i * n + (j + 1)
                        idx_d2 = start_idx + base_shear + (n - 1) * (n - 1) + cell_id
                        v_idx2 = ti.Vector([p_dl, p_ur])
                        pd_spring_energy.add_one_constraint_func(
                            energy_container,
                            idx_d2,
                            v_idx2,
                            r_shear,
                            stiffness * 0.5,
                        )
                else:
                    if (i % 2 == 1) and (j % 2 == 0):
                        if i < n - 1 and j < n - 1:
                            p_dr = data_offset + (i + 1) * n + (j + 1)
                            cell_id = i * (n - 1) + j
                            idx = start_idx + base_shear + cell_id
                            v_idx = ti.Vector([p_ij, p_dr])
                            pd_spring_energy.add_one_constraint_func(
                                energy_container,
                                idx,
                                v_idx,
                                r_shear,
                                stiffness,
                            )
                        if i > 0 and j < n - 1:
                            p_ur = data_offset + (i - 1) * n + (j + 1)
                            cell_id = (i - 1) * (n - 1) + j
                            idx = start_idx + base_shear + cell_id
                            v_idx = ti.Vector([p_ij, p_ur])
                            pd_spring_energy.add_one_constraint_func(
                                energy_container,
                                idx,
                                v_idx,
                                r_shear,
                                stiffness,
                            )
                        if i < n - 1 and j > 0:
                            p_dl = data_offset + (i + 1) * n + (j - 1)
                            cell_id = i * (n - 1) + (j - 1)
                            idx = start_idx + base_shear + cell_id
                            v_idx = ti.Vector([p_ij, p_dl])
                            pd_spring_energy.add_one_constraint_func(
                                energy_container,
                                idx,
                                v_idx,
                                r_shear,
                                stiffness,
                            )
                        if i > 0 and j > 0:
                            p_ul = data_offset + (i - 1) * n + (j - 1)
                            cell_id = (i - 1) * (n - 1) + (j - 1)
                            idx = start_idx + base_shear + cell_id
                            v_idx = ti.Vector([p_ij, p_ul])
                            pd_spring_energy.add_one_constraint_func(
                                energy_container,
                                idx,
                                v_idx,
                                r_shear,
                                stiffness,
                            )

                # 弯曲：水平 (i,j)-(i,j+2)
                if (j + 2) < n:
                    p_j2 = data_offset + i * n + (j + 2)
                    idx_bh = start_idx + base_bend_h + i * (n - 2) + j
                    v_idx = ti.Vector([p_ij, p_j2])
                    rest_bh = r_bend
                    pd_spring_energy.add_one_constraint_func(
                        energy_container,
                        idx_bh,
                        v_idx,
                        rest_bh,
                        stiffness * 0.25,
                    )

                # 弯曲：垂直 (i,j)-(i+2,j)
                if (i + 2) < n:
                    p_i2 = data_offset + (i + 2) * n + j
                    idx_bv = start_idx + base_bend_v + j * (n - 2) + i
                    v_idx = ti.Vector([p_ij, p_i2])
                    rest_bv = r_bend
                    pd_spring_energy.add_one_constraint_func(
                        energy_container,
                        idx_bv,
                        v_idx,
                        rest_bv,
                        stiffness * 0.25,
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
